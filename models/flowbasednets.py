import torch
import torch.nn as nn
import numpy as np

from models.se3nets.flownets import FlowNet
from models.se3nets.ctrlnets import MultiStepSE3PoseModel
from models.se3nets.se3nets import SE3Model
from data_generation.utils import project_pts_to_2d


class FlowBasedNets(nn.Module):
    def __init__(self, *args, **kwargs):
        super(FlowBasedNets, self).__init__()
        act_dim = kwargs.get('act_dim', 4)
        num_se3 = kwargs.get('num_se3', 8)
        type = kwargs.get('type', 'flow')
        se3_type = kwargs.get('se3_type', 'se3aa')

        assert type in ['flow', 'se3', 'se3pose']
        self.type = type
        if type == 'flow':
            self.model = FlowNet(
                act_dim, 
                init_flow_iden=True)
        elif type == 'se3':
            assert num_se3 is not None
            self.model = SE3Model(
                act_dim, 
                num_se3, 
                init_transse3_iden=True, 
                wide=True, 
                use_wt_sharpening=True,
                se3_type=se3_type)
        elif type == 'se3pose':
            assert num_se3 is not None
            self.model = MultiStepSE3PoseModel(
                act_dim, 
                num_se3,
                init_transse3_iden=True,
                init_posese3_iden=True,
                use_wt_sharpening=True,
                wide=True
                )

    def forward(self, opc, action, mask_2d_old=None, train_iter=0):
        if self.type == 'flow':
            mask = None
            flow = self.model([opc, None, action])
        elif self.type == 'se3':
            flow, [_, mask] = self.model([opc, None, action], mask_2d_old=mask_2d_old, train_iter=train_iter)
        elif self.type == 'se3pose':
            pose0, mask = self.model.forward_pose_mask([opc, None], train_iter=train_iter)
            deltapose, _ = self.model.forward_next_pose(pose0, action, None, None)
            def func(pc, mask, deltapose):
                batch_size, num_channels, data_height, data_width = pc.size()
                pc_ = torch.cat([
                    pc,
                    torch.ones(batch_size, 1, data_height, data_width).to(pc)
                ], dim=1)

                num_se3 = mask.size()[1]
                assert (num_channels == 3)
                assert (mask.size() == torch.Size([batch_size, num_se3, data_height, data_width]))
                assert (deltapose.size() == torch.Size([batch_size, num_se3, 4, 4])) 

                pc_next_ = torch.einsum('bnij, bnhw, bjhw -> bihw', deltapose, mask, pc_)
                return pc_next_[:, :3, :, :] - pc
            flow = func(opc, mask, deltapose)
        return flow, mask

    def train_step(self, data, optimizer, loss_function, clip_grad=None, device=f'cuda:0', **kwargs):
        opc = data['organized_pc'].to(device)
        action = data['action'].to(device)
        target = data['organized_flow'].to(device)
        train_iter = kwargs.get('train_iter', 0)
        mask_2d_old = data['mask_2d_old'].to(device)

        flow, mask_2d = self(opc, action, train_iter=train_iter)
        if mask_2d is not None:     
            mask_2d = torch.argmax(mask_2d.detach(), dim=1)
        else:
            mask_2d = mask_2d_old
        optimizer.zero_grad()

        # wts = mask_2d_old.clone().detach()
        # wts[wts >= 1] = 1 
        loss = loss_function(flow, target, wts=None)        

        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_grad)
        loss.backward()

        optimizer.step()

        # images_train = depth_img_generator(self, opc, flow, mask_2d)
        # images_true = depth_img_generator(self, opc, target, mask_2d_old)
        
        return {'loss': loss.item(),
                # 'depth_image_target*': images_true['depth_image'],
                # 'depth_image_estimate*': images_train['depth_image'], 
                # 'mask_gt*': images_true['mask_image],
                # 'mask_image*': images_train['mask_image']
        }

    def validation_step(self, data, loss_function, device=f'cuda:0', **kwargs):
        opc = data['organized_pc'].to(device)
        action = data['action'].to(device)
        target = data['organized_flow'].to(device)
        mask_2d_old = data['mask_2d_old'].to(device)
        train_iter = kwargs.get('train_iter', 0)

        flow, _ = self(opc, action, train_iter=train_iter)
        # t12 = t12_and_mask[0]

        # wts = mask_2d_old.clone().detach()
        # wts[wts > 1] = 1 
        loss = loss_function(flow, target, wts=None)
        # t12 = t12.view(-1, 4, 4)

        # eyes = torch.eye(4).unsqueeze(0).repeat(len(t12), 1, 1).to(t12)
        # diff_t12_btw_identity = torch.norm((t12 - eyes).view(len(t12), -1), dim=1).mean()

        return {"loss": loss.item()}#, "diff_t12_btw_identity_": diff_t12_btw_identity.item()}

    def forward_eval(self, data, device=f'cuda:0', **kwargs):
        opc = data['organized_pc'].to(device)
        action = data['action'].to(device)
        flow_target_batch = data['organized_flow'].to(device)
        mask_2d_gt = data['mask_2d_old']
        train_iter = kwargs.get('train_iter', 0)

        batch_size = len(opc)
        
        flow_pred_batch, mask_2d_pred = self(opc, action, train_iter=train_iter)

        flow_pred_batch = flow_pred_batch.detach()
        if mask_2d_pred is not None:
            mask_2d_pred = mask_2d_pred.detach()

        # print(flow_target_batch.shape)
        flow_target_batch = flow_target_batch

        ##### visible flow #####
        visible_flow_target_batch = []
        visible_flow_pred_batch = []

        mask_object_gt_batch = (mask_2d_gt != 0).reshape(len(mask_2d_gt), -1)

        for batch_idx, flow_target, flow_pred, mask_object_gt in zip(range(batch_size), flow_target_batch, flow_pred_batch, mask_object_gt_batch):
            visible_flow_target = 100 * flow_target.reshape(3, -1)[:, mask_object_gt]
            visible_flow_pred = 100 * flow_pred.reshape(3, -1)[:, mask_object_gt]
            # 100 converts measure from m to cm

            visible_flow_target_batch += [visible_flow_target]
            visible_flow_pred_batch += [visible_flow_pred]

        ##### 2d mask #####
        if mask_2d_pred is not None:
            mask_2d_pred = torch.argmax(mask_2d_pred, dim=1)
        else:
            mask_2d_pred = mask_2d_gt

        pc_old_batch = opc.reshape(len(opc), 3, -1)
        pc_new_batch = pc_old_batch + flow_pred_batch.reshape(len(flow_pred_batch), 3, -1)
        labels_batch = mask_2d_pred.reshape(len(mask_2d_pred), -1)

        camera_view_matrix = np.array(self.env.sim.camera_params[1]['camera_view_matrix']).reshape(4, 4).T
        camera_intr = self.env.sim.camera_params[1]['camera_intr']
        image_size = self.env.sim.camera_params[1]['camera_image_size']

        mask_2d_pred_batch = np.zeros([batch_size] + image_size, dtype=int)

        for batch_idx in range(batch_size):
            pc_new = pc_new_batch[batch_idx].detach().cpu()
            labels = labels_batch[batch_idx].cpu().numpy()

            mask_2d_pred = np.zeros(image_size)

            coords_2d = project_pts_to_2d(pc_new, camera_view_matrix, camera_intr)

            y = np.round(coords_2d[0]).astype(np.int)
            x = np.round(coords_2d[1]).astype(np.int)
            depth = coords_2d[2]

            valid_idx = np.logical_and(
                np.logical_and(x >= 0, x < image_size[0]),
                np.logical_and(y >= 0, y < image_size[1]),
            )
            x = x[valid_idx]
            y = y[valid_idx]
            depth = depth[valid_idx]
            labels = labels[valid_idx]

            sort_id = np.argsort(-depth)
            x = x[sort_id]
            y = y[sort_id]
            labels = labels[sort_id]

            mask_2d_pred[x, y] = labels

            mask_2d_pred_batch[batch_idx] = mask_2d_pred

            data.update({
                'visible_flow_target': visible_flow_target_batch,
                'visible_flow_pred': visible_flow_pred_batch,
                'mask_2d_pred': mask_2d_pred_batch,
            })

        return data

    def forward_vis(self, data, device=f'cuda:0', **kwargs):

        opc = data['organized_pc'].to(device)
        action = data['action'].to(device)
        mask_2d_gt = data['mask_2d_old']
        train_iter = kwargs.get('train_iter', 0)

        batch_size = len(opc)

        flow, mask_2d_pred_batch = self(opc, action, train_iter=train_iter)

        flow = flow.detach()

        if mask_2d_pred_batch is not None:
            mask_2d_pred_batch = mask_2d_pred_batch.detach()
            mask_2d_pred_batch = torch.argmax(mask_2d_pred_batch, dim=1)
        else:
            mask_2d_pred_batch = mask_2d_gt

        pc_old_batch = opc.reshape(len(opc), 3, -1)
        pc_new_batch = pc_old_batch + flow.reshape(len(flow), 3, -1)
        labels_batch = mask_2d_pred_batch.reshape(len(mask_2d_pred_batch), -1)

        camera_view_matrix = np.array(self.env.sim.camera_params[1]['camera_view_matrix']).reshape(4, 4).T
        camera_intr = self.env.sim.camera_params[1]['camera_intr']
        image_size = self.env.sim.camera_params[1]['camera_image_size']

        depth_images_batch = {}
        mask_images_batch = {}
        for key in ['old', 'new']:
            depth_images_batch[key] = np.full([batch_size] + image_size, np.inf)
            mask_images_batch[key] = np.zeros([batch_size] + image_size, dtype=int)

        for batch_idx in range(batch_size):
            pc = {}
            pc['old'] = pc_old_batch[batch_idx].cpu()
            pc['new'] = pc_new_batch[batch_idx].cpu()
            labels = labels_batch[batch_idx].cpu().numpy()

            for key in ['old', 'new']:
                coords_2d = project_pts_to_2d(pc[key], camera_view_matrix, camera_intr)

                y = np.round(coords_2d[0]).astype(np.int)
                x = np.round(coords_2d[1]).astype(np.int)
                depth = coords_2d[2]

                if key == 'old':
                    labels_valid = labels
                else:
                    valid_idx = np.logical_and(
                        np.logical_and(x >= 0, x < image_size[0]),
                        np.logical_and(y >= 0, y < image_size[1]),
                    )
                    x = x[valid_idx]
                    y = y[valid_idx]
                    depth = depth[valid_idx]
                    labels_valid = labels[valid_idx]

                sort_id = np.argsort(-depth)
                x = x[sort_id]
                y = y[sort_id]
                labels_valid = labels_valid[sort_id]
                depth = depth[sort_id]

                depth_images_batch[key][batch_idx][x, y] = depth
                mask_images_batch[key][batch_idx][x, y] = labels_valid

        return {
            'depth_image_old': depth_images_batch['old'], 'mask_image_old': mask_images_batch['old'],
            'depth_image_new': depth_images_batch['new'], 'mask_image_new': mask_images_batch['new']
        }
