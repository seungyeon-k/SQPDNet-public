import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
# from pynvrtc.compiler import Program
# from cupy.cuda import function
from collections import namedtuple
import itertools

from data_generation.utils import project_pts_to_2d

class ModelDSR(nn.Module):
    def __init__(self, *args, **kwargs):
        # transform_type options:   None, 'affine', 'se3euler', 'se3aa', 'se3quat', 'se3spquat'
        # motion_type options:      'se3', 'conv'
        # input volume size:        [128, 128, 48]

        super(ModelDSR, self).__init__()
        self.transform_type = kwargs.get('transform_type', 'se3euler')
        self.K = kwargs.get('object_num', 5)
        self.motion_type = kwargs.get('motion_type', 'se3')
        if self.motion_type == 'se3':
            self.loss_types = ['motion', 'mask']
        elif self.motion_type == 'conv':
            self.loss_types = ['motion']

        # modules
        self.forward_warp = Forward_Warp_Cupy.apply
        self.volume_encoder = VolumeEncoder()
        self.feature_decoder = FeatureDecoder()
        if self.motion_type == 'se3':
            self.mask_decoder = MaskDecoder(self.K)
            self.transform_decoder = TransformDecoder(
                transform_type=self.transform_type,
                object_num=self.K - 1
            )
            self.se3 = SE3(self.transform_type)
        elif self.motion_type == 'conv':
            self.motion_decoder = MotionDecoder()
        else:
            raise ValueError('motion_type doesn\'t support ', self.motion_type)

        # initialization
        for m in self.named_modules():
            if isinstance(m[1], nn.Conv3d) or isinstance(m[1], nn.Conv2d):
                nn.init.kaiming_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm3d) or isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

        # const value
        self.grids = torch.stack(torch.meshgrid(
            torch.linspace(0, 127, 128), torch.linspace(0, 127, 128), torch.linspace(0, 47, 48),
            indexing='ij'
        ))
        self.coord_feature = self.grids / torch.tensor([128, 128, 48]).view([3, 1, 1, 1])
        self.grids_flat = self.grids.view(1, 1, 3, 128 * 128 * 48)
        self.zero_vec = torch.zeros([1, 1, 3], dtype=torch.float)
        self.eye_mat = torch.eye(3, dtype=torch.float)

    def train_step(self, data, optimizer, loss_function, device, clip_grad=None, **kwargs):
        tsdf = data['tsdf'].to(device)
        action = data['action'].to(device)
        scene_flow_3d = data['scene_flow_3d'].to(device)
        mask_3d = data['mask_3d_old'].to(device)

        last_s = self.get_init_repr(len(tsdf)).to(action.device)

        output = self.forward(tsdf.unsqueeze(1), last_s, action, None, no_warp=True)

        loss = 0
        batch_order = None
        
        if 'motion' in self.loss_types:
            loss += loss_function.forward_motion(output['motion'], scene_flow_3d)

        if 'mask' in self.loss_types:
            if batch_order is None:
                batch_order = get_batch_order(output['init_logit'], mask_3d)
            loss += loss_function.forward_mask(output['init_logit'], mask_3d, batch_order)

        optimizer.zero_grad()

        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_grad)
        loss.backward()
        optimizer.step()

        return {'loss': loss.item()}

    def validation_step(self, data, loss_function, device, **kwargs):
        tsdf = data['tsdf'].to(device)
        action = data['action'].to(device)
        scene_flow_3d = data['scene_flow_3d'].to(device)
        mask_3d = data['mask_3d_old'].to(device)

        last_s = self.get_init_repr(len(tsdf)).to(action.device)

        output = self.forward(tsdf.unsqueeze(1), last_s, action, None, no_warp=True)

        loss = 0
        batch_order = None
        
        if 'motion' in self.loss_types:
            loss += loss_function.forward_motion(output['motion'], scene_flow_3d)

        if 'mask' in self.loss_types:
            if batch_order is None:
                batch_order = get_batch_order(output['init_logit'], mask_3d)
            loss += loss_function.forward_mask(output['init_logit'], mask_3d, batch_order)

        return {'loss': loss.item()}

    def forward(self, input_volume, last_s=None, input_action=None, input_motion=None, next_mask=False, no_warp=False):
        B, _, S1, S2, S3 = input_volume.size()
        K = self.K
        device = input_volume.device
        output = {}

        input = torch.cat((input_volume, self.coord_feature.expand(B, -1, -1, -1, -1).to(device)), dim=1)
        input = torch.cat((input, last_s), dim=1) # aggregate history

        volume_embedding, cache = self.volume_encoder(input)
        mask_feature = self.feature_decoder(volume_embedding, cache)

        if self.motion_type == 'conv':
            motion = self.motion_decoder(mask_feature, input_action)
            output['motion'] = motion

            return output

        assert(self.motion_type == 'se3')
        logit, mask = self.mask_decoder(mask_feature)
        output['init_logit'] = logit
        transform_param = self.transform_decoder(mask_feature, input_action)

        # trans, pivot: [B, K-1, 3]
        # rot_matrix:   [B, K-1, 3, 3]
        trans_vec, rot_mat = self.se3(transform_param)
        mask_object = torch.narrow(mask, 1, 0, K - 1)
        sum_mask = torch.sum(mask_object, dim=(2, 3, 4))
        heatmap = torch.unsqueeze(mask_object, dim=2) * self.grids.to(device)
        pivot_vec = torch.sum(heatmap, dim=(3, 4, 5)) / torch.unsqueeze(sum_mask, dim=2)

        # [Important] The last one is the background!
        trans_vec = torch.cat([trans_vec, self.zero_vec.expand(B, -1, -1).to(device)], dim=1).unsqueeze(-1)
        rot_mat = torch.cat([rot_mat, self.eye_mat.expand(B, 1, -1, -1).to(device)], dim=1)
        pivot_vec = torch.cat([pivot_vec, self.zero_vec.expand(B, -1, -1).to(device)], dim=1).unsqueeze(-1)

        grids_flat = self.grids_flat.to(device)
        grids_after_flat = rot_mat @ (grids_flat - pivot_vec) + pivot_vec + trans_vec
        motion = (grids_after_flat - grids_flat).view([B, K, 3, S1, S2, S3])

        motion = torch.sum(motion * torch.unsqueeze(mask, 2), 1)

        output['motion'] = motion

        # if no_warp:
        #     output['s'] = mask_feature
        # elif input_motion is not None:
        #     mask_feature_warp = self.forward_warp(
        #         mask_feature,
        #         input_motion,
        #         torch.sum(mask[:, :-1, ], dim=1)
        #     )
        #     output['s'] = mask_feature_warp
        # else:
        #     mask_feature_warp = self.forward_warp(
        #         mask_feature,
        #         motion,
        #         torch.sum(mask[:, :-1, ], dim=1)
        #     )
        #     output['s'] = mask_feature_warp

        # if next_mask:
        #     mask_warp = self.forward_warp(
        #         mask,
        #         motion,
        #         torch.sum(mask[:, :-1, ], dim=1)
        #     )
        #     output['next_mask'] = mask_warp

        return output

    def forward_eval(self, data, device, **kwargs):
        tsdf = data['tsdf'].to(device)
        action = data['action'].to(device)
        scene_flow_3d = data['scene_flow_3d'].to(device)
        mask_3d_old_gt = data['mask_3d_old'].to(device)

        batch_size = len(tsdf)

        last_s = self.get_init_repr(len(tsdf)).to(action.device)

        output = self.forward(tsdf.unsqueeze(1), last_s, action, None, no_warp=True)

        for key, val in output.items():
            output[key] = val.detach()

        flow_target_batch = scene_flow_3d * 0.4
        flow_pred_batch = output['motion'] * 0.4
        # 0.4 converts measure from voxel to cm. The voxel size is 0.4 cm

        flow_target_batch = flow_target_batch.reshape(len(flow_target_batch), flow_target_batch.shape[1], -1)
        flow_pred_batch = flow_pred_batch.reshape(len(flow_pred_batch), flow_pred_batch.shape[1], -1)

        ##### full flow #####
        full_flow_target_batch = []
        full_flow_pred_batch = []

        mask_object_gt_batch = (mask_3d_old_gt != 0).reshape(len(mask_3d_old_gt), -1)

        for batch_idx, flow_target, flow_pred, mask_object_gt in zip(range(batch_size), flow_target_batch, flow_pred_batch, mask_object_gt_batch):
            full_flow_target = flow_target[:, mask_object_gt]
            full_flow_pred = flow_pred[:, mask_object_gt]

            full_flow_target_batch += [full_flow_target]
            full_flow_pred_batch += [full_flow_pred]

        ##### visible flow #####
        visible_flow_target_batch = []
        visible_flow_pred_batch = []

        mask_surface_batch = ((tsdf > -0.99) * (tsdf < 0) * (mask_3d_old_gt > 0))
        mask_surface_batch[..., 0] = 0
        mask_surface_batch = mask_surface_batch.reshape(len(mask_surface_batch), -1)

        for batch_idx, flow_target, flow_pred, mask_surface in zip(range(batch_size), flow_target_batch, flow_pred_batch, mask_surface_batch):
            visible_flow_target = flow_target[:, mask_surface]
            visible_flow_pred = flow_pred[:, mask_surface]

            visible_flow_target_batch += [visible_flow_target]
            visible_flow_pred_batch += [visible_flow_pred]

        ##### 2d, 3d mask #####
        if 'init_logit' in output:
            batch_order = get_batch_order(output['init_logit'], mask_3d_old_gt.squeeze(1))

            logit = torch.cat([
                torch.stack([output['init_logit'][b:b+1, -1]] + [output['init_logit'][b:b+1, i]\
                    for i in batch_order[b]], dim=1) for b in range(len(batch_order))
            ], dim=0)

            mask_3d_old_pred_batch = torch.argmax(logit, dim=1).detach()
        else:
            mask_3d_old_pred_batch = mask_3d_old_gt.squeeze(1)

        camera_view_matrix = np.array(self.env.sim.camera_params[1]['camera_view_matrix']).reshape(4, 4).T
        camera_intr = self.env.sim.camera_params[1]['camera_intr']
        image_size = self.env.sim.camera_params[1]['camera_image_size']

        mask_3d_new_batch = torch.zeros_like(mask_3d_old_pred_batch)
        mask_2d_new_batch = np.zeros([batch_size] + image_size, dtype=int)

        for batch_idx in range(batch_size):
            # 2d mask
            mask_3d_old_pred = mask_3d_old_pred_batch[batch_idx]
            scene_flow_3d_pred = output['motion'][batch_idx]

            pc_old = (torch.ones_like(mask_3d_old_pred).nonzero()[:, [1, 0, 2]] * self.env.voxel_size + self.env.voxel_size / 2)
            pc_old += torch.Tensor(self.env.workspace_bounds[:, 0]).unsqueeze(0).to(device)
            pc_new = pc_old + scene_flow_3d_pred.reshape(len(scene_flow_3d_pred), -1).T[:, [1, 0, 2]] * 0.004

            pc_new_object = pc_new[mask_3d_old_pred.reshape(-1) != 0].cpu()
            labels = mask_3d_old_pred[mask_3d_old_pred != 0].reshape(-1).cpu().numpy()

            mask_2d_pred = np.zeros(image_size)
            
            coords_2d = project_pts_to_2d(pc_new_object.T, camera_view_matrix, camera_intr)

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

            mask_2d_new_batch[batch_idx] = mask_2d_pred

            # 3d mask
            object_ids = torch.unique(mask_3d_old_pred)

            for object_id in object_ids[object_ids != 0]:
                mask_3d_object = mask_3d_old_pred == object_id
                mask_3d_object_idxs = mask_3d_object.nonzero().T

                mask_3d_object_idxs_flow = scene_flow_3d_pred[:, mask_3d_old_pred == object_id]

                mask_3d_object_idxs = torch.round(mask_3d_object_idxs + mask_3d_object_idxs_flow).long()

                valid_idxs = torch.logical_and(
                    torch.logical_and(
                        torch.logical_and(mask_3d_object_idxs[0] >= 0, mask_3d_object_idxs[0] < mask_3d_new_batch.shape[1]),
                        torch.logical_and(mask_3d_object_idxs[1] >= 0, mask_3d_object_idxs[1] < mask_3d_new_batch.shape[2])
                    ),
                    torch.logical_and(mask_3d_object_idxs[2] >= 0, mask_3d_object_idxs[2] < mask_3d_new_batch.shape[3])
                )

                mask_3d_object_idxs = mask_3d_object_idxs[:, valid_idxs]

                mask_3d_new_batch[batch_idx, mask_3d_object_idxs[0], mask_3d_object_idxs[1], mask_3d_object_idxs[2]] = object_id

            data.update({
                'full_flow_target': full_flow_target_batch, 'full_flow_pred': full_flow_pred_batch,
                'visible_flow_target': visible_flow_target_batch, 'visible_flow_pred': visible_flow_pred_batch,
                'mask_2d_pred': mask_2d_new_batch, 'mask_3d_pred': mask_3d_new_batch.cpu().numpy()}
            )
        return data

    def forward_vis(self, data, device, **kwargs):
        tsdf = data['tsdf'].to(device)
        action = data['action'].to(device)
        mask_3d_old_gt = data['mask_3d_old'].to(device)

        batch_size = len(tsdf)

        last_s = self.get_init_repr(len(tsdf)).to(action.device)

        output = self.forward(tsdf.unsqueeze(1), last_s, action, None, no_warp=True)

        for key, val in output.items():
            output[key] = val.detach()

        mask_3d_old_gt = mask_3d_old_gt.unsqueeze(1)

        if 'init_logit' in output:
            batch_order = get_batch_order(output['init_logit'], mask_3d_old_gt.squeeze(1))

            logit = torch.cat([
                torch.stack([output['init_logit'][b:b+1, -1]] + [output['init_logit'][b:b+1, i]\
                    for i in batch_order[b]], dim=1) for b in range(len(batch_order))
            ], dim=0)

            mask_3d_pred_batch = torch.argmax(logit, dim=1).detach()
        else:
            mask_3d_pred_batch = mask_3d_old_gt.squeeze(1)

        camera_view_matrix = np.array(self.env.sim.camera_params[1]['camera_view_matrix']).reshape(4, 4).T
        camera_intr = self.env.sim.camera_params[1]['camera_intr']
        image_size = self.env.sim.camera_params[1]['camera_image_size']

        void_depth_image = np.load("assets/void_depth_image.npy")

        depth_images_batch = {}
        mask_images_batch = {}
        for key in ['old', 'new']:
            depth_images_batch[key] = np.zeros([batch_size] + image_size)
            mask_images_batch[key] = np.zeros([batch_size] + image_size, dtype=int)

        for batch_idx in range(batch_size):
            depth_images = {}
            labels_2d = {}
            mask_images = {}            
            for key in ['old', 'new']:
                depth_images[key] = np.full(image_size, np.inf)
                labels_2d[key] = np.zeros(image_size)
                mask_images[key] = np.zeros(image_size)

            mask_3d_pred = mask_3d_pred_batch[batch_idx]
            scene_flow_3d_pred = output['motion'][batch_idx]

            pc = {}
            pc['old'] = (torch.ones_like(mask_3d_pred).nonzero()[:, [1, 0, 2]] * self.env.voxel_size + self.env.voxel_size / 2)
            pc['old'] += torch.Tensor(self.env.workspace_bounds[:, 0]).unsqueeze(0).to(device)
            pc['new'] = pc['old'] + scene_flow_3d_pred.detach().reshape(len(scene_flow_3d_pred), -1).T[:, [1, 0, 2]] * 0.004
            labels_object = mask_3d_pred[mask_3d_pred != 0].reshape(-1).cpu().numpy()

            for key in ['old', 'new']:
                pc_object = pc[key][mask_3d_pred.reshape(-1) != 0].cpu()
                labels_2d = np.zeros(image_size)

                coords_2d = project_pts_to_2d(pc_object.T, camera_view_matrix, camera_intr)

                y = np.round(coords_2d[0]).astype(np.int)
                x = np.round(coords_2d[1]).astype(np.int)
                depth = coords_2d[2]

                valid_idxs = np.logical_and(
                    np.logical_and(x >= 0, x < image_size[0]),
                    np.logical_and(y >= 0, y < image_size[1]),
                )
                x = x[valid_idxs]
                y = y[valid_idxs]
                depth = depth[valid_idxs]
                labels_object_valid = labels_object[valid_idxs]

                sort_id = np.argsort(-depth)
                x = x[sort_id]
                y = y[sort_id]
                labels_object_valid = labels_object_valid[sort_id]
                depth = depth[sort_id]

                depth_images[key][x, y] = depth
                labels_2d[x, y] = labels_object_valid

                if key == 'new':
                    valid_idxs = (depth_images[key] - void_depth_image) <= 0

                    labels_object_valid = labels_2d[valid_idxs]
                    x, y = np.where(valid_idxs)
                
                mask_images[key][x, y] = labels_object_valid

                depth_images[key] = np.minimum(void_depth_image, depth_images[key])

                depth_images_batch[key][batch_idx] = depth_images[key]
                mask_images_batch[key][batch_idx] = mask_images[key]

        return {
            'depth_image_old': depth_images_batch['old'], 'depth_image_new': depth_images_batch['new'],
            'mask_image_old': mask_images_batch['old'], 'mask_image_new': mask_images_batch['new']
        }

    def get_init_repr(self, batch_size):
        return torch.zeros([batch_size, 8, 128, 128, 48], dtype=torch.float) 

class VolumeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        input_channel = 12
        self.conv00 = ConvBlock3D(input_channel, 16, stride=2, dilation=1, norm=True, relu=True) # 64x64x24

        self.conv10 = ConvBlock3D(16, 32, stride=2, dilation=1, norm=True, relu=True) # 32x32x12
        self.conv11 = ConvBlock3D(32, 32, stride=1, dilation=1, norm=True, relu=True)
        self.conv12 = ConvBlock3D(32, 32, stride=1, dilation=1, norm=True, relu=True)
        self.conv13 = ConvBlock3D(32, 32, stride=1, dilation=1, norm=True, relu=True)

        self.conv20 = ConvBlock3D(32, 64, stride=2, dilation=1, norm=True, relu=True) # 16x16x6
        self.conv21 = ConvBlock3D(64, 64, stride=1, dilation=1, norm=True, relu=True)
        self.conv22 = ConvBlock3D(64, 64, stride=1, dilation=1, norm=True, relu=True)
        self.conv23 = ConvBlock3D(64, 64, stride=1, dilation=1, norm=True, relu=True)

        self.conv30 = ConvBlock3D(64, 128, stride=2, dilation=1, norm=True, relu=True) # 8x8x3
        self.resn31 = ResBlock3D(128, 128)
        self.resn32 = ResBlock3D(128, 128)


    def forward(self, x):
        x0 = self.conv00(x)

        x1 = self.conv10(x0)
        x1 = self.conv11(x1)
        x1 = self.conv12(x1)
        x1 = self.conv13(x1)

        x2 = self.conv20(x1)
        x2 = self.conv21(x2)
        x2 = self.conv22(x2)
        x2 = self.conv23(x2)

        x3 = self.conv30(x2)
        x3 = self.resn31(x3)
        x3 = self.resn32(x3)

        return x3, (x2, x1, x0)


class FeatureDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv00 = ConvBlock3D(128, 64, norm=True, relu=True, upsm=True) # 16x16x6
        self.conv01 = ConvBlock3D(64, 64, norm=True, relu=True)

        self.conv10 = ConvBlock3D(64 + 64, 32, norm=True, relu=True, upsm=True) # 32x32x12
        self.conv11 = ConvBlock3D(32, 32, norm=True, relu=True)

        self.conv20 = ConvBlock3D(32 + 32, 16, norm=True, relu=True, upsm=True) # 64X64X24
        self.conv21 = ConvBlock3D(16, 16, norm=True, relu=True)

        self.conv30 = ConvBlock3D(16 + 16, 8, norm=True, relu=True, upsm=True) # 128X128X48
        self.conv31 = ConvBlock3D(8, 8, norm=True, relu=True)

    def forward(self, x, cache):
        m0, m1, m2 = cache

        x0 = self.conv00(x)
        x0 = self.conv01(x0)

        x1 = self.conv10(torch.cat([x0, m0], dim=1))
        x1 = self.conv11(x1)

        x2 = self.conv20(torch.cat([x1, m1], dim=1))
        x2 = self.conv21(x2)

        x3 = self.conv30(torch.cat([x2, m2], dim=1))
        x3 = self.conv31(x3)

        return x3

class MaskDecoder(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.decoder = nn.Conv3d(8, K, kernel_size=1)

    def forward(self, x):
        logit = self.decoder(x)
        mask = torch.softmax(logit, dim=1)
        return logit, mask


class TransformDecoder(nn.Module):
    def __init__(self, transform_type, object_num):
        super().__init__()
        num_params_dict = {
            'affine': 12,
            'se3euler': 6,
            'se3aa': 6,
            'se3spquat': 6,
            'se3quat': 7
        }
        self.num_params = num_params_dict[transform_type]
        self.object_num = object_num

        self.conv3d00 = ConvBlock3D(8 + 8, 8, stride=2, dilation=1, norm=True, relu=True)  # 64

        self.conv3d10 = ConvBlock3D(8 + 8, 16, stride=2, dilation=1, norm=True, relu=True)  # 32

        self.conv3d20 = ConvBlock3D(16 + 16, 32, stride=2, dilation=1, norm=True, relu=True)  # 16
        self.conv3d21 = ConvBlock3D(32, 32, stride=1, dilation=1, norm=True, relu=True)
        self.conv3d22 = ConvBlock3D(32, 32, stride=1, dilation=1, norm=True, relu=True)
        self.conv3d23 = ConvBlock3D(32, 64, stride=1, dilation=1, norm=True, relu=True)

        self.conv3d30 = ConvBlock3D(64, 128, stride=2, dilation=1, norm=True, relu=True)  # 8

        self.conv3d40 = ConvBlock3D(128, 128, stride=2, dilation=1, norm=True, relu=True)  # 4

        self.conv3d50 = nn.Conv3d(128, 128, kernel_size=(4, 4, 2))


        self.conv2d10 = ConvBlock2D(8, 64, stride=2, norm=True, relu=True)  # 64
        self.conv2d11 = ConvBlock2D(64, 64, stride=1, dilation=1, norm=True, relu=True)
        self.conv2d12 = ConvBlock2D(64, 64, stride=1, dilation=1, norm=True, relu=True)
        self.conv2d13 = ConvBlock2D(64, 64, stride=1, dilation=1, norm=True, relu=True)
        self.conv2d14 = ConvBlock2D(64, 8, stride=1, dilation=1, norm=True, relu=True)

        self.conv2d20 = ConvBlock2D(64, 128, stride=2, norm=True, relu=True)  # 32
        self.conv2d21 = ConvBlock2D(128, 128, stride=1, dilation=1, norm=True, relu=True)
        self.conv2d22 = ConvBlock2D(128, 128, stride=1, dilation=1, norm=True, relu=True)
        self.conv2d23 = ConvBlock2D(128, 128, stride=1, dilation=1, norm=True, relu=True)
        self.conv2d24 = ConvBlock2D(128, 16, stride=1, dilation=1, norm=True, relu=True)

        self.mlp = MLP(
            input_dim=128,
            output_dim=self.num_params * self.object_num,
            hidden_sizes=[512, 512, 512, 512],
            hidden_nonlinearity=F.leaky_relu
        )

    def forward(self, feature, action):
        # feature: [B, 8, 128, 128, 48]
        # action:  [B, 8,  128, 128]

        feature0 = self.conv3d00(torch.cat([feature, action], dim=1))

        action_compressed = torch.max(action, dim=4)[0]

        action1 = self.conv2d10(action_compressed)
        action1 = self.conv2d11(action1)
        action1 = self.conv2d12(action1)
        action1 = self.conv2d13(action1)

        action_embedding1 = self.conv2d14(action1)
        action_embedding1 = torch.unsqueeze(action_embedding1, -1).expand([-1, -1, -1, -1, 24])
        feature1 = self.conv3d10(torch.cat([feature0, action_embedding1], dim=1))

        action2 = self.conv2d20(action1)
        action2 = self.conv2d21(action2)
        action2 = self.conv2d22(action2)
        action2 = self.conv2d23(action2)

        action_embedding2 = self.conv2d24(action2)
        action_embedding2 = torch.unsqueeze(action_embedding2, -1).expand([-1, -1, -1, -1, 12])
        feature2 = self.conv3d20(torch.cat([feature1, action_embedding2], dim=1))
        feature2 = self.conv3d21(feature2)
        feature2 = self.conv3d22(feature2)
        feature2 = self.conv3d23(feature2)

        feature3 = self.conv3d30(feature2)
        feature4 = self.conv3d40(feature3)
        feature5 = self.conv3d50(feature4)

        params = self.mlp(feature5.view([-1, 128]))
        params = params.view([-1, self.object_num, self.num_params])

        return params


class SE3(nn.Module):
    def __init__(self, transform_type='affine', has_pivot=False):
        super().__init__()
        rot_param_num_dict = {
            'affine': 9,
            'se3euler': 3,
            'se3aa': 3,
            'se3spquat': 3,
            'se3quat': 4
        }
        self.transform_type = transform_type
        self.rot_param_num = rot_param_num_dict[transform_type]
        self.has_pivot = has_pivot
        self.num_param = rot_param_num_dict[transform_type] + 3
        if self.has_pivot:
            self.num_param += 3

    def forward(self, input):
        B, K, L = input.size()
        if L != self.num_param:
            raise ValueError('Dimension Error!')

        trans_vec = input.narrow(2, 0, 3)
        rot_params = input.narrow(2, 3, self.rot_param_num)
        if self.has_pivot:
            pivot_vec = input.narrow(2, 3 + self.rot_param_num, 3)


        if self.transform_type == 'affine':
            rot_mat = rot_params.view(B, K, 3, 3)
        elif self.transform_type == 'se3euler':
            rot_mat = Se3euler.apply(rot_params)
        elif self.transform_type == 'se3aa':
            rot_mat = Se3aa.apply(rot_params)
        elif self.transform_type == 'se3spquat':
            rot_mat = Se3spquat.apply(rot_params)
        elif self.transform_type == 'se3quat':
            rot_mat = Se3quat.apply(rot_params)

        if self.has_pivot:
            return trans_vec, rot_mat, pivot_vec
        else:
            return trans_vec, rot_mat


class MotionDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3d00 = ConvBlock3D(8 + 8, 8, stride=2, dilation=1, norm=True, relu=True)  # 64

        self.conv3d10 = ConvBlock3D(8 + 8, 16, stride=2, dilation=1, norm=True, relu=True)  # 32

        self.conv3d20 = ConvBlock3D(16 + 16, 32, stride=2, dilation=1, norm=True, relu=True)  # 16

        self.conv3d30 = ConvBlock3D(32, 16, dilation=1, norm=True, relu=True, upsm=True) # 32
        self.conv3d40 = ConvBlock3D(16, 8, dilation=1, norm=True, relu=True, upsm=True) # 64
        self.conv3d50 = ConvBlock3D(8, 8, dilation=1, norm=True, relu=True, upsm=True) # 128
        self.conv3d60 = nn.Conv3d(8, 3, kernel_size=3, padding=1)


        self.conv2d10 = ConvBlock2D(8, 64, stride=2, norm=True, relu=True)  # 64
        self.conv2d11 = ConvBlock2D(64, 64, stride=1, dilation=1, norm=True, relu=True)
        self.conv2d12 = ConvBlock2D(64, 64, stride=1, dilation=1, norm=True, relu=True)
        self.conv2d13 = ConvBlock2D(64, 64, stride=1, dilation=1, norm=True, relu=True)
        self.conv2d14 = ConvBlock2D(64, 8, stride=1, dilation=1, norm=True, relu=True)

        self.conv2d20 = ConvBlock2D(64, 128, stride=2, norm=True, relu=True)  # 32
        self.conv2d21 = ConvBlock2D(128, 128, stride=1, dilation=1, norm=True, relu=True)
        self.conv2d22 = ConvBlock2D(128, 128, stride=1, dilation=1, norm=True, relu=True)
        self.conv2d23 = ConvBlock2D(128, 128, stride=1, dilation=1, norm=True, relu=True)
        self.conv2d24 = ConvBlock2D(128, 16, stride=1, dilation=1, norm=True, relu=True)

    def forward(self, feature, action):
        # feature: [B, 8, 128, 128, 48]
        # action:  [B, 8,  128, 128]

        feature0 = self.conv3d00(torch.cat([feature, action], dim=1))

        action_compressed = action[:, :, :, :, 0]

        action1 = self.conv2d10(action_compressed)
        action1 = self.conv2d11(action1)
        action1 = self.conv2d12(action1)
        action1 = self.conv2d13(action1)

        action_embedding1 = self.conv2d14(action1)
        action_embedding1 = torch.unsqueeze(action_embedding1, -1).expand([-1, -1, -1, -1, 24])
        feature1 = self.conv3d10(torch.cat([feature0, action_embedding1], dim=1))

        action2 = self.conv2d20(action1)
        action2 = self.conv2d21(action2)
        action2 = self.conv2d22(action2)
        action2 = self.conv2d23(action2)

        action_embedding2 = self.conv2d24(action2)
        action_embedding2 = torch.unsqueeze(action_embedding2, -1).expand([-1, -1, -1, -1, 12])
        feature2 = self.conv3d20(torch.cat([feature1, action_embedding2], dim=1))

        feature3 = self.conv3d30(feature2)
        feature4 = self.conv3d40(feature3 + feature1)
        feature5 = self.conv3d50(feature4 + feature0)

        motion_pred = self.conv3d60(feature5)

        return motion_pred


class ConvBlock3D(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1, norm=False, relu=False, pool=False, upsm=False):
        super().__init__()

        self.conv = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=not norm)
        self.norm = nn.BatchNorm3d(planes) if norm else None
        self.relu = nn.LeakyReLU(inplace=True) if relu else None
        self.pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1) if pool else None
        self.upsm = upsm

    def forward(self, x):
        out = self.conv(x)

        out = out if self.norm is None else self.norm(out)
        out = out if self.relu is None else self.relu(out)
        out = out if self.pool is None else self.pool(out)
        out = out if not self.upsm else F.interpolate(out, scale_factor=2, mode='trilinear', align_corners=True)

        return out


class ConvBlock2D(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1, norm=False, relu=False, pool=False, upsm=False):
        super().__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=not norm)
        self.norm = nn.BatchNorm2d(planes) if norm else None
        self.relu = nn.LeakyReLU(inplace=True) if relu else None
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if pool else None
        self.upsm = upsm

    def forward(self, x):
        out = self.conv(x)

        out = out if self.norm is None else self.norm(out)
        out = out if self.relu is None else self.relu(out)
        out = out if self.pool is None else self.pool(out)
        out = out if not self.upsm else F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)

        return out


class ResBlock3D(nn.Module):
    def __init__(self, inplanes, planes, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MLP(nn.Module):
    """
    MLP Model.
    Args:
        input_dim (int) : Dimension of the network input.
        output_dim (int): Dimension of the network output.
        hidden_sizes (list[int]): Output dimension of dense layer(s).
            For example, (32, 32) means this MLP consists of two
            hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        layer_normalization (bool): Bool for using layer normalization or not.
    Return:
        The output torch.Tensor of the MLP
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_sizes,
                 hidden_nonlinearity=F.relu,
                 hidden_w_init=nn.init.xavier_normal_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_normal_,
                 output_b_init=nn.init.zeros_,
                 layer_normalization=False):
        super().__init__()

        self._input_dim = input_dim
        self._output_dim = output_dim
        self._hidden_nonlinearity = hidden_nonlinearity
        self._output_nonlinearity = output_nonlinearity
        self._layer_normalization = layer_normalization
        self._layers = nn.ModuleList()

        prev_size = input_dim

        for size in hidden_sizes:
            layer = nn.Linear(prev_size, size)
            hidden_w_init(layer.weight)
            hidden_b_init(layer.bias)
            self._layers.append(layer)
            prev_size = size

        layer = nn.Linear(prev_size, output_dim)
        output_w_init(layer.weight)
        output_b_init(layer.bias)
        self._layers.append(layer)

    def forward(self, input_val):
        """Forward method."""
        B = input_val.size(0)
        x = input_val.view(B, -1)
        for layer in self._layers[:-1]:
            x = layer(x)
            if self._hidden_nonlinearity is not None:
                x = self._hidden_nonlinearity(x)
            if self._layer_normalization:
                x = nn.LayerNorm(x.shape[1])(x)

        x = self._layers[-1](x)
        if self._output_nonlinearity is not None:
            x = self._output_nonlinearity(x)

        return x


class Forward_Warp_Cupy(Function):
    @staticmethod
    def forward(ctx, feature, flow, mask):
        kernel = '''
        extern "C"
        __global__ void warp_forward(
            const float * im0, // [B, C, W, H, D]
            const float * flow, // [B, 3, W, H, D]
            const float * mask, // [B, W, H, D]
            float * im1, // [B, C, W, H, D]
            float * cnt, // [B, W, H, D]
            const int vol_batch,
            const int vol_dim_x,
            const int vol_dim_y,
            const int vol_dim_z,
            const int feature_dim,
            const int warp_mode //0 (bilinear), 1 (nearest)
        ) {
            // Get voxel index
            int max_threads_per_block = blockDim.x;
            int block_idx = blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x;
            int voxel_idx = block_idx * max_threads_per_block + threadIdx.x;
            
            int voxel_size_product = vol_dim_x * vol_dim_y * vol_dim_z;
            
            // IMPORTANT
            if (voxel_idx >= vol_batch * voxel_size_product) return;
            
            // Get voxel grid coordinates (note: be careful when casting)
            int tmp = voxel_idx;
            
            int voxel_z = tmp % vol_dim_z;
            tmp = tmp / vol_dim_z;
            
            int voxel_y = tmp % vol_dim_y;
            tmp = tmp / vol_dim_y;
            
            int voxel_x = tmp % vol_dim_x;
            int batch = tmp / vol_dim_x;
            
            int voxel_idx_BCWHD = voxel_idx + batch * (voxel_size_product * (feature_dim - 1));
            int voxel_idx_flow = voxel_idx + batch * (voxel_size_product * (3 - 1));
            
            // Main part
            if (warp_mode == 0) {
                // bilinear
                float x_float = voxel_x + flow[voxel_idx_flow];
                float y_float = voxel_y + flow[voxel_idx_flow + voxel_size_product];
                float z_float = voxel_z + flow[voxel_idx_flow + voxel_size_product + voxel_size_product];
                
                int x_floor = x_float;
                int y_floor = y_float;
                int z_floor = z_float;
                
                for(int t = 0; t < 8; t++) {
                    int dx = (t >= 4);
                    int dy = (t - 4 * dx) >= 2;
                    int dz = t - 4 * dx - dy * 2;
                    
                    int x = x_floor + dx;
                    int y = y_floor + dy;
                    int z = z_floor + dz;
                    
                    if (x >= 0 && x < vol_dim_x && y >= 0 && y < vol_dim_y && z >= 0 && z < vol_dim_z) {
                        float weight = mask[voxel_idx];
                        weight *= (dx == 0 ? (x_floor + 1 - x_float) : (x_float - x_floor));
                        weight *= (dy == 0 ? (y_floor + 1 - y_float) : (y_float - y_floor));
                        weight *= (dz == 0 ? (z_floor + 1 - z_float) : (z_float - z_floor));
                        int idx = (((int)batch * vol_dim_x + x) * vol_dim_y + y) * vol_dim_z + z;
                        atomicAdd(&cnt[idx], weight);
                        
                        int idx_BCWHD = (((int)batch * feature_dim * vol_dim_x + x) * vol_dim_y + y) * vol_dim_z + z;
                        
                        for(int c = 0, offset = 0; c < feature_dim; c++, offset += voxel_size_product) {
                            atomicAdd(&im1[idx_BCWHD + offset], im0[voxel_idx_BCWHD + offset] * weight);
                        }
                    }
                    
                }
            } else {
                // nearest
                int x = round(voxel_x + flow[voxel_idx_flow]);
                int y = round(voxel_y + flow[voxel_idx_flow + voxel_size_product]);
                int z = round(voxel_z + flow[voxel_idx_flow + voxel_size_product + voxel_size_product]);
                
                if (x >= 0 && x < vol_dim_x && y >= 0 && y < vol_dim_y && z >= 0 && z < vol_dim_z) {
                    int idx = (((int)batch * vol_dim_x + x) * vol_dim_y + y) * vol_dim_z + z;
                    float mask_weight = mask[voxel_idx];
                    atomicAdd(&cnt[idx], mask_weight);
                    
                    int idx_BCWHD = (((int)batch * feature_dim * vol_dim_x + x) * vol_dim_y + y) * vol_dim_z + z;
                    
                    for(int c = 0, offset = 0; c < feature_dim; c++, offset += voxel_size_product) {
                        atomicAdd(&im1[idx_BCWHD + offset], im0[voxel_idx_BCWHD + offset] * mask_weight);
                    }
                }
            }
        }
        '''
        program = Program(kernel, 'warp_forward.cu')
        ptx = program.compile()
        m = function.Module()
        m.load(bytes(ptx.encode()))
        f = m.get_function('warp_forward')
        Stream = namedtuple('Stream', ['ptr'])
        s = Stream(ptr=torch.cuda.current_stream().cuda_stream)

        B, C, W, H, D = feature.size()
        warp_mode = 0
        n_blocks = np.ceil(B * W * H * D / 1024.0)
        grid_dim_x = int(np.cbrt(n_blocks))
        grid_dim_y = int(np.sqrt(n_blocks / grid_dim_x))
        grid_dim_z = int(np.ceil(n_blocks / grid_dim_x / grid_dim_y))
        assert grid_dim_x * grid_dim_y * grid_dim_z * 1024 >= B * W * H * D

        feature_new = torch.zeros_like(feature)
        cnt = torch.zeros_like(mask)

        f(grid=(grid_dim_x, grid_dim_y, grid_dim_z), block=(1024, 1, 1),
          args=[feature.data_ptr(), flow.data_ptr(), mask.data_ptr(),  feature_new.data_ptr(), cnt.data_ptr(),
                B, W, H, D, C, warp_mode], stream=s)

        eps=1e-3
        cnt = torch.max(cnt, other=torch.ones_like(cnt) * eps)
        feature_new = feature_new / torch.unsqueeze(cnt, 1)

        return feature_new

    @staticmethod
    def backward(ctx, feature_new_grad):
        # Not implemented
        return None, None, None


class Se3euler(Function):
    @staticmethod
    def forward(ctx, input):
        batch_size, num_se3, num_params = input.size()

        rot_params = input.view(batch_size * num_se3, -1)

        # Create rotations about X,Y,Z axes
        # R = Rz(theta3) * Ry(theta2) * Rx(theta1)
        # Last 3 parameters are [theta1, theta2 ,theta3]
        rotx = create_rotx(rot_params[:, 0])  # Rx(theta1)
        roty = create_roty(rot_params[:, 1])  # Ry(theta2)
        rotz = create_rotz(rot_params[:, 2])  # Rz(theta3)

        # Compute Rz(theta3) * Ry(theta2)
        rotzy = torch.bmm(rotz, roty)  # Rzy = R32

        # Compute rotation matrix R3*R2*R1 = R32*R1
        # R = Rz(t3) * Ry(t2) * Rx(t1)
        output = torch.bmm(rotzy, rotx)  # R = Rzyx

        ctx.save_for_backward(input, output, rotx, roty, rotz, rotzy)

        return output.view(batch_size, num_se3, 3, 3)

    @staticmethod
    def backward(ctx, grad_output):
        input, output, rotx, roty, rotz, rotzy = ctx.saved_tensors
        batch_size, num_se3, num_params = input.size()
        grad_output = grad_output.contiguous().view(batch_size * num_se3, 3, 3)

        # Gradient w.r.t Euler angles from Barfoot's book (http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser15.pdf)
        grad_input_list = []
        for k in range(3):
            gradr = grad_output[:, k]  # Gradient w.r.t angle (k)
            vec = torch.zeros(1, 3).type_as(gradr)
            vec[0][k] = 1  # Unit vector
            skewsym = create_skew_symmetric_matrix(vec).view(1, 3, 3).expand_as(output)  # Skew symmetric matrix of unit vector
            if (k == 0):
                Rv = torch.bmm(torch.bmm(rotzy, skewsym), rotx)  # Eqn 6.61c
            elif (k == 1):
                Rv = torch.bmm(torch.bmm(rotz, skewsym), torch.bmm(roty, rotx))  # Eqn 6.61b
            else:
                Rv = torch.bmm(skewsym, output)
            grad_input_list.append(torch.sum(-Rv * grad_output, dim=(1, 2)))
        grad_input = torch.stack(grad_input_list, 1).view(batch_size, num_se3, 3)

        return grad_input


class Se3aa(Function):
    @staticmethod
    def forward(ctx, input):
        batch_size, num_se3, num_params = input.size()
        N = batch_size * num_se3
        eps = 1e-12

        rot_params = input.view(batch_size * num_se3, -1)

        # Get the un-normalized axis and angle
        axis = rot_params.view(N, 3, 1)  # Un-normalized axis
        angle2 = (axis * axis).sum(1).view(N, 1, 1)  # Norm of vector (squared angle)
        angle = torch.sqrt(angle2)  # Angle

        # Compute skew-symmetric matrix "K" from the axis of rotation
        K = create_skew_symmetric_matrix(axis)
        K2 = torch.bmm(K, K)  # K * K

        # Compute sines
        S = torch.sin(angle) / angle
        S.masked_fill_(angle2.lt(eps), 1)  # sin(0)/0 ~= 1

        # Compute cosines
        C = (1 - torch.cos(angle)) / angle2
        C.masked_fill_(angle2.lt(eps), 0)  # (1 - cos(0))/0^2 ~= 0

        # Compute the rotation matrix: R = I + (sin(theta)/theta)*K + ((1-cos(theta))/theta^2) * K^2
        rot = torch.eye(3).view(1, 3, 3).repeat(N, 1, 1).type_as(rot_params)  # R = I
        rot += K * S.expand(N, 3, 3)  # R = I + (sin(theta)/theta)*K
        rot += K2 * C.expand(N, 3, 3)  # R = I + (sin(theta)/theta)*K + ((1-cos(theta))/theta^2)*K^2

        ctx.save_for_backward(input, rot)

        return rot.view(batch_size, num_se3, 3, 3)

    @staticmethod
    def backward(ctx, grad_output):
        input, rot = ctx.saved_tensors
        batch_size, num_se3, num_params = input.size()
        N = batch_size * num_se3
        eps = 1e-12
        grad_output =grad_output.contiguous().view(N, 3, 3)

        rot_params = input.view(batch_size * num_se3, -1)

        axis = rot_params.view(N, 3, 1)  # Un-normalized axis
        angle2 = (axis * axis).sum(1)  # (Bk) x 1 x 1 => Norm of the vector (squared angle)
        nSmall = angle2.lt(eps).sum()  # Num angles less than threshold

        # Compute: v x (Id - R) for all the columns of (Id-R)
        I = torch.eye(3).type_as(input).repeat(N, 1, 1).add(-1, rot)  # (Bk) x 3 x 3 => Id - R
        vI = torch.cross(axis.expand_as(I), I, 1)  # (Bk) x 3 x 3 => v x (Id - R)

        # Compute [v * v' + v x (Id - R)] / ||v||^2
        vV = torch.bmm(axis, axis.transpose(1, 2))  # (Bk) x 3 x 3 => v * v'
        vV = (vV + vI) / (angle2.view(N, 1, 1).expand_as(vV))  # (Bk) x 3 x 3 => [v * v' + v x (Id - R)] / ||v||^2

        # Iterate over the 3-axis angle parameters to compute their gradients
        # ([v * v' + v x (Id - R)] / ||v||^2 _ k) x (R) .* gradOutput  where "x" is the cross product
        grad_input_list = []
        for k in range(3):
            # Create skew symmetric matrix
            skewsym = create_skew_symmetric_matrix(vV.narrow(2, k, 1))

            # For those AAs with angle^2 < threshold, gradient is different
            # We assume angle = 0 for these AAs and update the skew-symmetric matrix to be one w.r.t identity
            if (nSmall > 0):
                vec = torch.zeros(1, 3).type_as(skewsym)
                vec[0][k] = 1  # Unit vector
                idskewsym = create_skew_symmetric_matrix(vec)
                for i in range(N):
                    if (angle2[i].squeeze()[0] < eps):
                        skewsym[i].copy_(idskewsym.squeeze())  # Use the new skew sym matrix (around identity)

            # Compute the gradients now
            grad_input_list.append(torch.sum(torch.bmm(skewsym, rot) * grad_output, dim=(1, 2)))  # [(Bk) x 1 x 1] => (vV x R) .* gradOutput
        grad_input = torch.stack(grad_input_list, 1).view(batch_size, num_se3, 3)

        return grad_input


class Se3spquat(Function):
    @staticmethod
    def forward(ctx, input):
        batch_size, num_se3, num_params = input.size()

        rot_params = input.view(batch_size * num_se3, -1)

        unitquat = create_unitquat_from_spquat(rot_params)

        output = create_rot_from_unitquat(unitquat).view(batch_size, num_se3, 3, 3)

        ctx.save_for_backward(input)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        batch_size, num_se3, num_params = input.size()

        rot_params = input.view(batch_size * num_se3, -1)

        unitquat = create_unitquat_from_spquat(rot_params)

        # Compute dR/dq'
        dRdqh = compute_grad_rot_wrt_unitquat(unitquat)

        # Compute dq'/dq = d(q/||q||)/dq = 1/||q|| (I - q'q'^T)
        dqhdspq = compute_grad_unitquat_wrt_spquat(rot_params)


        # Compute dR/dq = dR/dq' * dq'/dq
        dRdq = torch.bmm(dRdqh, dqhdspq).view(batch_size, num_se3, 3, 3, 3)  # B x k x 3 x 3 x 3

        # Scale by grad w.r.t output and sum to get gradient w.r.t quaternion params
        grad_out = grad_output.contiguous().view(batch_size, num_se3, 3, 3, 1).expand_as(dRdq)  # B x k x 3 x 3 x 3

        grad_input = torch.sum(dRdq * grad_out, dim=(2, 4))  # (Bk) x 3

        return grad_input


class Se3quat(Function):
    @staticmethod
    def forward(ctx, input):
        batch_size, num_se3, num_params = input.size()

        rot_params = input.view(batch_size * num_se3, -1)

        unitquat = F.normalize(rot_params)

        output = create_rot_from_unitquat(unitquat).view(batch_size, num_se3, 3, 3)

        ctx.save_for_backward(input)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        batch_size, num_se3, num_params = input.size()

        rot_params = input.view(batch_size * num_se3, -1)

        unitquat = F.normalize(rot_params)

        # Compute dR/dq'
        dRdqh = compute_grad_rot_wrt_unitquat(unitquat)

        # Compute dq'/dq = d(q/||q||)/dq = 1/||q|| (I - q'q'^T)
        dqhdq = compute_grad_unitquat_wrt_quat(unitquat, rot_params)


        # Compute dR/dq = dR/dq' * dq'/dq
        dRdq = torch.bmm(dRdqh, dqhdq).view(batch_size, num_se3, 3, 3, 4)  # B x k x 3 x 3 x 4

        # Scale by grad w.r.t output and sum to get gradient w.r.t quaternion params
        grad_out = grad_output.contiguous().view(batch_size, num_se3, 3, 3, 1).expand_as(dRdq)  # B x k x 3 x 3 x 4

        grad_input = torch.sum(dRdq * grad_out, dim=(2, 3))  # (Bk) x 3

        return grad_input


# Rotation about the X-axis by theta
# From Barfoot's book: http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser15.pdf (6.7)
def create_rotx(theta):
    N = theta.size(0)
    rot = torch.eye(3).type_as(theta).view(1, 3, 3).repeat(N, 1, 1)
    rot[:, 1, 1] = torch.cos(theta)
    rot[:, 2, 2] = rot[:, 1, 1]
    rot[:, 1, 2] = torch.sin(theta)
    rot[:, 2, 1] = -rot[:, 1, 2]
    return rot


# Rotation about the Y-axis by theta
# From Barfoot's book: http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser15.pdf (6.6)
def create_roty(theta):
    N = theta.size(0)
    rot = torch.eye(3).type_as(theta).view(1, 3, 3).repeat(N, 1, 1)
    rot[:, 0, 0] = torch.cos(theta)
    rot[:, 2, 2] = rot[:, 0, 0]
    rot[:, 2, 0] = torch.sin(theta)
    rot[:, 0, 2] = -rot[:, 2, 0]
    return rot


# Rotation about the Z-axis by theta
# From Barfoot's book: http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser15.pdf (6.5)
def create_rotz(theta):
    N = theta.size(0)
    rot = torch.eye(3).type_as(theta).view(1, 3, 3).repeat(N, 1, 1)
    rot[:, 0, 0] = torch.cos(theta)
    rot[:, 1, 1] = rot[:, 0, 0]
    rot[:, 0, 1] = torch.sin(theta)
    rot[:, 1, 0] = -rot[:, 0, 1]
    return rot


# Create a skew-symmetric matrix "S" of size [B x 3 x 3] (passed in) given a [B x 3] vector
def create_skew_symmetric_matrix(vector):
    # Create the skew symmetric matrix:
    # [0 -z y; z 0 -x; -y x 0]
    N = vector.size(0)
    vec = vector.contiguous().view(N, 3)
    output = vec.new().resize_(N, 3, 3).fill_(0)
    output[:, 0, 1] = -vec[:, 2]
    output[:, 1, 0] = vec[:, 2]
    output[:, 0, 2] = vec[:, 1]
    output[:, 2, 0] = -vec[:, 1]
    output[:, 1, 2] = -vec[:, 0]
    output[:, 2, 1] = vec[:, 0]
    return output


# Compute Unit Quaternion from SP-Quaternion
def create_unitquat_from_spquat(spquat):
    N = spquat.size(0)
    unitquat = spquat.new_zeros([N, 4])
    x, y, z = spquat[:, 0], spquat[:, 1], spquat[:, 2]
    alpha2 = x * x + y * y + z * z  # x^2 + y^2 + z^2
    unitquat[:, 0] = (2 * x) / (1 + alpha2)  # qx
    unitquat[:, 1] = (2 * y) / (1 + alpha2)  # qy
    unitquat[:, 2] = (2 * z) / (1 + alpha2)  # qz
    unitquat[:, 3] = (1 - alpha2) / (1 + alpha2)  # qw

    return unitquat


# Compute the rotation matrix R from a set of unit-quaternions (N x 4):
# From: http://www.tech.plymouth.ac.uk/sme/springerusv/2011/publications_files/Terzakis%20et%20al%202012,%20A%20Recipe%20on%20the%20Parameterization%20of%20Rotation%20Matrices...MIDAS.SME.2012.TR.004.pdf (Eqn 9)
def create_rot_from_unitquat(unitquat):
    # Init memory
    N = unitquat.size(0)
    rot = unitquat.new_zeros([N, 3, 3])

    # Get quaternion elements. Quat = [qx,qy,qz,qw] with the scalar at the rear
    x, y, z, w = unitquat[:, 0], unitquat[:, 1], unitquat[:, 2], unitquat[:, 3]
    x2, y2, z2, w2 = x * x, y * y, z * z, w * w

    # Row 1
    rot[:, 0, 0] = w2 + x2 - y2 - z2  # rot(0,0) = w^2 + x^2 - y^2 - z^2
    rot[:, 0, 1] = 2 * (x * y - w * z)  # rot(0,1) = 2*x*y - 2*w*z
    rot[:, 0, 2] = 2 * (x * z + w * y)  # rot(0,2) = 2*x*z + 2*w*y

    # Row 2
    rot[:, 1, 0] = 2 * (x * y + w * z)  # rot(1,0) = 2*x*y + 2*w*z
    rot[:, 1, 1] = w2 - x2 + y2 - z2  # rot(1,1) = w^2 - x^2 + y^2 - z^2
    rot[:, 1, 2] = 2 * (y * z - w * x)  # rot(1,2) = 2*y*z - 2*w*x

    # Row 3
    rot[:, 2, 0] = 2 * (x * z - w * y)  # rot(2,0) = 2*x*z - 2*w*y
    rot[:, 2, 1] = 2 * (y * z + w * x)  # rot(2,1) = 2*y*z + 2*w*x
    rot[:, 2, 2] = w2 - x2 - y2 + z2  # rot(2,2) = w^2 - x^2 - y^2 + z^2

    return rot


# Compute the derivatives of the rotation matrix w.r.t the unit quaternion
# From: http://www.tech.plymouth.ac.uk/sme/springerusv/2011/publications_files/Terzakis%20et%20al%202012,%20A%20Recipe%20on%20the%20Parameterization%20of%20Rotation%20Matrices...MIDAS.SME.2012.TR.004.pdf (Eqn 33-36)
def compute_grad_rot_wrt_unitquat(unitquat):
    # Compute dR/dq' (9x4 matrix)
    N = unitquat.size(0)
    x, y, z, w = unitquat.narrow(1, 0, 1), unitquat.narrow(1, 1, 1), unitquat.narrow(1, 2, 1), unitquat.narrow(1, 3, 1)
    dRdqh_w = 2 * torch.cat([w, -z, y, z, w, -x, -y, x, w], 1).view(N, 9, 1)  # Eqn 33, rows first
    dRdqh_x = 2 * torch.cat([x, y, z, y, -x, -w, z, w, -x], 1).view(N, 9, 1)  # Eqn 34, rows first
    dRdqh_y = 2 * torch.cat([-y, x, w, x, y, z, -w, z, -y], 1).view(N, 9, 1)  # Eqn 35, rows first
    dRdqh_z = 2 * torch.cat([-z, -w, x, w, -z, y, x, y, z], 1).view(N, 9, 1)  # Eqn 36, rows first
    dRdqh = torch.cat([dRdqh_x, dRdqh_y, dRdqh_z, dRdqh_w], 2)  # N x 9 x 4

    return dRdqh


# Compute the derivatives of a unit quaternion w.r.t a SP quaternion
# From: http://www.tech.plymouth.ac.uk/sme/springerusv/2011/publications_files/Terzakis%20et%20al%202012,%20A%20Recipe%20on%20the%20Parameterization%20of%20Rotation%20Matrices...MIDAS.SME.2012.TR.004.pdf (Eqn 42-45)
def compute_grad_unitquat_wrt_spquat(spquat):
    # Compute scalars
    N = spquat.size(0)
    x, y, z = spquat.narrow(1, 0, 1), spquat.narrow(1, 1, 1), spquat.narrow(1, 2, 1)
    x2, y2, z2 = x * x, y * y, z * z
    s = 1 + x2 + y2 + z2  # 1 + x^2 + y^2 + z^2 = 1 + alpha^2
    s2 = (s * s).expand(N, 4)  # (1 + alpha^2)^2

    # Compute gradient dq'/dspq
    dqhdspq_x = (torch.cat([2 * s - 4 * x2, -4 * x * y, -4 * x * z, -4 * x], 1) / s2).view(N, 4, 1)
    dqhdspq_y = (torch.cat([-4 * x * y, 2 * s - 4 * y2, -4 * y * z, -4 * y], 1) / s2).view(N, 4, 1)
    dqhdspq_z = (torch.cat([-4 * x * z, -4 * y * z, 2 * s - 4 * z2, -4 * z], 1) / s2).view(N, 4, 1)
    dqhdspq = torch.cat([dqhdspq_x, dqhdspq_y, dqhdspq_z], 2)

    return dqhdspq


# Compute the derivatives of a unit quaternion w.r.t a quaternion
def compute_grad_unitquat_wrt_quat(unitquat, quat):
    # Compute the quaternion norms
    N = quat.size(0)
    unitquat_v = unitquat.view(-1, 4, 1)
    norm2 = (quat * quat).sum(1)  # Norm-squared
    norm = torch.sqrt(norm2)  # Length of the quaternion

    # Compute gradient dq'/dq
    # TODO: No check for normalization issues currently
    I = torch.eye(4).view(1, 4, 4).expand(N, 4, 4).type_as(quat)
    qQ = torch.bmm(unitquat_v, unitquat_v.transpose(1, 2))  # q'*q'^T
    dqhdq = (I - qQ) / (norm.view(N, 1, 1).expand_as(I))

    return dqhdq


def get_batch_order(logit_pred, mask_gt):
    batch_order = []
    B, K, S1, S2, S3 = logit_pred.size()
    sum = 0
    for b in range(B):
        all_p = list(itertools.permutations(list(range(K - 1))))
        best_loss, best_p = None, None
        for p in all_p:
            permute_pred = torch.stack(
                [logit_pred[b:b + 1, -1]] + [logit_pred[b:b + 1, i] for i in p],
                dim=1).contiguous()
            cur_loss = nn.CrossEntropyLoss()(permute_pred, mask_gt[b:b + 1]).item()
            if best_loss is None or cur_loss < best_loss:
                best_loss = cur_loss
                best_p = p
        batch_order.append(best_p)
        sum += best_loss
    return batch_order
