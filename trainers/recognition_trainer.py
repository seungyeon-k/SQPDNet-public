import os
import time
import torch
import numpy as np
import open3d as o3d

from functions.utils import averageMeter
from functions.utils_tensorboard import pc_with_coordinates, mesh_generator_recog_gt, mesh_generator_recog_pred, two_meshes_to_numpy
from metrics import get_metric
from loss import get_loss


class RecognitionTrainer:
    """Trainer for a conventional iterative training of model"""

    def __init__(self, cfg_trainer, device):
        self.cfg = cfg_trainer
        self.device = device
        self.time_meter = averageMeter()
        self.loss_meter = {'train': averageMeter(), 'val': averageMeter()}

        self.loss = get_loss(cfg_trainer.loss)
        self.show_metric = False
        cfg_metric = cfg_trainer.get('metric', None)
        if cfg_metric is not None:
            self.metric = get_metric(cfg_metric)
            self.show_metric = True

    def train(self, dataloaders, model, opt, writer, lr_cfg=None):
        cfg = self.cfg
        logdir = writer.file_writer.get_logdir()
        best_val_loss = np.inf
        iter = 0

        train_loader, val_loader = (dataloaders['training'], dataloaders['validation'])

        for epoch in range(cfg.n_epoch):
            for x, y, obj_info, po, ori, mean_xyz, diag_len in train_loader:
                # training
                iter += 1

                model.train()
                x = x.to(self.device)
                y = y.to(self.device)

                start_ts = time.time()
                train_step_result = model.train_step(x, y, obj_info, po, ori, mean_xyz, diag_len, optimizer=opt, loss_function=self.loss)
                
                self.time_meter.update(time.time() - start_ts)
                self.loss_meter["train"].update(train_step_result["loss"])

                if iter % cfg.print_interval == 0:
                    self.record_results(writer, iter, "train", train_step_result)
                    print(f"[Training] Iter [{iter:d}] Avg Loss: {self.loss_meter['train'].avg:.4f} Elapsed time: {self.time_meter.sum:.4f}")
                    self.time_meter.reset()

                # save model
                if iter % cfg.save_interval == 0:
                    self.save_model(model, logdir, best=False, i_iter=iter, epoch=epoch)

                if iter % cfg.val_interval == 0:
                    # validation
                    model.eval()
                    for val_x, val_y, val_obj_info, val_po, val_ori, val_mean_xyz, val_diag_len in val_loader:
                        val_x = val_x.to(self.device)
                        val_y = val_y.to(self.device)

                        val_step_result = model.validation_step(val_x, val_y, val_obj_info, val_po, val_ori, val_mean_xyz, val_diag_len, loss_function=self.loss)
                        
                        self.loss_meter["val"].update(val_step_result["loss"])
                        if self.show_metric:
                            pass

                    # record
                    self.record_results(writer, iter, "val", val_step_result)
                    val_loss = self.loss_meter["val"].avg
                    print(f"[Validation] Iter [{iter:d}] Avg Loss: {val_loss:.4f}")

                    # save model
                    if val_loss < best_val_loss:
                        self.save_model(model, logdir, best=True, i_iter=iter, epoch=epoch)
                        print(f"[Validation] Iter [{iter:d}] best model saved {val_loss} <= {best_val_loss}")
                        best_val_loss = val_loss

    def record_results(self, writer, i, tag, results):
        # record loss
        writer.add_scalar(f"loss/{tag}_loss", self.loss_meter[tag].avg, i)

        # record segmentation result
        if i % self.cfg["visualize_interval"] == 0:
            pc = results["pc"][0 : self.cfg["visualize_number"]]
            pc_gt = results["pc_gt"][0 : self.cfg["visualize_number"]]
            output = results["output"][0 : self.cfg["visualize_number"]]
            object_info = results["object_info"][0 : self.cfg["visualize_number"]]
            position = results["position"][0 : self.cfg["visualize_number"]]
            orientation = results["orientation"][0 : self.cfg["visualize_number"]]
            mean_xyz = results["mean_xyz"][0 : self.cfg["visualize_number"]]
            diagonal_len = results["diag_len"][0 : self.cfg["visualize_number"]]

            pc_coordinated, color_pc = pc_with_coordinates(pc)
            pc_gt_coordinated, color_pc_gt = pc_with_coordinates(pc_gt)

            mesh_pred = mesh_generator_recog_pred(output)
            mesh_gt = mesh_generator_recog_gt(object_info, position, orientation, mean_xyz, diagonal_len)
            total_vertices, total_faces, total_colors = two_meshes_to_numpy(mesh_pred, mesh_gt)

            # write to Tensorboard
            writer.add_mesh(f"{tag} pc", vertices=pc_coordinated, colors=color_pc, global_step=i)
            writer.add_mesh(f"{tag} pc_gt", vertices=pc_gt_coordinated, colors=color_pc_gt, global_step=i)
            writer.add_mesh(f"{tag} mesh & mesh_gt", vertices=total_vertices, faces=total_faces, colors=total_colors, global_step=i)

        # record metrics
        if self.show_metric:
            pass

    def save_model(self, model, logdir, best=False, i_iter=None, epoch=None):
        if best:
            pkl_name = "model_best.pkl"
        else:
            if i_iter is not None:
                pkl_name = f"model_iter_{i_iter}.pkl"
            else:
                pkl_name = f"model_epoch_{epoch}.pkl"
        state = {"epoch": epoch, "model_state": model.state_dict(), "iter": i_iter}
        save_path = os.path.join(logdir, pkl_name)
        torch.save(state, save_path)
        print(f"Model saved: {pkl_name}")