import os
import time
#from types import NoneType
import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy

from functions.utils import averageMeter
from functions.utils_tensorboard import meshes_to_numpy, mesh_generator_motion
from loss import get_loss
from metrics import get_metric

class SQPDNetTrainer:
    """Trainer for a conventional iterative training of model"""

    def __init__(self, cfg_trainer, device):
        self.cfg = cfg_trainer
        self.device = device
        self.time_meter = averageMeter()
        self.loss_meter = {"train": averageMeter(), "val": averageMeter()}

        self.loss = get_loss(cfg_trainer.loss)
        self.clip_grad = self.cfg.get('clip_grad', None)
        cfg_metric = cfg_trainer.get("metric", None)
        if cfg_metric is not None:
            self.metric = get_metric(cfg_metric)

    def train(self, dataloaders, model, opt, writer, lr_cfg=None):
        cfg = self.cfg
        logdir = writer.file_writer.get_logdir()
        best_val_loss = np.inf

        if lr_cfg is not None:
            initial_lr = lr_cfg['initial_lr']
            decay_rate = lr_cfg['decay_rate']
            decay_epochs = lr_cfg['decay_epochs']

        train_loader, val_loader = (dataloaders["training"], dataloaders["validation"])

        iteration = 0
        for epoch in range(cfg.n_epoch):

            # adjust learning rate
            if lr_cfg is not None:
                lr = initial_lr * (decay_rate ** (epoch // decay_epochs))
                for param_group in opt.param_groups:
                    param_group['lr'] = lr

            for data in train_loader:
                # training
                iteration += 1

                model.train()

                start_ts = time.time()
                train_step_result = model.train_step(data, optimizer=opt, loss_function=self.loss, device=self.device, clip_grad=self.clip_grad, train_iter=iteration)
                
                self.time_meter.update(time.time() - start_ts)
                self.loss_meter["train"].update(train_step_result["loss"])

                # record
                if iteration % cfg.print_interval == 0:
                    self.record_results(writer, iteration, 'train', train_step_result)
                    print(f"[Training] Iter [{iteration:d}] Avg Loss: {self.loss_meter['train'].avg:.4f} Elapsed time: {self.time_meter.sum:.4f}")
                    self.time_meter.reset()

                # save model
                if iteration % cfg.save_interval == 0:
                    self.save_model(model, logdir, best=False, i_iter=iteration, epoch=epoch)

                # validation
                if iteration % cfg.val_interval == 0:
                    model.eval()
                    val_batch_choice = np.random.choice(len(val_loader))
                    iter_val = 0
                    for val_data in tqdm(val_loader, desc="Validating ... ", leave=False):
                        val_step_result = model.validation_step(val_data, loss_function=self.loss, device=self.device, train_iter=iteration)
                        self.loss_meter["val"].update(val_step_result["loss"])
                        self.metric.update(val_step_result["motion_gt"], val_step_result["motion_pred"])

                        if iter_val == val_batch_choice:
                            val_step_result_visualize = deepcopy(val_step_result)

                        iter_val += 1

                    # record
                    self.record_results(writer, iteration, 'val', val_step_result_visualize)
                    val_loss = self.loss_meter["val"].avg
                    print(f"[Validation] Iter [{iteration:d}] Avg Loss: {val_loss:.4f}")

                    # save model
                    if val_loss < best_val_loss:
                        self.save_model(model, logdir, best=True, i_iter=iteration, epoch=epoch)
                        print(f"[Validation] Iter [{iteration:d}] best model saved {val_loss} <= {best_val_loss}")
                        best_val_loss = val_loss

                    self.metric.reset()

    def record_results(self, writer, i, tag, results):
        # record loss
        writer.add_scalar(f"loss/{tag}_loss", self.loss_meter[tag].avg, i)

        # record segmentation result
        if i % self.cfg["visualize_interval"] == 0:
            scene = results["scene"][0 : self.cfg["visualize_number"]]
            action = results["action"][0 : self.cfg["visualize_number"]]
            motion_pred = results["motion_pred"][0 : self.cfg["visualize_number"]]
            motion_gt = results["motion_gt"][0 : self.cfg["visualize_number"]]

            mesh_pred = mesh_generator_motion(scene, action, motion_pred, moved_color=[0, 0, 1])
            mesh_gt = mesh_generator_motion(scene, action, motion_gt, moved_color=[0, 1, 0])

            vertices_pred, faces_pred, colors_pred = meshes_to_numpy(mesh_pred)
            vertices_gt, faces_gt, colors_gt = meshes_to_numpy(mesh_gt)

            # write to Tensorboard
            writer.add_mesh(f"{tag} prediction", vertices=vertices_pred, faces=faces_pred, colors=colors_pred, global_step=i)
            writer.add_mesh(f"{tag} ground-truth", vertices=vertices_gt, faces=faces_gt, colors=colors_gt, global_step=i)

        # record metrics
        if tag == "val":
            scores = self.metric.get_scores()
            for key, score in scores.items():
                writer.add_scalar(f"metrics/{tag}_{key.replace(' ', '_')}", score, i)

    def save_model(self, model, logdir, best=False, i_iter=None, epoch=None):
        if best:
            pkl_name = "model_best.pkl"
        else:
            pkl_name = f"model_iter_{i_iter}.pkl" if i_iter is not None else f"model_epoch_{epoch}.pkl"
        state = {"epoch": epoch, "model_state": model.state_dict(), "iter": i_iter}
        save_path = os.path.join(logdir, pkl_name)
        torch.save(state, save_path)
        print(f"Model saved: {pkl_name}")
