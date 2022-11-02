import os
import time
import torch
import numpy as np

from functions.utils import averageMeter
from functions.utils_tensorboard import color_pc_segmentation
from metrics import get_metric
from loss import get_loss


class SegmentationTrainer:
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
        best_val_overall_acc = 0

        train_loader, val_loader = (dataloaders["training"], dataloaders["validation"])

        iter = 0
        
        for epoch in range(cfg.n_epoch):
            for x, y in train_loader:
                # training
                iter += 1

                model.train()
                x = x.to(self.device)
                y = y.to(self.device)

                start_ts = time.time()
                train_step_result = model.train_step(x, y=y, optimizer=opt, loss_function=self.loss, clip_grad=self.clip_grad, kwargs=cfg)

                self.time_meter.update(time.time() - start_ts)
                self.loss_meter["train"].update(train_step_result["loss"])

                # record
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
                    for val_x, val_y in val_loader:
                        val_x = val_x.to(self.device)
                        val_y = val_y.to(self.device)

                        val_step_result = model.validation_step(val_x, y=val_y, loss_function=self.loss, kwargs=cfg)
                        self.loss_meter["val"].update(val_step_result["loss"])
                        self.metric.update(val_step_result["gt"].argmax(axis=2), val_step_result["pred"].argmax(axis=2))

                    # record
                    self.record_results(writer, iter, "val", val_step_result)
                    val_loss = self.loss_meter["val"].avg
                    val_acc = self.metric.get_scores()
                    print(f"[Validation] Iter [{iter:d}] Avg Loss: {val_loss:.4f} Average Overall Acc: {val_acc['Overall Acc']:.4f} Average Mean Acc: {val_acc['Mean Acc']:.4f}")

                    # save model
                    if val_acc['Overall Acc'] > best_val_overall_acc:
                        self.save_model(model, logdir, best=True, i_iter=iter, epoch=epoch)
                        print(f"[Validation] Iter [{iter:d}] best model saved {val_acc['Overall Acc']} > {best_val_overall_acc}")
                        best_val_overall_acc = val_acc['Overall Acc']

                    self.metric.reset()

    def record_results(self, writer, i, tag, results):
        # record loss
        writer.add_scalar(f"loss/{tag}_loss", self.loss_meter[tag].avg, i)

        # record segmentation result
        if i % self.cfg["visualize_interval"] == 0:
            pc = results["pc"][0 : self.cfg["visualize_number"]]
            gt = results["gt"][0 : self.cfg["visualize_number"]]
            pred = results["pred"][0 : self.cfg["visualize_number"]]

            pc_colors_gt = color_pc_segmentation(gt)
            pc_colors_pred = color_pc_segmentation(pred)

            writer.add_mesh(f"{tag} gt", vertices=pc, colors=pc_colors_gt, global_step=i)
            writer.add_mesh(f"{tag} pred", vertices=pc, colors=pc_colors_pred, global_step=i)

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