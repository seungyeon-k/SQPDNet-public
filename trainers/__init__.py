import os
import torch

from .segmentation_trainer import SegmentationTrainer
from .recognition_trainer import RecognitionTrainer
from .sqpdnet_dynamics_trainer import SQPDNetTrainer
from .baseline_dynamics_trainer import BaselineDynamicsTrainer


def get_trainer(cfg_trainer, device):
    trainer_type = cfg_trainer.get('type', None)

    if trainer_type == 'segmentation':
        trainer = SegmentationTrainer(cfg_trainer, device=device)
    elif trainer_type == 'recognition':
        trainer = RecognitionTrainer(cfg_trainer, device=device)
    elif trainer_type == 'sqpdnet':
        trainer = SQPDNetTrainer(cfg_trainer, device=device)
    elif trainer_type == 'baseline':
        trainer = BaselineDynamicsTrainer(cfg_trainer, device=device)
    else:
        raise NotImplementedError(f"Trainer {trainer_type} not implemented")

    return trainer