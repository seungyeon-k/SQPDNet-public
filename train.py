import argparse
from omegaconf import OmegaConf
from datetime import datetime
import numpy as np
import os.path as osp
import random
import torch
from tensorboardX import SummaryWriter

from loader import get_dataloader
from models import get_model
from trainers import get_trainer
from functions.utils import save_yaml
from optimizers import get_optimizer

def run(cfg, writer):
    # setup seeds
    seed = cfg.get('seed', 1)
    print(f"Running with random seed : {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # setup device
    device = cfg.device
    
    # setup dataloader
    dataloaders = {}
    for split, cfg_data in cfg.data.items():
        if 'num_classes' in cfg.model:
            cfg_data.num_classes = cfg.model.num_classes
        if 'num_primitives' in cfg.model:
            cfg_data.num_primitives = cfg.model.num_primitives
        if 'motion_dim' in cfg.model:
            cfg_data.motion_dim = cfg.model.motion_dim
        if hasattr(cfg.model, 'motion_module'):
            cfg_data.num_primitives = cfg.model.motion_module.num_primitives
            cfg_data.motion_dim = cfg.model.motion_module.motion_dim
        dataloaders[split] = get_dataloader(split, cfg_data)
    
    # Setup model
    model = get_model(cfg.model).to(device)

    # setup trainer
    if 'num_classes' in cfg.model:
        cfg.trainer.metric.num_classes = cfg.model.num_classes
    if 'motion_dim' in cfg.model:
        cfg.trainer.loss.motion_dim = cfg.model.motion_dim
        cfg.trainer.metric.motion_dim = cfg.model.motion_dim
    if hasattr(cfg.model, 'motion_module'):
        cfg.trainer.loss.motion_dim = cfg.model.motion_module.motion_dim
    trainer = get_trainer(cfg.trainer, device)

    # setup optimizer, lr_scheduler and loss function
    if hasattr(model, 'own_optimizer') and model.own_optimizer:
        optimizer = model.get_optimizer(cfg.trainer.optimizer)
    else:
        optimizer = get_optimizer(cfg.trainer.optimizer, model.parameters())

    if hasattr(cfg.trainer, 'lr_scheduler'):
        lr_cfg = cfg.trainer.lr_scheduler
    else:
        lr_cfg = None
    # train
    trainer.train(dataloaders, model, optimizer, writer, lr_cfg=lr_cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--device', default=0)
    parser.add_argument('--logdir', default='train_results/')
    parser.add_argument('--run', default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.device == 'cpu':
        cfg.device = 'cpu'
    else:
        cfg.device = f'cuda:{args.device}'
    
    if args.run is None:
        run_id = datetime.now().strftime('%Y%m%d-%H%M')
    else:
        run_id = args.run

    config_basename = osp.basename(args.config).split('.')[0]

    # setup writer
    logdir = osp.join(args.logdir, config_basename, str(run_id))
    writer = SummaryWriter(logdir=logdir)
    print(f"Result directory: {logdir}")

    # copy config file
    copied_yml = osp.join(logdir, osp.basename(args.config))
    save_yaml(copied_yml, OmegaConf.to_yaml(cfg))
    print(f"Config saved as {copied_yml}")

    run(cfg, writer)
