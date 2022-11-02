import numpy as np
import os.path as osp
import random
import torch
from tensorboardX import SummaryWriter

import argparse
from omegaconf import OmegaConf
from datetime import datetime
from loader import get_dataloader
from functions.utils_tensorboard import meshes_to_numpy, mesh_generator_motion

def run(cfg, writer):
    # Setup seeds
    seed = cfg.get('seed', 1)
    print(f"Running with random seed : {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Setup dataloader
    dataloaders = {}
    for split, cfg_data in cfg.items():
        cfg_data.num_primitives = 5
        cfg_data.motion_dim = '2D'
        dataloaders[split] = get_dataloader(split, cfg_data)

    # write data
    i = 0
    for x, a, y in dataloaders['training']:

        # index
        i += 1
        print(i)

        scene = x[0:3].numpy()
        action = a[0:3].numpy()
        motion_gt = y[0:3].numpy()

        mesh_gt = mesh_generator_motion(scene, action, motion_gt, moved_color=[0, 1, 0])
        gt_vertices, gt_faces, gt_colors = meshes_to_numpy(mesh_gt)

        # write to Tensorboard
        writer.add_mesh(
            f"[{i}] ground-truth",
            vertices=gt_vertices,
            faces=gt_faces,
            colors=gt_colors,
            global_step=0
        )

        if i == 100:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--logdir', default='train_results/')
    parser.add_argument('--run', default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.run is None:
        run_id = datetime.now().strftime('%Y%m%d-%H%M')
    else:
        run_id = args.run 

    config_basename = osp.basename(args.config).split('.')[0]

    # Setup writer
    logdir = osp.join(args.logdir, config_basename, str(run_id))
    writer = SummaryWriter(logdir=logdir)
    print(f"Result directory: {logdir}")

    run(cfg, writer)