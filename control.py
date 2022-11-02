import argparse
from omegaconf import OmegaConf
import os.path as osp

from models import get_model
from control.controller import Controller

def load_config(cfg_model):
    if cfg_model.arch == 'sqpdnet':
        for key, cfg_module in cfg_model.items():
            if 'module' in key:
                if 'motion_only' in cfg_module.path.split('/')[-2]:
                    cfg_model[key] = OmegaConf.load(osp.join(cfg_module.path, cfg_module.path.split('/')[-2]+".yml")).model.motion_module
                else:
                    cfg_model[key] = OmegaConf.load(osp.join(cfg_module.path, cfg_module.path.split('/')[-2]+".yml")).model
                cfg_model[key]['pretrained'] = osp.join(cfg_module.path, cfg_module.checkpoint)
    else:
        checkpoint_path = osp.join(cfg_model.path, cfg_model.checkpoint)
        cfg_model = OmegaConf.load(osp.join(cfg_model.path, cfg_model.path.split('/')[-2]+".yml")).model
        cfg_model['pretrained'] = checkpoint_path

    return cfg_model

def run(cfg, args):

    # setup debugging mode
    cfg.controller.debug = args.debug

    # setup device
    device = cfg.device

    # load config files
    cfg.model = load_config(cfg.model)

    # setup model
    model = get_model(cfg.model).to(device)

    # real world
    realworld = cfg.get('realworld', False)
    simulator = cfg.get('simulator', False)

    if not realworld:
        # load objects
        model.env.reset(cfg.objects)
    else:
        # communication parameters
        cfg.controller.ip = args.ip
        cfg.controller.port = args.port

    # setup controller
    controller = Controller(cfg.controller, model, device, realworld=realworld, simulator=simulator)

    # control
    controller.control()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--device', default=0)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--ip', type=str)
    parser.add_argument('--port', type=int)

    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    if args.device == 'cpu':
        cfg.device = 'cpu'
    else:
        cfg.device = f'cuda:{args.device}'

    config_basename = osp.basename(args.config).split('.')[0]

    run(cfg, args)
    