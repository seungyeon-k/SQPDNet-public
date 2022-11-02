import torch

from .motion_prediction_network import MotionPredictionNetwork
from .segmenation_network import SegmentationNetwork
from .dgcnn import DGCNN
from .sqpdnet import SuperquadricMotionPredictionNetwork
from .dsr_net import ModelDSR
from .flowbasednets import FlowBasedNets
from .sqnet import SuperquadricNetwork


def get_model(cfg_model, *args, **kwargs):
    name = cfg_model['arch']

    if cfg_model.get('backbone', False):
        cfg_backbone = cfg_model.pop('backbone')
        backbone = get_backbone_instance(cfg_backbone['arch'])(**cfg_backbone)
    else:
        backbone = None

    model = get_model_instance(name)
    model = model(backbone, **cfg_model)

    if cfg_model.get('pretrained', None):
        pretrained_model_path = cfg_model['pretrained']

        ckpt = torch.load(pretrained_model_path, map_location='cpu')
        model.load_state_dict(ckpt['model_state'])
        model.iter = ckpt['iter']

    return model

def get_model_instance(name):
    try:
        return {
            'sqnet': SuperquadricNetwork,
            'segnet': SegmentationNetwork,
            'motionnet': MotionPredictionNetwork,
            'sqpdnet': SuperquadricMotionPredictionNetwork,
            'dsr-net': ModelDSR,
            'flowbasednets': FlowBasedNets,
        }[name]
    except:
        raise ('Model {} not available'.format(name))

def get_backbone_instance(name):
    try:
        return {
            'dgcnn': DGCNN,
        }[name]
    except:
        raise (f"Backbone {name} not available")
