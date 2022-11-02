from loss.sq_loss import SuperquadricLoss
from .motion_prediction_loss import MotionPredictionLoss
from .segmentation_loss import SegmentationLoss
from .dsr_net_loss import DSRNetLoss
from .flow_loss import FlowLoss
from .sq_loss import SuperquadricLoss


def get_loss(cfg_loss, *args, **kwargs):
    name = cfg_loss.type
    loss_instance = get_loss_instance(name)
    return loss_instance(**cfg_loss)

def get_loss_instance(name):
    try:
        return {
            'sq_loss': SuperquadricLoss,
            'segmentation_loss': SegmentationLoss,
            'motion_loss': MotionPredictionLoss,
            'dsr-net_loss': DSRNetLoss,
            'motionnormalizedloss3d': FlowLoss,
            'mse': FlowLoss
        }[name]
    except:
        raise ("Loss {} not available".format(name))
