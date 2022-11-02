from .segmentation_metric import SegementationAccuracy
from .motion_prediction_metric import MotionPredictionError
from .scene_flow_metric import FlowError


def get_metric(metric_dict, **kwargs):
    name = metric_dict.pop("type")
    metric_instance = get_metric_instance(name)
    return metric_instance(**metric_dict)


def get_metric_instance(name):
    try:
        return {
            'segmentation': SegementationAccuracy,
            'motion_prediction': MotionPredictionError,
            'flow_error': FlowError,
        }[name]
    except:
        raise ("Metric {} not available".format(name))
