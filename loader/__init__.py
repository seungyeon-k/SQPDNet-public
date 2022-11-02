import torch

from .segmentation_dataset import SegmentationDataset
from .baseline_dataset import BaselineDataset
from .recognition_dataset import SurroundRecognitionDataset

def get_dataloader(split, cfg_data):
    dataset = get_dataset(cfg_data, split)

    # dataloader   
    loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size = cfg_data['batch_size'], 
            num_workers = cfg_data['num_workers'], 
            shuffle = cfg_data['shuffle'])
    
    return loader

def get_dataset(cfg_data, split):
    name = cfg_data['loader']
    dataset = _get_dataset_instance(name)

    return dataset(split, cfg_data)

def _get_dataset_instance(name):
    try:
        return {
            'segmentation': get_segmentation,
            'recognition': get_recognition,
            'sqpdnet': get_baseline,
            'dsr-net': get_baseline,
            'se3-nets': get_baseline,
        }[name]
    except:
        raise ("Dataset {} not available".format(name))

def get_segmentation(split, cfg_data):
    dataset = SegmentationDataset(split, cfg_data)
    return dataset

def get_recognition(split, cfg_data):
    dataset = SurroundRecognitionDataset(split, cfg_data)
    return dataset

def get_baseline(split, cfg_data):
    dataset = BaselineDataset(split, cfg_data)
    return dataset