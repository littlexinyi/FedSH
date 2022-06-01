import this
from torch import nn
import torch
import csv
import numpy as np
import os
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from fllib.server import test_model
from fllib.datasets.base import PEDE_img_dateset, PEDE_txt_dateset
from fllib.server.visualization import vis_scalar

GLOBAL_ROUND = 'Round'
GLOBAL_R1 = 'Rank-1'
GLOBAL_R5 = 'Rank-5'
GLOBAL_R10 = 'Rank-10'


def save_test_records(self, save_path, save_file_name):
    header = list(self.train_records.keys())
    content = [[r, a, l] for r,a,l in zip(self.train_records[GLOBAL_ROUND], self.train_records[GLOBAL_R1], self.train_records[GLOBAL_R5], self.train_records[GLOBAL_R10])]

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    with open(os.path.join(save_path, save_file_name)+'.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(content)

def merge_config(config_file, config_=None):
    """Load and merge configuration from file and input

    Args:
        file (str): filename of the configuration.
        conf (dict): Configurations.

    Returns:
        omegaconf.dictconfig.DictConfig: Internal configurations managed by OmegaConf.
    """
    config = OmegaConf.load(config_file)
    if config_ is not None:
        config = OmegaConf.merge(config, config_)
    return config

def load_checkpoint(save_path, save_file, curr_round):
    save_file = save_file + str("_") + str(curr_round)
    checkpoint_path = os.path.join(save_path, save_file)
 
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = None
    return checkpoint

def load_config(config_=None):
    curpath = os.path.dirname(os.path.realpath(__file__))
    yamlpath = os.path.join(curpath, '/fllib/config_example.yaml')

    
    return merge_config(yamlpath, config_)

def mimic_blank_model(model_proto):
    blank_model_dict = dict()
    for name, params in model_proto.items():
        blank_model_dict[name] = torch.zeros_like(params)
    return blank_model_dict


confi_ = {
    'dataset':{
        'data_name': "CUHK-PEDES",
        'vocab_size': 5000,       # "CUHK-PEDES"
        'class_num': 11000  

    },
    'server':{
        'aggregation_detail': {
        'PRL': True             #can be changed by yourself
        },
    }
}
config = load_config(confi_)
#the path of your model needed to be tested
save_path = './results/CUHK-PEDES_SSAN_c10_p5_le3_fedavg[equal]_Adam_Backbone_True_PRL_True/'
save_file = 'test_model'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
checkround_path = os.path.join(save_path, "current_round")
total_round = torch.load(checkround_path)
for i in range(0, total_round['round']+1):
    checkpoint = load_checkpoint(save_path, save_file, curr_round=i)
    everyround_model = checkpoint['model']
    test_imgset = PEDE_img_dateset(config)
    test_txtset = PEDE_txt_dateset(config)
    test_imgLoader = DataLoader(dataset = test_imgset, batch_size=config.client.batch_size, shuffle=True)
    test_txtLoader = DataLoader(dataset = test_txtset, batch_size=config.client.batch_size, shuffle=True)
    R1, R5, R10 = test_model.test(config, device, i, everyround_model, test_imgLoader, test_txtLoader, save_path, save_file)
    print(f"round{i}:")
    print(f"R1:{R1}, R5:{R5}, R10:{R10}")
    vis = config.is_visualization
    if vis is not None:
        vis_scalar(vis=vis, figure_name=GLOBAL_R1, scalar_name=GLOBAL_R1, x=i, y=R1)
        vis_scalar(vis=vis, figure_name=GLOBAL_R5, scalar_name=GLOBAL_R5, x=i, y=R5)
        vis_scalar(vis=vis, figure_name=GLOBAL_R10, scalar_name=GLOBAL_R10, x=i, y=R10)
    
