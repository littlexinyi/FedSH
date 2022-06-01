# import torch.nn as nn
import importlib
import logging
from os import path

from fllib.models.SSAN import TextImgPersonReidNet

logger = logging.getLogger(__name__)
base_package = 'fllib'



def load_model(model_name: str):
    dir_path = path.dirname(path.realpath(__file__))    #__file__返回运行的py所在的目录路径
    model_file = path.join(dir_path, "{}.py".format(model_name))   
    if not path.exists(model_file):
        logger.error("Please specify a valid model.")
    model_path = "{}.models.{}".format(base_package, model_name)
    model_lib = importlib.import_module(model_path) #import model_path这个模块
    model = getattr(model_lib, "TextImgPersonReidNet")  #获取这个模块的TextImgPersonReidNet类为model
   
    return model()


def load_TextImgPersonReidNet_model(config=None, device = None):
    model = TextImgPersonReidNet(config=config, device=device)
    return model