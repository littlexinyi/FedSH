import os
import sys
import time
import logging
import copy
import random
from fllib.client.FedProxClient import FedProxClient
from fllib.datasets.base import FederatedDataset, PEDEDataset,PEDE_img_dateset,PEDE_txt_dateset
from fllib.client.base import BaseClient
from fllib.server.base import BaseServer
from fllib.models.base import load_TextImgPersonReidNet_model
from visdom import Visdom
import numpy as np
import torch
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class BaseFL(object):
    ''' BasedFL class coordinates the server and clients in FL.
    Each time the package is imported, a instance of BaseFL will be initilized.
    '''
    def __init__(self):
        self.server = None
        self.client_class = None
        self.config = None
        self.clients_id = None
        self.source_train_dataset = None
        self.source_test_dataset = None
        self.source_dataset = None
        self.test_imgLoader = None
        self.test_txtLoader = None
        self.fl_dataset = None

        self.vis = None
        
        self.global_model = None
        self.exp_name = None

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.client_device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    def init_exp_name(self):
        self.exp_name = '{}_{}_c{}_p{}_le{}_{}_{}_Backbone_{}_PRL_{}'.format(self.config.dataset.data_name, 
                                                self.config.server.model_name, 
                                                self.config.server.clients_num, 
                                                self.config.server.clients_per_round,
                                                self.config.client.local_epoch,
                                                self.config.server.aggregation_rule + '['+ self.config.server.aggregation_detail.type +']',
                                                self.config.client.optimizer.type,
                                                self.config.server.aggregation_detail.Backbone,
                                                self.config.server.aggregation_detail.PRL
                                                )
        self.config.server.records_save_folder = os.path.join(self.config.server.records_save_folder, self.exp_name)

        if not os.path.exists(self.config.server.records_save_folder):
            os.makedirs(self.config.server.records_save_folder)


    def init_config(self, config):
        '''Initialize the configuration from yaml files or dict from the code

        Args:
            config: input configurations: dict or yaml file
        
        Return:
            configurations
        '''
        self.config = config
        self.init_exp_name()
        OmegaConf.save(self.config, f=f'{self.config.server.records_save_folder}/param.yaml')

        logger.info('Configurations loaded.')

    def init_fl_dataset(self, fl_dataset=None):
        if fl_dataset is not None:
            self.fl_dataset = fl_dataset
        else:
            self.source_train_dataset = PEDEDataset(self.config)
            # self.init_source_dataset(source_train_dataset)
            trainset = self.source_train_dataset

     
            test_imgset = PEDE_img_dateset(self.config)
            test_txtset = PEDE_txt_dateset(self.config)
            self.clients_id = self.init_clients_id()
            self.fl_dataset = FederatedDataset(data_name=self.config.dataset.data_name,
                                                trainset=trainset,
                                                test_imgset=test_imgset,
                                                test_txtset=test_txtset,
                                                simulated=self.config.dataset.simulated,
                                                simulated_root=self.config.dataset.simulated_root,
                                                distribution_type=self.config.dataset.distribution_type,
                                                clients_id=self.clients_id,
                                                class_per_client=self.config.dataset.class_per_client,
                                                alpha=self.config.dataset.alpha,
                                                min_size=self.config.dataset.min_size
                                                )

        self.test_imgLoader, self.test_txtLoader = self.fl_dataset.get_dataloader(client_id=None, 
                                                            batch_size=self.config.client.test_batch_size,
                                                            istrain=False) 

        logger.info('FL dataset distributed successfully.')

        return self.fl_dataset


    def init_server(self, current_round=0):
        logger.info('Server initialization.')
        self.server = BaseServer(config=self.config, 
                                clients=self.clients_id, 
                                client_class=self.client_class,
                                global_model=self.global_model,
                                fl_dataset=self.fl_dataset, 
                                test_imgset=self.test_imgLoader, 
                                test_txtset=self.test_txtLoader,
                                current_round=current_round,   
                                device=self.device,
                                records_save_filename=self.config.trial_name,
                                vis=self.vis
                                )
     
        

    def init_clients_id(self):
        self.clients_id = []
        for c in range(self.config.server.clients_num):
            self.clients_id.append("clients%05.0f" % c)
        return self.clients_id


    def init_clients(self):
        logger.info('Clients initialization.')

        if(self.config.client.optimizer.type == 'FedProx'):
            logger.info('FedProx Clients initialization.')
            self.client_class = FedProxClient(config=self.config, device=self.device)
        else:
            logger.info('Base Clients initialization.')
            self.client_class = BaseClient(config=self.config, device=self.device)
        
        return self.client_class

    def init_global_model(self, global_model=None,config=None):
        logger.info('Global model initialization.')
        if global_model is not None:
            self.global_model = copy.deepcopy(global_model)
        else:
            self.global_model = load_TextImgPersonReidNet_model(config=self.config, device = self.device)
            

        return self.global_model

    
    def init_visualization(self, vis=None):
        if vis is not None:
            self.vis = vis
        else:
            self.vis = Visdom()


        
    def init_fl(self, config=None, global_model=None, fl_dataset=None, current_round=0):
        self.init_config(config=config)   

        if self.config.resume:
            if (global_model is None) and (current_round == 0):
                global_model, current_round = self.load_checkpoint()

        self.init_global_model(global_model=global_model,config=config)

        self.init_fl_dataset(fl_dataset=fl_dataset)
        self.init_clients()

        if self.config.is_visualization:
            self.init_visualization()

        self.init_server(current_round=current_round)

    def run(self):
        start_time = time.time()

        self.server.multiple_steps()  
        logger.info('Total training time {:.4f}s'.format(time.time() - start_time))

    def load_checkpoint(self):
        checkround_path = self.config.server.records_save_folder + '/' + self.config.round_name
        if os.path.exists(checkround_path):
            checkround = torch.load(checkround_path)
            checkround = checkround['round']
            checkpoint_path = self.config.server.records_save_folder + '/' + self.config.trial_name+ str("_") + str(checkround)
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                global_model = checkpoint['model']
                current_round = checkpoint['round']
                logger.info(f'Checkpoint successfully loaded from {checkpoint_path}' )
            else:
                global_model, current_round = None, 0
        else:
                global_model, current_round = None, 0
        return global_model, current_round

global_fl = BaseFL()

def init_config(config=None):
    """Initialize configuration. 

    Args:
        conf (dict): Configurations.

    Returns:
        omegaconf.dictconfig.DictConfig: Internal configurations managed by OmegaConf.
    """
    here = os.path.abspath(os.path.dirname(__file__))
    config_file = os.path.join(here, 'config_example.yaml')
    return merge_config(config_file, config)   

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

def init_logger(log_level):
    """Initialize internal logger of EasyFL.

    Args:
        log_level (int): Logger level, e.g., logging.INFO, logging.DEBUG
    """
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-6.5s]  %(message)s")
    root_logger = logging.getLogger()

    log_level = logging.INFO if not log_level else log_level
    root_logger.setLevel(log_level)

    file_path = os.path.join(os.getcwd(), "logs")
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_path = os.path.join(file_path, "train" + time.strftime(".%m_%d_%H_%M_%S") + ".log")
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)    



def init(config=None):

    global global_fl
    
    config = init_config(config)

    init_logger(config.log_level)

    # set_all_random_seed(config.seed) 

    global_fl.init_fl(config)

def run():
    global global_fl

    global_fl.run()


