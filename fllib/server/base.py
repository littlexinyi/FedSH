import copy
import csv
import time
import numpy as np
import torch
import logging
import os
import gc
from fllib.server.aggeration import FedAvg


logger = logging.getLogger(__name__)
class BaseServer(object):
    def __init__(self, config, clients, client_class, global_model, fl_dataset, test_imgset, test_txtset, device, current_round=0, records_save_filename=None,vis=None):
        self.config = config
        self.clients = clients
        self.client_class = client_class
        self.selected_clients = None

        self.global_model = copy.deepcopy(global_model)
        self.aggregated_model_dict = None
      
        self.train_time = 0

        self.fl_dataset = fl_dataset
        self.train_batchsize = config.client.batch_size
        self.text_imgset = test_imgset
        self.test_txtset = test_txtset    

        self.local_updates= {}

        self.device = device
        
        self.current_round = current_round
        self.records_save_filename = records_save_filename
        self.round_name = self.config.round_name

        self.vis = vis


    def one_round(self):
        '''One round training process in the server
        '''
        logger.info('----- Round {}th -----'.format(self.current_round))
        start_time = time.time()
        # clients selection
        self.client_selection(clients=self.clients, clients_per_round=self.config.server.clients_per_round)
        self.client_training()
        self.aggregation()
        self.update_global_model()
        torch.cuda.empty_cache()

        self.train_time = time.time() - start_time
        logger.info('{}th round use {:.4f}s.'.format(self.current_round, self.train_time))

        self.current_round += 1
        return self.global_model

    def multiple_steps(self):
        i = self.current_round
        for _ in range(i, self.config.server.rounds):

            self.one_round()     #every round
            self.save_checkpoint(save_path=self.config.server.records_save_folder, save_file=self.records_save_filename, save_round = self.round_name)
            gc.collect()
            torch.cuda.empty_cache()

        print(f"{self.config.server.records_save_folder} train over!")
        return self.global_model


    def client_selection(self, clients, clients_per_round):
        '''Select the client in one round.
        
        Args:
            clients: list[Object:'BaseClient']. 
            clients_per_round: int;  the number of clients in each round        
        
        Return:
            selected_clients: list[Object:'BaseClient'] with length clients_per_round
        '''
        if clients_per_round >= len(clients):
            logger.warning('Clients for selected are smaller than the required.')
        
        clients_per_round = min(len(clients), clients_per_round)
        if self.config.server.random_select:   #choose clients random or not
            self.selected_clients = np.random.choice(clients, clients_per_round, replace=False) 
        else:
            self.selected_clients = clients[:clients_per_round]
        
        return self.selected_clients

        
    def client_training(self):     
        '''The global model is distributed to these selected clients.
        And the clients start local training.
        '''
        if len(self.selected_clients) > 0:
            for client in self.selected_clients: 
                self.local_updates[client] = {
                    'model': self.client_class.step(global_model=self.global_model, 
                                                    client_id=client, 
                                                    local_trainset=self.fl_dataset.get_dataloader(client_id=client, batch_size=self.train_batchsize)).state_dict(),
                    'size': self.fl_dataset.get_client_datasize(client_id=client)
                }
        
        else:
            logger.warning('No clients in this round')
            self.local_updates = None
        return self.local_updates
        

    def aggregation(self):
        '''Different aggregation methods

        Return:
            The aggregated global model
        '''
        
        aggregation_algorithm = self.config.server.aggregation_rule
        if self.local_updates is None:
            self.aggregated_model_dict = self.global_model.state_dict()
            
        else:
            if aggregation_algorithm == 'fedavg':
                self.aggregated_model_dict = FedAvg(self.local_updates, agg_type=self.config.server.aggregation_detail.type, dataset = self.config.dataset.data_name, Backbone = self.config.server.aggregation_detail.Backbone)
 

        return self.aggregated_model_dict

    def update_global_model(self):
        if self.global_model is not None:
            self.global_model.load_state_dict(self.aggregated_model_dict)
        return self.global_model
        

    def load_loss_function(self):
        if self.config.server.loss_fn == 'cross_entropy':
            return torch.nn.CrossEntropyLoss()
        elif self.config.server.loss_fn == 'mse':
            return torch.nn.MSELoss()
        else: 
            # defualt is cross entropy
            return torch.nn.CrossEntropyLoss().to(self.device)

    def set_global_model(self, model):
        self.global_model = copy.deepcopy(model)

    
    def load_checkpoint(self, save_path, save_file):
        save_file = save_file + str("_") + str(self.current_round - 1)
        checkpoint_path = os.path.join(save_path, save_file)

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            global_model = checkpoint['model']
            logger.info(f'Checkpoint successfully loaded from {checkpoint_path}' )
        else:
            global_model = self.global_model
        return global_model


    def save_checkpoint(self, save_path, save_file, save_round):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        checkpoint = {
            'model': self.global_model,
            'round': self.current_round
        }  
        check_round = {
            'round': self.current_round  - 1
        }
        save_file = save_file + str("_") + str(self.current_round - 1)
        save_round = save_round
        torch.save(checkpoint, os.path.join(save_path, save_file))
        torch.save(check_round, os.path.join(save_path, save_round))



