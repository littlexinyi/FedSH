import copy
import time
import os
import logging
import numpy as np
import torch
import gc
from torch.autograd import Variable
from fllib.loss.Id_loss import Id_Loss
from fllib.loss.RankingLoss import CRLoss
logger = logging.getLogger(__name__)


class BaseClient(object):
    '''The base client class in federated learning
    '''
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.local_model = None

    def download(self, model, **kwargs):    
        '''  Download the global model from the server, the global model might be compressed
        '''

        if self.local_model is not None:
            self.local_model.load_state_dict(model.state_dict())
        else:
            self.local_model = copy.deepcopy(model)


    def train_preparation(self):
        '''The function prepares the basic tools or operations in the local training process
        '''
        id_loss_fun_global = Id_Loss(self.config, 1, self.config.dataset.feature_length).to(self.device)
        id_loss_fun_local = Id_Loss(self.config, self.config.dataset.part, self.config.dataset.feature_length).to(self.device)
        id_loss_fun_non_local = Id_Loss(self.config, self.config.dataset.part, 512).to(self.device)
        cr_loss_fun = CRLoss(self.config, self.device)

        cnn_params = list(map(id, self.local_model.ImageExtract.parameters()))
        other_params = filter(lambda p: id(p) not in cnn_params, self.local_model.parameters())
        other_params = list(other_params)
        other_params.extend(list(id_loss_fun_global.parameters()))      
        other_params.extend(list(id_loss_fun_local.parameters()))
        other_params.extend(list(id_loss_fun_non_local.parameters()))
        param_groups = [{'params': other_params, 'lr': self.config.client.optimizer.lr},
                        {'params': self.local_model.ImageExtract.parameters(), 'lr': self.config.client.optimizer.lr * 0.1}]

        optimizer = torch.optim.Adam(param_groups, betas=(self.config.client.optimizer.adam_alpha, self.config.client.optimizer.adam_beta))
        # 
        self.local_model.train()
        # # parallel
        # self.local_model = torch.nn.DataParallel(self.local_model, device_ids = device_ids)

        self.local_model.to(self.device)
        
        return id_loss_fun_global, id_loss_fun_local, id_loss_fun_non_local, cr_loss_fun, optimizer

    def load_loss_function(self):
        if self.config.client.loss_fn == 'cross_entropy':
            return torch.nn.CrossEntropyLoss()
        elif self.config.client.loss_fn == 'mse':
            return torch.nn.MSELoss()
        else: 
            # defualt is cross entropy
            return torch.nn.CrossEntropyLoss().to(self.device)

    def load_optimizer(self):
        
        if self.config.client.optimizer.type == 'Adam':
            optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.config.client.optimizer.lr)
        elif self.config.client.optimizer.type == 'SGD':
            optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.config.client.optimizer.lr, momentum=self.config.client.optimizer.momentum, weight_decay=self.config.client.optimizer.weight_decay)
        else:
            # defualt is Adam
            optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.config.client.client.optimizer.lr)
        return optimizer


    def train(self, client_id, local_trainset):
        ''' Local training.

        Key variables:
        local_model: the model in the client and prepared to train with the local data
        local_trianset: the local data set stored in the client(local device)
        local_epochs: the iterations of the local training
        optimizer: the local optimizer for the local training

        '''    
        start_time = time.time()
        id_loss_fun_global, id_loss_fun_local, id_loss_fun_non_local, cr_loss_fun, optimizer = self.train_preparation()
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.config.client.epoch_decay)

        for e in range(self.config.client.local_epoch):
            id_losses = []
            ranking_losses = []
            # train_accuracy.reset()
            for imgs, labels, caption_code, caption_length, caption_code_cr, caption_length_cr in local_trainset:
                imgs, labels, caption_code, caption_length, caption_code_cr, caption_length_cr = Variable(imgs.to(self.device)), Variable(labels.to(self.device)), Variable(caption_code.to(self.device)), caption_length.to(self.device), Variable(caption_code_cr.to(self.device)), caption_length_cr.to(self.device)
                
                optimizer.zero_grad()

                img_global, img_local, img_non_local, txt_global, txt_local, txt_non_local = self.local_model(imgs, caption_code, caption_length)
                #该img的负样本对
                # txt_global_cr, txt_local_cr, txt_non_local_cr = self.local_model.module.txt_embedding(caption_code_cr, caption_length_cr)
                txt_global_cr, txt_local_cr, txt_non_local_cr = self.local_model.txt_embedding(caption_code_cr, caption_length_cr)

                id_loss_global = id_loss_fun_global(img_global, txt_global, labels)
                id_loss_local = id_loss_fun_local(img_local, txt_local, labels)
                if(self.config.server.aggregation_detail.PRL):
                    id_loss_non_local = id_loss_fun_non_local(img_non_local, txt_non_local, labels)
                else:
                    id_loss_non_local = 0

                id_loss = id_loss_global + (id_loss_local + id_loss_non_local) * 0.5            #loss one

                cr_loss_global = cr_loss_fun(img_global, txt_global, txt_global_cr, labels, e >= self.config.client.epoch_begin)
                cr_loss_local = cr_loss_fun(img_local, txt_local, txt_local_cr, labels, e >= self.config.client.epoch_begin)
                if(self.config.server.aggregation_detail.PRL):
                    cr_loss_non_local = cr_loss_fun(img_non_local, txt_non_local,
                                                txt_non_local_cr, labels, e >= self.config.client.epoch_begin)
                else:
                    cr_loss_non_local = 0

                ranking_loss = cr_loss_global + (cr_loss_local + cr_loss_non_local) * 0.5       #loss two
                # Loss and model parameters update
                loss = (id_loss + ranking_loss)
                loss.backward()
                optimizer.step()
                id_losses.append(id_loss.item())
                ranking_losses.append(ranking_loss.item())

            current_epoch_id_loss = np.mean(id_losses)                  #ID_loss，for global
            current_epoch_ranking_loss = np.mean(ranking_losses)        #CR_loss， for part

            scheduler.step()

            logger.info('Client: {}, local epoch: {}, id_loss: {:.4f}, ranking_loss:{:.4f}'.format(client_id, e, current_epoch_id_loss, current_epoch_ranking_loss))
            
            gc.collect()
            torch.cuda.empty_cache()
        train_time = time.time() - start_time
        logger.info('Client: {}, training {:.4f}s'.format(client_id, train_time))


    def upload(self):
        '''Upload the local models(compressed, if it is) to the server
        '''
        if(isinstance(self.local_model, torch.nn.DataParallel)):        #如果模型是并行模式
            self.local_model = self.local_model.module      #DataParell转为正常Net后上传

        return self.local_model


    def step(self, global_model, client_id, local_trainset, is_train=True, **kwargs):
        
        self.download(model=global_model, **kwargs)       
        if is_train:
            self.train(client_id=client_id, local_trainset=local_trainset) 
             
        return self.upload()







    



