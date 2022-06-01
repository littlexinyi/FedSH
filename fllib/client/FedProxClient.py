import logging
import copy
import time
import numpy as np
import torch
from fllib.client.base import BaseClient
from torch.autograd import Variable
import gc

logger = logging.getLogger(__name__)

CLIENT_LOSS = 'train_loss'

max_norm = 10

class FedProxClient(BaseClient):

    def __init__(self, config, device):
        super(FedProxClient, self).__init__(config, device)


    def train(self, client_id, local_trainset):
        ''' The local training process of FedProx
        '''
        start_time = time.time()
        id_loss_fun_global, id_loss_fun_local, id_loss_fun_non_local, cr_loss_fun, optimizer = self.train_preparation()
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.config.client.epoch_decay)

        last_global_model = copy.deepcopy(self.local_model)
        for e in range(self.config.client.local_epoch):
            id_losses = []
            ranking_losses = []
            for imgs, labels, caption_code, caption_length, caption_code_cr, caption_length_cr in local_trainset:
                imgs, labels, caption_code, caption_length, caption_code_cr, caption_length_cr = Variable(imgs.to(self.device)), Variable(labels.to(self.device)), Variable(caption_code.to(self.device)), caption_length.to(self.device), Variable(caption_code_cr.to(self.device)), caption_length_cr.to(self.device)
                
                optimizer.zero_grad()

                img_global, img_local, img_non_local, txt_global, txt_local, txt_non_local = self.local_model(imgs, caption_code, caption_length)

                # txt_global_cr, txt_local_cr, txt_non_local_cr = self.local_model.module.txt_embedding(caption_code_cr, caption_length_cr)
                txt_global_cr, txt_local_cr, txt_non_local_cr = self.local_model.txt_embedding(caption_code_cr, caption_length_cr)
                
                proximal_term = 0.0

                for w, w_t in zip(self.local_model.parameters(), last_global_model.parameters()):
                    proximal_term = proximal_term + (w - w_t).norm(2) 

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
                loss = (id_loss + ranking_loss) + (self.config.client.optimizer.mu / 2) * proximal_term
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.local_model.parameters(), max_norm=max_norm)
                optimizer.step()
                # batch_loss.append(loss.item())
                id_losses.append(id_loss.item())
                ranking_losses.append(ranking_loss.item())

            current_epoch_id_loss = np.mean(id_losses)
            current_epoch_ranking_loss = np.mean(ranking_losses)


            scheduler.step()

            logger.info('Client: {}, local epoch: {}, id_los: {:.4f}, ranking_loss:{:.4f}'.format(client_id, e, current_epoch_id_loss, current_epoch_ranking_loss))
            
            gc.collect()
            torch.cuda.empty_cache()
        train_time = time.time() - start_time
        logger.info('Client: {}, training {:.4f}s'.format(client_id, train_time))
