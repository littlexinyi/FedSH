import torch
import logging
import numpy as np
import time
logger = logging.getLogger(__name__)

def mimic_blank_model(model_proto):
    blank_model_dict = dict()
    for name, params in model_proto.items():
        blank_model_dict[name] = torch.zeros_like(params)
    return blank_model_dict


def select_key(pre_key, Backbone):
    pre_key = list(pre_key)
    selected_keys = []
    if(not(Backbone)):  #aggerate half of the backbone(pre-trained model)
        for i in pre_key:
            if ('Image' in i):
                continue                
            else:
                selected_keys.append(i)
        return selected_keys
    else:    
        return pre_key
    

def FedAvg(local_updates, agg_type='equal', dataset = "", Backbone = True):
    '''Aggregate the local updates by using FedAvg

    Args:
        local_updates: the local updates including the data size from the selected clients.
        type: aggregation type in FedAvg, options: equal, weight_by_size
            equal: all the local updates(model parameters directly average)
            weight_by_size: the local updates with a weight, which depends on the local data size.
    Return:
        aggregated model
    '''
    local_models = [local_updates[i]['model'] for i in local_updates.keys()]
    local_datasize = [local_updates[i]['size'] for i in local_updates.keys()]
    weights = [i/sum(local_datasize) for i in local_datasize]

    updates_num = len(local_updates)
    aggregated_model_dict = mimic_blank_model(local_models[0])
    keys = local_models[0].keys()
    selected_keys = select_key(keys, Backbone)
    if(dataset == "CUHK-PEDES"):
        pre_model_path = './fllib/models/pretrained_model_CUHK'
    elif(dataset == "ICFG-PEDES"):
        pre_model_path = './fllib/models/pretrained_model_ICFG'
    pre_model = torch.load(pre_model_path)      #dict

    start_time = time.time()
    with torch.no_grad():
        for name, param in aggregated_model_dict.items():
            if name in selected_keys:
                if agg_type == 'equal':
                    for i in range(updates_num):
                        param = param + torch.div(local_models[i][name], updates_num)
                    aggregated_model_dict[name] = param
                elif agg_type == 'weight_by_size':
                    for i in range(updates_num):
                        param = param + torch.mul(local_models[i][name], weights[i])
                    aggregated_model_dict[name] = param
            else:       
                pre_model[name].requires_grad = False
                aggregated_model_dict[name] = pre_model[name]

    aggregation_time = time.time() - start_time
    logger.info('Aggregation time {:.4f}s'.format(aggregation_time))
    return aggregated_model_dict
    
def reshape_model_param(model_state_dict):
    param_reshape = np.asarray([])
    for _, w in model_state_dict.items():
        param_reshape = np.hstack((param_reshape, w.detach().cpu().numpy().reshape(-1)))  
    return param_reshape

    
                





