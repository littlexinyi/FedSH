from itertools import combinations
import math
import copy
import numpy as np
from collections import defaultdict


def generate_class_comb(num_groups, num_class, num_class_each_comb):
    class_comb = list(combinations(range(num_class), num_class_each_comb))
    np.random.shuffle(class_comb)
    if len(class_comb) <= num_groups:
        for _ in range(num_groups//len(class_comb)):
            class_comb += class_comb
    class_comb = class_comb[:num_groups]

    return class_comb


def shuffle(data_x, data_y):
    num_of_data = len(data_y)
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    index = [i for i in range(num_of_data)]
    np.random.shuffle(index)
    data_x = list(data_x[index])
    data_y = list(data_y[index])
    return data_x, data_y

def size_of_division(num_groups, size):
    if isinstance(num_groups, int):
        num_per_group = [size // num_groups] * num_groups
    elif isinstance(num_groups, list):
        num_per_group = [math.floor(size * w) for w in num_groups]

    for i in np.random.choice(len(num_per_group), size - sum(num_per_group), replace=False):
        num_per_group[i] += 1
    
    return num_per_group
       

def equal_division(num_groups, data_x, data_y=None):

    if data_y is not None:
        assert (len(data_x) == len(data_y))
        data_x, data_y = shuffle(data_x, data_y)
    else:
        np.random.shuffle(data_x)

    data_num = len(data_x)
    assert data_num > 0

    num_per_group = size_of_division(num_groups, data_num)
     
    div_data_x, div_data_y = [], []
    base_index = 0
    for i in num_per_group:
        div_data_x.append(data_x[base_index: base_index + i])
        if data_y is not None:
            div_data_y.append(data_y[base_index: base_index + i])
        base_index += i

    return div_data_x, div_data_y


def quantity_division(group_weights, data_x, data_y=None):
    # check the sum of the group weights should be 1
    assert (round(sum(group_weights), 4) == 1)

    # shuffle the data
    if data_y is not None:
        assert (len(data_x) == len(data_y))
        data_x, data_y = shuffle(data_x, data_y)
    else:
        np.random.shuffle(data_x)
    
    # get and check the data num should larger than 0
    data_num = len(data_x)
    assert data_num > 0

    # get the number of data in each group
    num_per_group = size_of_division(group_weights, data_num)
    
    div_data_x, div_data_y = [], []
    base_index = 0
    for i in num_per_group:
        div_data_x.append(data_x[base_index: base_index + i])
        if data_y is not None:
            div_data_y.append(data_y[base_index: base_index + i])
        base_index += i 
    return div_data_x, div_data_y


def iid(clients_id, data_idxs_dict):
    '''Partition dataset into multiple clients with iid.
    The data size is equal(or the difference is less than 1).
    all the class should be randomly distributed in each client

    Args: 
        clients_num: the number of clients.
        data_idxs_dict: the data index dictionary. 
            The structure is:
                {
                    'label_0': [...],
                    'label_1': [...],
                    ...
                }
    '''

    clients_data = defaultdict(list)
    clients_num = len(clients_id)

    # get the how many class in the dataset
    class_num = len(data_idxs_dict)

    data_num_per_class = [len(data_idxs_dict[i]) for i in data_idxs_dict.keys()]
    data_num = sum(data_num_per_class)

    datasize_per_client = size_of_division(clients_num, data_num)
    datasize_per_client_per_class = [size_of_division(class_num, i) for i in datasize_per_client]

    temp = [set(i) for i in data_idxs_dict.values()]

    for c in range(clients_num):
        cur_client_id = clients_id[c] 
        for i in range(class_num):
            num_data_per_cli = datasize_per_client_per_class[c][i]
            
            if len(temp[i]) < num_data_per_cli:
                rand_set = np.random.choice(list(temp[i]), num_data_per_cli, replace=True)
            elif len(temp[i]) == num_data_per_cli:
                rand_set = np.random.choice(list(temp[i]), num_data_per_cli, replace=False)
            else:
                rand_set = np.random.choice(list(temp[i]), num_data_per_cli, replace=False)
                temp[i] = temp[i] - set(rand_set)
            
            clients_data[cur_client_id].extend(rand_set)

    return clients_data

def non_iid(clients_id, data_idxs_dict, class_per_client):
    '''Partition dataset into multiple clients with iid.
    The data size is equal(or the difference is less than 1).
    each client only have 2 class.

    Args: 
        clients_num: the number of clients.
        data_idxs_dict: the data index dictionary. 
            The structure is:
                {
                    'label_0': [...],
                    'label_1': [...],
                    ...
                }
    '''   
    clients_data = defaultdict(list)
    clients_num = len(clients_id)
    class_num = len(data_idxs_dict)

    data_num_per_class = [len(data_idxs_dict[i]) for i in data_idxs_dict.keys()]
    data_num = sum(data_num_per_class)

    datasize_per_client = size_of_division(clients_num, data_num)
    class_comb = generate_class_comb(clients_num, class_num, class_per_client)
    
    datasize_per_client_per_class = [size_of_division(len(class_comb[i]), datasize_per_client[i]) for i in range(clients_num)]

    temp = [set(i) for i in data_idxs_dict.values()]

    for c in range(clients_num):
        cur_client_id = clients_id[c] 
        for i in range(class_per_client):
            num_data_per_cli = datasize_per_client_per_class[c][i]
            cur_client_class_ = class_comb[c][i]

            if len(temp[cur_client_class_]) < num_data_per_cli:
                rand_set = np.random.choice(list(temp[cur_client_class_]), num_data_per_cli, replace=True)
            elif len(temp[cur_client_class_]) == num_data_per_cli:
                rand_set = np.random.choice(list(temp[cur_client_class_]), num_data_per_cli, replace=False)
            else:
                rand_set = np.random.choice(list(temp[cur_client_class_]), num_data_per_cli, replace=False)
                temp[cur_client_class_] = temp[cur_client_class_] - set(rand_set)  

            clients_data[cur_client_id].extend(rand_set)

    return clients_data              

def non_iid_dirichlet(clients_id, data_idxs_dict, alpha, min_size=1):
    
    clients_data = defaultdict(list)
    clients_num = len(clients_id)
    class_num = len(data_idxs_dict)

    temp = [set(i) for i in data_idxs_dict.values()]

    for i in range(class_num):
        
        sampled_prob = len(temp[i]) * np.random.dirichlet(np.repeat(alpha, clients_num))

        for c in range(clients_num):
            cur_client_id = clients_id[c]

            num_data_per_cli = max(int(round(sampled_prob[c])), min_size)

            if len(temp[i]) < num_data_per_cli:
                rand_set = np.random.choice(list(temp[i]), num_data_per_cli, replace=True)
            elif len(temp[i]) == num_data_per_cli:
                rand_set = np.random.choice(list(temp[i]), num_data_per_cli, replace=False)
            else:
                rand_set = np.random.choice(list(temp[i]), num_data_per_cli, replace=False)
                temp[i] = temp[i] - set(rand_set)

            clients_data[cur_client_id].extend(rand_set)

    return clients_data

def iid_v2(clients_id, datasize):
    data_idx_list = np.arange(datasize)
    clients_data = defaultdict(list)
    clients_num = len(clients_id)

    data_per_client = datasize // clients_num

    temp = set(data_idx_list)

    for c in range(clients_num):
        cur_client_id = clients_id[c]
        
        if len(temp) < data_per_client:
            rand_set = np.random.choice(data_idx_list, size=data_per_client, replace=True)
        elif len(temp) == data_per_client:
            rand_set = np.random.choice(data_idx_list, size=data_per_client, replace=False)
        else:
            rand_set = np.random.choice(data_idx_list, size=data_per_client, replace=False)
            temp = temp - set(rand_set)
        
        clients_data[cur_client_id] = rand_set

    return clients_data


def data_distribution_simulation(clients_id, data_idx_dict, distribution_type, class_per_client=2, alpha=0.9, min_size=1, datasize=None):
    if distribution_type == 'iid':
        # clients_data = iid(clients_id, data_idx_dict)
        clients_data = iid_v2(clients_id, datasize)
    elif distribution_type == 'non_iid_class':
        clients_data = non_iid(clients_id, data_idx_dict, class_per_client)
    elif distribution_type == 'non_iid_dir':
        clients_data = non_iid_dirichlet(clients_id, data_idx_dict, alpha, min_size)
    else:
        raise ValueError('The options of the data distribution type should be: iid, non_iid_class, non_iid_dir.')

    return clients_data

