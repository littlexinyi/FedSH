import logging
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import torch.utils.data as data
from torch.utils.data import DataLoader
from fllib.datasets.simulation import data_distribution_simulation
from fllib.datasets.utils import TransformDataset
from fllib.datasets.read_write_data import read_dict

support_dataset = ['mnist', 'fmnist', 'kmnist', 'emnist', 'cifar10', 'cifar100','ICFG-PEDES', 'CUHK-PEDES']
logger = logging.getLogger(__name__)

class BaseDataset(object):
    '''The internal base dataset class, most of the dataset is based on the torch lib

    Args:
        type: The dataset name, options: mnist, fmnist, kmnist, emnist,cifar10
        root: The root directory of the dataset folder.
        download: The dataset should be download or not

    '''
    def __init__(self, datatype, root, download):
        self.type = datatype
        self.root = root
        self.download = download
        self.trainset = None
        self.testset = None
        self.idx_dict = {}

    def get_dataset(self):
        if self.type == 'mnist':
            simple_transform = torchvision.transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
            self.trainset = torchvision.datasets.MNIST(root=self.root, train=True, transform=simple_transform, download=self.download)
            self.testset = torchvision.datasets.MNIST(root=self.root, train=False, transform=simple_transform, download=self.download)
        
        elif self.type == 'fmnist':
            simple_transform = torchvision.transforms.Compose([transforms.ToTensor(),])
            self.trainset = torchvision.datasets.FashionMNIST(root=self.root, train=True, transform=simple_transform, download=self.download)
            self.testset = torchvision.datasets.FashionMNIST(root=self.root, train=False, transform=simple_transform, download=self.download)
        
        elif self.type == 'kmnist':
            simple_transform = torchvision.transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
            self.trainset = torchvision.datasets.KMNIST(root=self.root, train=True, transform=simple_transform, download=self.download)
            self.testset = torchvision.datasets.KMNIST(root=self.root, train=False, transform=simple_transform, download=self.download)

        elif self.type == 'emnist':
            simple_transform = torchvision.transforms.Compose([transforms.ToTensor()])
            self.trainset = torchvision.datasets.EMNIST(root=self.root, train=True, split='byclass', transform=simple_transform, download=self.download)
            self.testset = torchvision.datasets.EMNIST(root=self.root, train=False, split='byclass', transform=simple_transform, download=self.download)

        elif self.type == 'cifar10':
            simple_transform = torchvision.transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])])
            self.trainset = torchvision.datasets.CIFAR10(root=self.root, train=True, transform=simple_transform, download=self.download)
            self.testset = torchvision.datasets.CIFAR10(root=self.root, train=False, transform=simple_transform, download=self.download)

        elif self.type == 'cifar100':
            simple_transform = torchvision.transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])])
            self.trainset = torchvision.datasets.CIFAR100(root=self.root, train=True, transform=simple_transform, download=self.download)
            self.testset = torchvision.datasets.CIFAR100(root=self.root, train=False, transform=simple_transform, download=self.download)

        else:
            raise ValueError(f'Dataset name is not correct, the options are listed as follows: {support_dataset}')
        
    def get_train_dataset(self):
        return self.trainset
    def get_test_dataset(self):
        return self.testset


class FederatedDataset(object):

    def __init__(self, data_name, trainset, test_imgset, test_txtset, simulated, simulated_root, distribution_type, clients_id, class_per_client=2, alpha=0.9, min_size=1):
        
        self.trainset = trainset
        self.test_imgset = test_imgset
        self.test_txtset = test_txtset
        # self.idx_dict = self.build_idx_dict()
        self.idx_dict = {}
        
        self.data_name = data_name

        self.simulated = simulated
        self.simulated_root = simulated_root
        self.distribution_type = distribution_type

        self.clients_id = clients_id
        self.clients_num = len(clients_id)

        if self.distribution_type == 'iid':
            distribution_args = 0
        elif self.distribution_type == 'non_iid_class':
            distribution_args = class_per_client
        elif self.distribution_type == 'non_iid_dir':
            distribution_args = alpha

        self.store_file_name = f'{self.data_name}_{self.distribution_type}_clients{self.clients_num}_args{distribution_args}'

        if os.path.exists(os.path.join(self.simulated_root, self.store_file_name)) and (not self.simulated):
            logger.info(f'Clients data file {self.store_file_name} already exist. Loading......')
            self.clients_data = torch.load(os.path.join(simulated_root, self.store_file_name))
            
        else:
            if not os.path.exists(self.simulated_root):
                os.makedirs(self.simulated_root)
            logger.info(f'Initialize the file {self.store_file_name}.')
            self.clients_data = data_distribution_simulation(self.clients_id, self.idx_dict, self.distribution_type, class_per_client, alpha, min_size, datasize=len(trainset))
            torch.save(self.clients_data, os.path.join(self.simulated_root, self.store_file_name))

    def build_idx_dict(self):
        self.idx_dict = {}
        for idx, data in enumerate(self.trainset):
            _, label, _, _, _, _ = data
            if label in self.idx_dict:
                self.idx_dict[label].append(idx)
            else:
                self.idx_dict[label] = [idx]
        return self.idx_dict

    def get_dataloader(self, client_id, batch_size, istrain=True, drop_last=False): #client/server获取dataloader
        if self.data_name in support_dataset:
            if istrain: #训练集
                if client_id in self.clients_id:
                    
                    data_idx = self.clients_data[client_id]
                    
                    imgs, labels, caption_code, caption_length, caption_code_cr, caption_length_cr = [], [], [], [], [], []
                    
                    for i in data_idx:
                        imgs.append(self.trainset[i][0])
                        labels.append(self.trainset[i][1])
                        caption_code.append(self.trainset[i][2]) 
                        caption_length.append(self.trainset[i][3])
                        caption_code_cr.append(self.trainset[i][4])
                        caption_length_cr.append(self.trainset[i][5])
                        
                    train_dataset=TransformDataset(imgs, labels, caption_code, caption_length, caption_code_cr, caption_length_cr)
                    
                    return DataLoader(train_dataset, batch_size=min(len(data_idx), batch_size), shuffle=True, drop_last=drop_last)
                else:
                    raise ValueError('The client id is not existed.')
            else:   
                test_imgLoader = DataLoader(dataset=self.test_imgset, batch_size=batch_size, shuffle=True)
                test_txtLoader = DataLoader(dataset=self.test_txtset, batch_size=batch_size, shuffle=True)
                return test_imgLoader, test_txtLoader
        
        else:
            raise ValueError(f'Dataset name is not correct, the options are listed as follows: {support_dataset}')
    
    def get_client_datasize(self, client_id=None):
        if client_id in self.clients_id:
            return len(self.clients_data[client_id])
        else:
            raise ValueError('The client id is not existed.')
    
    def get_total_datasize(self):
        return len(self.trainset)
                
        
def fliplr(img, dim):
    """
    flip horizontal
    :param img:
    :return:
    """
    inv_idx = torch.arange(img.size(dim) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(dim, inv_idx)
    return img_flip

def tran(mode):
    transform_list = [
    transforms.RandomHorizontalFlip(),
    transforms.Resize((384, 128), Image.BICUBIC),   # interpolation
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                            (0.5, 0.5, 0.5))]
    if(mode == "test"):
        transform_list.pop(0)
    trans = transforms.Compose(transform_list)
    return trans

class PEDEDataset(data.Dataset):    #继承了 data.Dataset该数据集类
    def __init__(self, config):
        self.config = config
        self.dataroot = os.path.join(config.dataset.root,config.dataset.data_name)
        self.mode = "train"
        self.flip_flag = 1
        self.caption_length_max = config.dataset.caption_length_max
        data_save = read_dict(os.path.join(self.dataroot, 'processed_data', 'train_save.pkl'))

        self.img_path = [os.path.join(self.dataroot, img_path) for img_path in data_save['img_path']]

        self.label = data_save['id']

        self.caption_code = data_save['lstm_caption_id']

        self.same_id_index = data_save['same_id_index']

        self.transform = tran(self.mode)       #数据处理工具

        self.num_data = len(self.img_path)

    def __getitem__(self, index):
        """
        :param index:
        :return: image and its label
        """
        image = Image.open(self.img_path[index])
        image = self.transform(image)
        label = torch.from_numpy(np.array([self.label[index]])).long()
        caption_code, caption_length = self.caption_mask(self.caption_code[index])

        same_id_index = np.random.randint(len(self.same_id_index[index]))
        same_id_index = self.same_id_index[index][same_id_index]
        same_id_caption_code, same_id_caption_length = self.caption_mask(self.caption_code[same_id_index])

        return image, label, caption_code, caption_length, same_id_caption_code, same_id_caption_length

    def get_data(self, index, img=True):
        if img:
            image = Image.open(self.img_path[index])
            image = self.transform(image)
        else:
            image = 0

        label = torch.from_numpy(np.array([self.label[index]])).long()

        caption_code, caption_length = self.caption_mask(self.caption_code[index])

        return image, label, caption_code, caption_length

    def caption_mask(self, caption):
        caption_length = len(caption)
        caption = torch.from_numpy(np.array(caption)).view(-1).long()

        if caption_length < self.caption_length_max:
            zero_padding = torch.zeros(self.caption_length_max - caption_length).long()
            caption = torch.cat([caption, zero_padding], 0)
        else:
            caption = caption[:self.caption_length_max]
            caption_length = self.caption_length_max

        return caption, caption_length

    def __len__(self):
        return self.num_data


class PEDE_img_dateset(data.Dataset):
    def __init__(self, config):

        self.config = config
        self.mode = "test"
        self.dataroot = os.path.join(config.dataset.root,config.dataset.data_name)
        data_save = read_dict(os.path.join(self.dataroot, 'processed_data', 'test_save.pkl'))
        self.img_path = [os.path.join(self.dataroot, img_path) for img_path in data_save['img_path']]
        self.label = data_save['id']
        self.transform = tran(self.mode)
        self.num_data = len(self.img_path)

    def __getitem__(self, index):
        """
        :param index:
        :return: image and its label
        """

        image = Image.open(self.img_path[index])
        image = self.transform(image)

        label = torch.from_numpy(np.array([self.label[index]])).long()

        return image, label

    def __len__(self):
        return self.num_data

class PEDE_txt_dateset(data.Dataset):
    def __init__(self, config):

        self.config = config
        self.mode = "test"
        self.dataroot = os.path.join(config.dataset.root,config.dataset.data_name)
        data_save = read_dict(os.path.join(self.dataroot, 'processed_data', 'test_save.pkl'))
        self.label = data_save['caption_label']
        self.caption_code = data_save['lstm_caption_id']
        self.num_data = len(self.caption_code)

    def __getitem__(self, index):
        label = torch.from_numpy(np.array([self.label[index]])).long()

        caption_code, caption_length = self.caption_mask(self.caption_code[index])
        return label, caption_code, caption_length

    def caption_mask(self, caption):
        caption_length = len(caption)
        caption = torch.from_numpy(np.array(caption)).view(-1).float()
        if caption_length < self.config.dataset.caption_length_max:
            zero_padding = torch.zeros(self.config.dataset.caption_length_max - caption_length)
            caption = torch.cat([caption, zero_padding], 0)
        else:
            caption = caption[:self.config.dataset.caption_length_max]
            caption_length = self.config.dataset.caption_length_max

        return caption, caption_length

    def __len__(self):
        return self.num_data





