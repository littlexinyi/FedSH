from fllib.base import *

# this config can be changed by your needs
config = {
    'dataset': {
         'data_name': "CUHK-PEDES",
         'class_num': 11000,
         'vocab_size': 5000
     },    
    'server': {
       'rounds': 30,
       'clients_per_round': 5,
       'aggregation_detail': {
            'Backbone': True,
            'PRL': True
       }
    },
    'client': {
        'local_epoch': 3,
        'optimizer': {
            'type': "Adam"          #option: Adam or FedProx 
        }
    }   
}

init(config=config)

run()
