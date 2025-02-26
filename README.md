# FedSH: Towards Privacy-preserving Text-based Person Re-Identification

[![LICENSE](https://img.shields.io/badge/license-Apache2.0-green)](https://github.com/littlexinyi/FedSH/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.11.0-%237732a8) 

The implementation of paper [**FedSH: Towards Privacy-preserving Text-based Person Re-Identification**]


## Requirements

```
torch >= 1.7.0
yaml
omegaconf
visdom
Pillow 8.2.0
```
## Data preparation

1.  **CUHK-PEDES**

  Download the CUHK-PEDES dataset from [here](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description) 
   
  Organize them in `data` folder as follows:
   ~~~
   |-- data/
   |   |-- <CUHK-PEDES>/
   |       |-- imgs
               |-- cam_a
               |-- cam_b
               |-- CUHK01
               |-- CUHK03
               |-- Market
   |       |-- reid_raw.json
   |-- fllib/
   ~~~
 

2.  **ICFG-PEDES**

  Download the ICFG-PEDES dataset from [here](https://github.com/zifyloo/SSAN)   


   Organize them in `data` folder as follows:

   ~~~
   |-- data/
   |   |-- <ICFG-PEDES>/
   |       |-- imgs
               |-- test
               |-- train 
   |       |-- ICFG_PEDES.json
   |-- fllib/
   ~~~
3. **Data preprocessing**

then run the `process_CUHK_data.py` and `process_ICFG_data.py` in [SSAN](https://github.com/zifyloo/SSAN)


## How to Run

* if you have opened the visualization, you should start the visdom first.

  ```
  call start_visdom.bat
  ```

  You can see the training lines in localhost:8097(Default)

* Then start to train directly

    ```
    python train.py
    ```
* After training done, you can test your model by run:

    ```
    python test.py
    ```

## Citation
If you find FedSH useful in your work, please consider staring 🌟 this repo and citing 📑 our paper:
```
@ARTICLE{10310121,
  author={Ma, Wentao and Wu, Xinyi and Zhao, Shan and Zhou, Tongqing and Guo, Dan and Gu, Lichuan and Cai, Zhiping and Wang, Meng},
  journal={IEEE Transactions on Multimedia}, 
  title={FedSH: Towards Privacy-Preserving Text-Based Person Re-Identification}, 
  year={2024},
  volume={26},
  number={},
  pages={5065-5077},
  keywords={Semantics;Training;Task analysis;Privacy;Visualization;Federated learning;Servers;Text-based Person ReID;Cross-modal Retrieval;Federated Learning;Multi-granularity Representation},
  doi={10.1109/TMM.2023.3330091}}
```

## Copyright
* The code is provided by Wentao Ma and Xinyi Wu from NUDT. If you have any question, please contact wtma@nudt.edu.cn or wuxinyi17@nudt.edu.cn.

## Acknowledgments

Our code is based on [SSAN](https://github.com/zifyloo/SSAN) and [FLLIB](https://github.com/zenghui9977/FLLIB_develop).
