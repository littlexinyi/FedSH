# FedSH: Towards Privacy-preserving Text-based Person Re-Identification
[![LICENSE]]

[![Python](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.11.0-%237732a8) 

The implementation of paper [**FedSH: Towards Privacy-preserving Text-based Person Re-Identification**]

FedSH is a framework for exploring privacy-preserving text-based person ReID.


## Requirement

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
   
  Organize them in `dataset` folder as follows:
   ~~~
   |-- dataset/
   |   |-- <CUHK-PEDES>/
   |       |-- imgs
               |-- cam_a
               |-- cam_b
               |-- CUHK01
               |-- CUHK03
               |-- Market
   |       |-- reid_raw.json
   
   ~~~
 

2.  **ICFG-PEDES**

  Download the ICFG-PEDES dataset from [here](https://github.com/zifyloo/SSAN)   


   Organize them in `dataset` folder as follows:

   ~~~
   |-- dataset/
   |   |-- <ICFG-PEDES>/
   |       |-- imgs
               |-- test
               |-- train 
   |       |-- ICFG_PEDES.json
   
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
If you find FedSH useful in your work, you can cite the following paper:


## Copyright
* The code is provided by Wentao Ma and Xinyi Wu from NUDT. If you have any question, please contact wtma@nudt.edu.cn or wuxinyi17@nudt.edu.cn.

## Acknowledgments

Our code is based on [SSAN](https://github.com/zifyloo/SSAN) and [FLLIB](https://github.com/zenghui9977/FLLIB_develop).