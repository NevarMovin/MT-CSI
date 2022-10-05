# MT-CSI
Source code of multi-task training approach for CSI feedback in massive MIMO systems


# Introduction
This repository contains the program of the multi-task training approach proposed in Boyuan Zhang, Haozhen Li, Xin Liang, Xinyu Gu, and Lin Zhang, "Multi-task Training Approach for CSI Feedback in Massive MIMO Systems" (submitted to IEEE Communications Letters).

# Requirements
- Python 3.5 (or 3.6)
- Keras (>=2.1.1)
- Tensorflow (>=1.4)
- Numpy

# Instructions
The following instructions are necessary before the network training and testing procedures:
- The repository provides the programs used for the training and testing of the multi-task training based on CsiNet, CRNet, SALDR and MRNet in the form of python files. The network models in the form of h5 files are also not included in the "result" folder for the convenience of reproducing the results shown in the letter.
- The part "settings of GPU" in each python file should be adjusted in advance according to the specific device setting of the user.
- The experiments of different Compression Rates can be performed by adjusting the "encoded_dim" in the programs.
- The folder named "data" should be established in advance to store the dataset used for training and testing.
- The dataset used in the network training can be downloaded from https://drive.google.com/drive/folders/1_lAMLk_5k1Z8zJQlTr5NRnSD6ACaNRtj?usp=sharing, which is first provided in https://github.com/sydney222/Python_CsiNet). The dataset should be put in the folds "data" in advance.
Therefore, the structure of the fold should be:
```
*.py
result/
data/
  *.mat
```
# Training and testing procedures 
The training and testing procedures for the multi-task training approach can be achieved through Pre-training.py, Fine-tuning-indoor.py and Fine-tuning-outdoor.py. In addition, the experimental results can be directly obtained by running "model_test.py". The scenarios, compression rates and the corresponding models can be adjusted in "model_test.py".

The results are given in the submitted manuscript "Multi-task Training Approach for CSI Feedback in Massive MIMO Systems".
