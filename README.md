# MT-CSI
Source code of multi-task training approach for CSI feedback in massive MIMO systems


# Introduction
This repository contains the program of the training and testing procedures of FCS-CsiNet and FCS-CRNet proposed in Boyuan Zhang, Haozhen Li, Xin Liang, Xinyu Gu, and Lin Zhang, "Multi-task Training Approach for CSI Feedback in Massive MIMO Systems" (submitted to IEEE Communications Letters).

# Requirements
- Python 3.5 (or 3.6)
- Keras (>=2.1.1)
- Tensorflow (>=1.4)
- Numpy

# Instructions
The following instructions are necessary before the network training and testing procedures:
- The repository only provide the programs used for the training and testing of the three training approaches including single-task training, transfer learning and multi-task training based on CsiNet in the form of python files. The network models in the form of h5 files are not included.
- The part "settings of GPU" in each python file should be adjusted in advance according to the specific device setting of the user.
- The experiments of different Compression Rates can be performed by adjusting the "encoded_dim" in the programs.
- The folds named "result" and "data" should be established in advance in the folds "Single-Task", "Transfer" and "Multi-task" to store the models obtained during the training procedure and to store the dataset used for training and testing.
- The dataset used in the network training can be downloaded from https://drive.google.com/drive/folders/1_lAMLk_5k1Z8zJQlTr5NRnSD6ACaNRtj?usp=sharing, which is first provided in https://github.com/sydney222/Python_CsiNet). The dataset should be put in the folds "data" in advance.
Therefore, the structure of the folds "Single-Task", "Transfer" and "Multi-task" should be:
```
*.py
result/
data/
  *.mat
```
# Training and testing procedures 
The training and testing procedures for the three training approaches are demonstrated as follows:
## Step.1 Single-task training

## Step.2 Transfer learning

## Step.3 Multi-task training

The results are given in the submitted manuscript "Multi-task Training Approach for CSI Feedback in Massive MIMO Systems".
