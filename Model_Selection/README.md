# Introduction
The programs and models for the decoder model selection module are contained in this folder, including the experiments based on CsiNet, CRNet, and the data visualization.

# Requirements
- Python 3.5 (or 3.6)
- Keras (>=2.1.1)
- Tensorflow (>=1.4)
- Numpy

# Instructions
The following instructions are necessary before the training and testing procedures of decoder model selection:
- The repository provides the programs used for the training and testing of the decoder model selection module based on CsiNet and CRNet in the form of python files. The network models in the form of h5 files, as well as the classification models in the form of .sav files are also included in the "result" folder for the convenience of reproducing the results shown in the letter.
- The part "settings of GPU" at the beginning of each python file should be adjusted in advance according to the specific device setting of the user.
- The experiments of different Compression Rates can be performed by adjusting the "encoded_dim" in the programs.
- The folder named "data" should be established in advance in "MS-CsiNet" and "MS-CRNet" to store the dataset used for training and testing.
The dataset used in the network training can be downloaded from https://drive.google.com/drive/folders/1_lAMLk_5k1Z8zJQlTr5NRnSD6ACaNRtj?usp=sharing, which is first provided in https://github.com/sydney222/Python_CsiNet). The dataset should be put in the folds "data" in advance. Therefore, the structure of the fold should be:


- The programs and models based on CsiNet are contained in "MS-CsiNet", and  The programs and models based on CsiNet are contained in "MS-CsiNet". Take "MS-CsiNet" the example: ModelSelection4CR.py performs the training of the decoder selection module when the compression rate == 4 and get the classification accuracy; Test_ModelSelection4CR.py can be used to test the selection module to obtain the end-to-end feedback accuracy with the decoder model selection.
