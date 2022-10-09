import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU, Activation, subtract, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, Callback
from tensorflow.keras.callbacks import ModelCheckpoint
import scipy.io as sio
import numpy as np
import random
import math
import time
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

import hdf5storage #load Matlab data bigger than 2GB
from sklearn.neural_network import MLPClassifier
import joblib

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

tf.reset_default_graph()

# 40%
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.30
session = tf.Session(config=config)

# epochs = 1000
# batch_size = 200

# size_of_trainingset = 50000  # dataset size 训练集合容量

# image params
img_height = 32
img_width = 32
img_channels = 2
img_total = img_height * img_width * img_channels
# network params
residual_num = 2
encoded_dim = 512  # 压缩率
# compress rate=1/4->dim.=512, compress rate=1/16->dim.=128, compress rate=1/32->dim.=64, compress rate=1/64->dim.=32


# Bulid the autoencoder model of CRNet
def add_common_layers(y):
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)
    return y


def encoder_network(x):
    # encoder
    sidelink = x

    x = Conv2D(2, (3, 3), padding='same', data_format="channels_first")(x)
    x = add_common_layers(x)
    x = Conv2D(2, (1, 9), padding='same', data_format="channels_first")(x)
    x = add_common_layers(x)
    x = Conv2D(2, (9, 1), padding='same', data_format="channels_first")(x)
    x = add_common_layers(x)

    sidelink = Conv2D(2, (3, 3), padding='same', data_format="channels_first")(sidelink)
    sidelink = add_common_layers(sidelink)

    x = concatenate([x, sidelink], axis=1)

    x = Conv2D(2, (1, 1), padding='same', data_format="channels_first")(x)
    x = add_common_layers(x)

    x = Reshape((img_total,))(x)
    encoded = Dense(encoded_dim, activation='linear')(
        x)  # Note that CRNet has no quantization operation, so the linear activation function is used

    return encoded


def decoder_network(encoded):
    # decoder
    x = Dense(img_total, activation='linear')(
        encoded)  # Due to the settings in CRNet, the linear activation function is adopted
    x = Reshape((img_channels, img_height, img_width,))(x)

    x = Conv2D(2, (5, 5), padding='same', data_format="channels_first")(x)
    x = add_common_layers(x)

    # CRBlock
    for i in range(2):
        sidelink = x
        shortcut = x

        x = Conv2D(7, (3, 3), padding='same', data_format="channels_first")(x)
        x = add_common_layers(x)
        x = Conv2D(7, (1, 9), padding='same', data_format="channels_first")(x)
        x = add_common_layers(x)
        x = Conv2D(7, (9, 1), padding='same', data_format="channels_first")(x)
        x = add_common_layers(x)

        sidelink = Conv2D(7, (1, 5), padding='same', data_format="channels_first")(sidelink)
        sidelink = add_common_layers(sidelink)
        sidelink = Conv2D(7, (5, 1), padding='same', data_format="channels_first")(sidelink)
        sidelink = add_common_layers(sidelink)

        x = concatenate([x, sidelink], axis=1)

        x = Conv2D(2, (1, 1), padding='same', data_format="channels_first")(x)
        x = add_common_layers(x)

        x = add([x, shortcut])

    x = Activation('sigmoid')(x)
    return x


image_tensor = keras.Input(shape=(img_channels, img_height, img_width))
codewords_vector = keras.Input(shape=(encoded_dim,))

encoder = keras.Model(inputs=[image_tensor], outputs=[encoder_network(image_tensor)])
decoder = keras.Model(inputs=[codewords_vector], outputs=[decoder_network(codewords_vector)])
autoencoder = keras.Model(inputs=[image_tensor], outputs=[decoder(encoder(image_tensor))])

adam = keras.optimizers.Adam(lr=0.0001)

autoencoder.layers[1].trainable = False

encoder.compile(optimizer=adam, loss='mse')
decoder.compile(optimizer=adam, loss='mse')
autoencoder.compile(optimizer=adam, loss='mse')  # 训练参数

print(autoencoder.summary())

autoencoder.load_weights('result/CRNet_MT_dim512_indoor_model.h5')

# Data loading  载入数据，调整数据格式
mat = sio.loadmat('dataset/DATA_Htestin.mat')
x_test_in = mat['HT']  # array

x_test_in = x_test_in.astype('float32')
print(x_test_in.shape)

x_test_in = np.reshape(x_test_in, (
len(x_test_in), img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format

codeword_test_in = encoder.predict(x_test_in)
# ******* 获取indoor原始test数据通过indoor encoder后的码字

autoencoder.load_weights('result/CRNet_MT_dim512_outdoor_model.h5')
# Data loading  载入数据，调整数据格式
mat = sio.loadmat('dataset/DATA_Htestout.mat')
x_test_out = mat['HT']  # array

x_test_out = x_test_out.astype('float32')
print(x_test_out.shape)

x_test_out = np.reshape(x_test_out, (
len(x_test_out), img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format

codeword_test_out = encoder.predict(x_test_out)
# ******* 获取outdoor原始test数据通过indoor encoder后的码字

x_test = np.append(x_test_in, x_test_out, axis=0)
codeword_test = np.append(codeword_test_in, codeword_test_out, axis=0)
# ******* 合并两部分test数据

BS_MLP = joblib.load('result/BS_MLP_indoor5w+outdoor5w_dim512_model.sav')

test_pred = BS_MLP.predict(codeword_test)

x_hat_in = []
x_hat_out = []
x_hat_in = np.reshape(x_hat_in, (len(x_hat_in), 2, 32, 32))
x_hat_out = np.reshape(x_hat_out, (len(x_hat_out), 2, 32, 32))

autoencoder.load_weights('result/CRNet_MT_dim512_indoor_model.h5')
for i in range(0, 20000):
    cur_test = x_test[i: i + 1]
    if test_pred[i] == 1:
        x_hat_in = np.append(x_hat_in, autoencoder.predict(cur_test), axis=0)
    else:
        autoencoder.load_weights('result/CRNet_MT_dim512_outdoor_model.h5')
        x_hat_in = np.append(x_hat_in, autoencoder.predict(cur_test), axis=0)
        autoencoder.load_weights('result/CRNet_MT_dim512_indoor_model.h5')
    if i % 100 == 0:
        print("cur test i: ", i)

autoencoder.load_weights('result/CRNet_MT_dim512_outdoor_model.h5')
for i in range(20000, 40000):
    cur_test = x_test[i: i + 1]
    if test_pred[i] == 0:
        x_hat_out= np.append(x_hat_out, autoencoder.predict(cur_test), axis=0)
    else:
        autoencoder.load_weights('result/CRNet_MT_dim512_indoor_model.h5')
        x_hat_out = np.append(x_hat_out, autoencoder.predict(cur_test), axis=0)
        autoencoder.load_weights('result/CRNet_MT_dim512_outdoor_model.h5')
    if i % 100 == 0:
        print("cur test i: ", i)

# indoor NMSE
x_test_in_real = np.reshape(x_test_in[:, 0, :, :], (len(x_test_in), -1))
x_test_in_imag = np.reshape(x_test_in[:, 1, :, :], (len(x_test_in), -1))
x_test_in_C = x_test_in_real-0.5 + 1j*(x_test_in_imag-0.5)
x_hat_in_real = np.reshape(x_hat_in[:, 0, :, :], (len(x_hat_in), -1))
x_hat_in_imag = np.reshape(x_hat_in[:, 1, :, :], (len(x_hat_in), -1))
x_hat_in_C = x_hat_in_real-0.5 + 1j*(x_hat_in_imag-0.5)

power = np.sum(abs(x_test_in_C)**2, axis=1)

mse = np.sum(abs(x_test_in_C-x_hat_in_C)**2, axis=1)

print("In indoor environment")
print("When dimension is", encoded_dim)
print("NMSE is ", 10*math.log10(np.mean(mse/power)))

# outdoor NMSE
x_test_out_real = np.reshape(x_test_out[:, 0, :, :], (len(x_test_out), -1))
x_test_out_imag = np.reshape(x_test_out[:, 1, :, :], (len(x_test_out), -1))
x_test_out_C = x_test_out_real-0.5 + 1j*(x_test_out_imag-0.5)
x_hat_out_real = np.reshape(x_hat_out[:, 0, :, :], (len(x_hat_out), -1))
x_hat_out_imag = np.reshape(x_hat_out[:, 1, :, :], (len(x_hat_out), -1))
x_hat_out_C = x_hat_out_real-0.5 + 1j*(x_hat_out_imag-0.5)

power = np.sum(abs(x_test_out_C)**2, axis=1)

mse = np.sum(abs(x_test_out_C-x_hat_out_C)**2, axis=1)

print("In outdoor environment")
print("When dimension is", encoded_dim)
print("NMSE is ", 10*math.log10(np.mean(mse/power)))
