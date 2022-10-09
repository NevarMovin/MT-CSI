import tensorflow as tf
from keras.layers import Input, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU, Activation, Lambda
from keras.models import Model
from keras.callbacks import TensorBoard, Callback, ModelCheckpoint
from keras import optimizers
import scipy.io as sio
import numpy as np
import math
import time
import hdf5storage  # load Matlab data bigger than 2GB
from sklearn.neural_network import MLPClassifier
import joblib

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.reset_default_graph()

# 40%
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.40
session = tf.Session(config=config)

# epochs = 1000
# batch_size = 200

size_of_trainingset = 50000  # dataset size 训练集合容量

# image params
img_height = 32
img_width = 32
img_channels = 2
img_total = img_height * img_width * img_channels
# network params
residual_num = 2
encoded_dim = 512  # 压缩率
# compress rate=1/4->dim.=512, compress rate=1/16->dim.=128, compress rate=1/32->dim.=64, compress rate=1/64->dim.=32


# Build the autoencoder model of CsiNet  网络构建
def encoder_network(x):
    def add_common_layers(y):
        y = BatchNormalization()(y)
        y = LeakyReLU()(y)
        return y

    def residual_block_decoded(y):
        shortcut = y
        y = Conv2D(8, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
        y = add_common_layers(y)

        y = Conv2D(16, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
        y = add_common_layers(y)

        y = Conv2D(2, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
        y = BatchNormalization()(y)

        y = add([shortcut, y])
        y = LeakyReLU()(y)

        return y

    def dense_normalization(y):
        y = tf.nn.l2_normalize(y, axis=1)  # normalization
        return y

    # encoder
    x = Conv2D(2, (3, 3), padding='same', data_format="channels_first")(x)
    x = add_common_layers(x)

    x = Reshape((img_total,))(x)
    encoded = Dense(encoded_dim, activation='linear', name='encoded_layer')(x)  # encoded result
    #     encoded = Lambda(dense_normalization, name = 'normalized_encoded')(encoded) # normalized encoded result

    return encoded


def decoder_network(encoded):
    def add_common_layers(y):
        y = BatchNormalization()(y)
        y = LeakyReLU()(y)
        return y

    def residual_block_decoded(y):
        shortcut = y
        y = Conv2D(8, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
        y = add_common_layers(y)

        y = Conv2D(16, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
        y = add_common_layers(y)

        y = Conv2D(2, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
        y = BatchNormalization()(y)

        y = add([shortcut, y])
        y = LeakyReLU()(y)

        return y

    # decoder
    x = Dense(img_total, activation='linear')(encoded)
    x = Reshape((img_channels, img_height, img_width,), name='reconstructed_image')(x)

    for i in range(residual_num):
        x = residual_block_decoded(x)

    x = Conv2D(2, (3, 3), activation='sigmoid', padding='same', data_format="channels_first")(x)

    return x


image_tensor = Input(shape=(img_channels, img_height, img_width))
input_vector = Input(shape=(encoded_dim,))

encoder = Model(image_tensor, encoder_network(image_tensor))
decoder = Model(input_vector, decoder_network(input_vector))
autoencoder = Model(image_tensor, decoder(encoder(image_tensor)))

adam = optimizers.Adam(lr=0.0001)  # reset the learning rate

encoder.layers[1].trainable = False
encoder.layers[3].trainable = False
encoder.layers[5].trainable = False

encoder.compile(optimizer=adam, loss='mse')
decoder.compile(optimizer=adam, loss='mse')
autoencoder.compile(optimizer=adam, loss='mse')  # 训练参数

print(autoencoder.summary())

autoencoder.load_weights('result/CsiNet_MT_dim512_indoor_model.h5')
# Data loading  载入数据，调整数据格式
mat = sio.loadmat('dataset/DATA_Htrainin.mat')
x_train = mat['HT']  # array

x_train = x_train.astype('float32')
x_train = x_train[0:size_of_trainingset]
print(x_train.shape)

x_train = np.reshape(x_train, (
len(x_train), img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format

codeword_in_train = encoder.predict(x_train)

sio.savemat('codeword_in.mat', {'codeword_indoor': codeword_in_train})
# ******* 获取indoor原始train数据通过indoor encoder后的码字

autoencoder.load_weights('result/CsiNet_MT_dim512_outdoor_model.h5')
# Data loading  载入数据，调整数据格式
mat = sio.loadmat('dataset/DATA_Htrainout.mat')
x_train = mat['HT']  # array

x_train = x_train.astype('float32')
x_train = x_train[0:size_of_trainingset]
print(x_train.shape)

x_train = np.reshape(x_train, (
len(x_train), img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format

codeword_out_train = encoder.predict(x_train)
sio.savemat('codeword_out.mat', {'codeword_outdoor': codeword_out_train})
# ******* 获取outdoor原始train数据通过indoor encoder后的码字
