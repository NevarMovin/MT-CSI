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

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.reset_default_graph()

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.20
session = tf.Session(config=config)

envir = 'indoor'

img_height = 32
img_width = 32
img_channels = 2
img_total = img_height * img_width * img_channels

residual_num = 2
encoded_dim = 512
# compress rate=1/4->dim.=512, compress rate=1/16->dim.=128, compress rate=1/32->dim.=64, compress rate=1/64->dim.=32


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


image_tensor = Input(shape=(img_channels, img_height, img_width))  # 设定网络输入 编码器输入
input_vector = Input(shape=(encoded_dim,))  # 译码器输入

encoder = Model(image_tensor, encoder_network(image_tensor))
decoder = Model(input_vector, decoder_network(input_vector))
autoencoder = Model(image_tensor, decoder(encoder(image_tensor)))  # 使用inputs与outputs建立函数链式模型

adam = optimizers.Adam(lr=0.0001)  # reset the learning rate

encoder.layers[0].trainable = False
encoder.layers[1].trainable = False
encoder.layers[3].trainable = False
encoder.layers[4].trainable = False
encoder.layers[5].trainable = False

encoder.compile(optimizer=adam, loss='mse')
decoder.compile(optimizer=adam, loss='mse')
autoencoder.compile(optimizer=adam, loss='mse')  # 训练参数

dataset_path = '/home/hzl/ZBY/dataset'

if envir == 'indoor':
    mat = sio.loadmat(dataset_path + '/DATA_Htestin.mat')
    x_test = mat['HT']  # array

elif envir == 'outdoor':
    mat = sio.loadmat(dataset_path + '/DATA_Htestout.mat')
    x_test = mat['HT']  # array


x_test = x_test.astype('float32')

x_test = np.reshape(x_test, (
len(x_test), img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format

autoencoder.load_weights('result/CsiNet_MT_dim512_indoor_model.h5')

tStart = time.time()
x_hat = autoencoder.predict(x_test)
tEnd = time.time()
print("It cost %f sec" % ((tEnd - tStart) / x_test.shape[0]))

x_test_real = np.reshape(x_test[:, 0, :, :], (len(x_test), -1))
x_test_imag = np.reshape(x_test[:, 1, :, :], (len(x_test), -1))
x_test_C = x_test_real - 0.5 + 1j * (x_test_imag - 0.5)
x_hat_real = np.reshape(x_hat[:, 0, :, :], (len(x_hat), -1))
x_hat_imag = np.reshape(x_hat[:, 1, :, :], (len(x_hat), -1))
x_hat_C = x_hat_real - 0.5 + 1j * (x_hat_imag - 0.5)

print("x_test_C shape is ", x_test_C.shape)
power = np.sum(abs(x_test_C) ** 2, axis=1)

mse = np.sum(abs(x_test_C - x_hat_C) ** 2, axis=1)

print("In " + envir + " environment")
print("When dimension is", encoded_dim)
print("MSE is ", 10 * math.log10(np.mean(mse)))
print("NMSE is ", 10 * math.log10(np.mean(mse / power)))

