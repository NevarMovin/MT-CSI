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

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.reset_default_graph()

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.20
session = tf.Session(config=config)

envir = 'indoor'
batch_size = 200

# Network Settings
img_height = 32
img_width = 32
img_channels = 2
img_total = img_height*img_width*img_channels
encoded_dim = 512

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
    encoded = Dense(encoded_dim, activation='linear')(x) # Note that CRNet has no quantization operation, so the linear activation function is used

    return encoded


def decoder_network(encoded):
    # decoder
    x = Dense(img_total, activation='linear')(encoded) # Due to the settings in CRNet, the linear activation function is adopted
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

print(encoder.summary())
print(decoder.summary())
print(autoencoder.summary())

autoencoder.load_weights('result/CRNet_MT_dim512_indoor_model.h5')

if envir == 'indoor':
    mat = sio.loadmat('dataset/DATA_Htestin.mat')
    x_test = mat['HT']

elif envir == 'outdoor':
    mat = sio.loadmat('dataset/DATA_Htestout.mat')
    x_test = mat['HT']


x_test = x_test.astype('float32')

x_test = np.reshape(x_test, [len(x_test), img_channels, img_height, img_width])

# Testing data

tStart = time.time()
x_hat = autoencoder.predict(x_test, batch_size=batch_size)
tEnd = time.time()
print("It cost %f sec" % ((tEnd - tStart)/x_test.shape[0]))

x_test_real = np.reshape(x_test[:, 0, :, :], (len(x_test), -1))
x_test_imag = np.reshape(x_test[:, 1, :, :], (len(x_test), -1))
x_test_C = x_test_real-0.5 + 1j*(x_test_imag-0.5)
x_hat_real = np.reshape(x_hat[:, 0, :, :], (len(x_hat), -1))
x_hat_imag = np.reshape(x_hat[:, 1, :, :], (len(x_hat), -1))
x_hat_C = x_hat_real-0.5 + 1j*(x_hat_imag-0.5)

power = np.sum(abs(x_test_C)**2, axis=1)

mse = np.sum(abs(x_test_C-x_hat_C)**2, axis=1)

print("In "+envir+" environment")
print("When dimension is", encoded_dim)
print("NMSE is ", 10*math.log10(np.mean(mse/power)))
