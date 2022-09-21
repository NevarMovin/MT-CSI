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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

tf.reset_default_graph()

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.40
session = tf.Session(config=config)

epochs = 1000
batch_size = 200

size_of_trainingset = 50000  # dataset size
size_of_validationset = 15000

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
autoencoder.compile(optimizer='adam', loss='mse')

print(encoder.summary())
print(decoder.summary())
print(autoencoder.summary())

mat = sio.loadmat('dataset/DATA_Htrainin.mat')
x_train_1 = mat['HT']
mat = sio.loadmat('dataset/DATA_Hvalin.mat')
x_val_1 = mat['HT']

mat = sio.loadmat('dataset/DATA_Htrainout.mat')
x_train_2 = mat['HT']
mat = sio.loadmat('dataset/DATA_Hvalout.mat')
x_val_2 = mat['HT']

x_train_1 = x_train_1.astype('float32')
x_train_2 = x_train_2.astype('float32')
x_val_1 = x_val_1.astype('float32')
x_val_2 = x_val_2.astype('float32')

x_train_1 = x_train_1[0:size_of_trainingset]
x_train_2 = x_train_2[0:size_of_trainingset]
x_val_1 = x_val_1[0:size_of_validationset]
x_val_2 = x_val_2[0:size_of_validationset]

x_train = np.append(x_train_1, x_train_2, axis=0)
x_val = np.append(x_val_1, x_val_2, axis=0)

np.random.shuffle(x_train)
np.random.shuffle(x_val)

print(x_train.shape)
print(x_val.shape)

x_train = np.reshape(x_train, [len(x_train), img_channels, img_height, img_width])
x_val = np.reshape(x_val, [len(x_val), img_channels, img_height, img_width])


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses_train = []
        self.losses_val = []

    def on_batch_end(self, batch, logs={}):
        self.losses_train.append(logs.get('loss'))
        
    def on_epoch_end(self, epoch, logs={}):
        self.losses_val.append(logs.get('val_loss'))


file = 'CRNet_' + 'dim' + str(encoded_dim) + '_datasize' + str(size_of_trainingset)+'_Pre-train'
path = 'result/TensorBoard_%s' % file  # Tensorboard

save_dir = os.path.join(os.getcwd(), 'result/')
model_name = '%s_model.h5' % file
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             mode='min',
                             verbose=1,
                             save_best_only=True)

history = LossHistory()

callbacks = [checkpoint, history, tf.keras.callbacks.TensorBoard(log_dir=path)]

autoencoder.fit(x=x_train, y=x_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_val, x_val),
                callbacks=callbacks)


# model save
outfile = 'result/%s_model.h5' % file
autoencoder.save_weights(outfile)

print("Pre-training Finished!")
