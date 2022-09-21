import tensorflow as tf
from keras.layers import Input, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU, Activation, Lambda
from keras.models import Model
from keras.callbacks import TensorBoard, Callback, ModelCheckpoint
import scipy.io as sio
import numpy as np
import math
import time
import hdf5storage  # load Matlab data bigger than 2GB

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

tf.reset_default_graph()

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.20
session = tf.Session(config=config)

epochs = 1000
batch_size = 200

size_of_trainingset = 50000  # dataset size 训练集合容量
size_of_validationset = 15000  # dataset size 训练集合容量

# image params
img_height = 32
img_width = 32
img_channels = 2
img_total = img_height * img_width * img_channels
# network params
residual_num = 2
encoded_dim = 512  # 压缩率
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


image_tensor = Input(shape=(img_channels, img_height, img_width))
input_vector = Input(shape=(encoded_dim,))

encoder = Model(image_tensor, encoder_network(image_tensor))
decoder = Model(input_vector, decoder_network(input_vector))
autoencoder = Model(image_tensor, decoder(encoder(image_tensor)))
autoencoder.compile(optimizer='adam', loss='mse')
print(autoencoder.summary())

# Data loading  载入数据，调整数据格式
mat_1 = sio.loadmat('dataset/DATA_Htrainin.mat')
x_train_1 = mat_1['HT']  # array
mat_2 = sio.loadmat('dataset/DATA_Hvalin.mat')
x_val_1 = mat_2['HT']  # array

mat_3 = sio.loadmat('dataset/DATA_Htrainout.mat')
x_train_2 = mat_3['HT']  # array
mat_4 = sio.loadmat('dataset/DATA_Hvalout.mat')
x_val_2 = mat_4['HT']  # array

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

x_train = np.reshape(x_train, (
len(x_train), img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format
x_val = np.reshape(x_val, (
len(x_val), img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses_train = []
        self.losses_val = []

    def on_batch_end(self, batch, logs={}):
        self.losses_train.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs={}):
        self.losses_val.append(logs.get('val_loss'))


file = 'CsiNet_' + 'dim' + str(encoded_dim) +'_datasize' + str(size_of_trainingset) + '_Pre-train'
path = 'result/TensorBoard_%s' % file

save_dir = os.path.join(os.getcwd(), 'result/')
model_name = '%s_model.h5' % file
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=False)

history = LossHistory()

callbacks = [checkpoint, history, TensorBoard(log_dir=path)]

autoencoder.fit(x_train, x_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_val, x_val),
                callbacks=callbacks)

outfile = 'result/%s_model.h5' % file
autoencoder.save_weights(outfile)

print("Pre-training Finished!")



