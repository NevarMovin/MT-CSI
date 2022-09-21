import tensorflow as tf
from keras.layers import Input, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU, Activation, Lambda
from keras.models import Model
from keras.callbacks import TensorBoard, Callback, ModelCheckpoint
from keras import optimizers
import scipy.io as sio 
import numpy as np
import math
import time
import hdf5storage # load Matlab data bigger than 2GB

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.reset_default_graph()

# 40%
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
session = tf.Session(config=config)

envir = 'outdoor' # 'indoor' or 'outdoor'

# training params
epochs = 500
batch_size = 200

size_of_trainingset = 25000  # dataset size
size_of_validationset = 7500

# image params
img_height = 32
img_width = 32
img_channels = 2 
img_total = img_height*img_width*img_channels

residual_num = 2
encoded_dim = 128  # compress rate=1/4->dim.=512, compress rate=1/16->dim.=128


# Bulid the autoencoder model of MRNet-1R
def residual_network(x, residual_num, encoded_dim):
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
        
        y = Conv2D(16, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
        y = BatchNormalization()(y)

        y = add([shortcut, y])
        y = LeakyReLU()(y)

        return y
    
    # encoder
    x = Conv2D(16, (3, 3), padding='same', data_format="channels_first")(x)
    x = add_common_layers(x)
    
    x = Conv2D(2, (3, 3), padding='same', data_format="channels_first")(x)
    x = add_common_layers(x)

    x = Reshape((img_total,))(x)
    encoded = Dense(encoded_dim, activation='linear')(x) # encoded result
    
    # decoder
    x = Dense(img_total, activation='linear')(encoded)
    x = Reshape((img_channels, img_height, img_width,))(x)

    x = Conv2D(16, (3, 3), padding='same', data_format="channels_first")(x)
    x = add_common_layers(x)

    for i in range(residual_num):
        x = residual_block_decoded(x)
    
    x = Conv2D(2, (3, 3), activation='sigmoid', padding='same', data_format="channels_first")(x)

    return x


image_tensor = Input(shape=(img_channels, img_height, img_width))
network_output = residual_network(image_tensor, residual_num, encoded_dim)
autoencoder = Model(inputs=[image_tensor], outputs=[network_output])

adam = optimizers.Adam(lr=0.0001)  # reset the learning rate

autoencoder.load_weights('result/MRNet-1R_dim128_Pre-train_model.h5')

autoencoder.layers[0].trainable = False
autoencoder.layers[1].trainable = False
autoencoder.layers[2].trainable = False
autoencoder.layers[3].trainable = False
autoencoder.layers[4].trainable = False
autoencoder.layers[5].trainable = False
autoencoder.layers[6].trainable = False
autoencoder.layers[7].trainable = False
autoencoder.layers[8].trainable = False

autoencoder.compile(optimizer=adam, loss='mse')
print(autoencoder.summary())


if envir == 'indoor':
    mat = sio.loadmat('dataset/DATA_Htrainin.mat')
    x_train = mat['HT']
    mat = sio.loadmat('dataset/DATA_Hvalin.mat')
    x_val = mat['HT']
    mat = sio.loadmat('dataset/DATA_Htestin.mat')
    x_test = mat['HT']

elif envir == 'outdoor':
    mat = sio.loadmat('dataset/DATA_Htrainout.mat')
    x_train = mat['HT']
    mat = sio.loadmat('dataset/DATA_Hvalout.mat')
    x_val = mat['HT']
    mat = sio.loadmat('dataset/DATA_Htestout.mat')
    x_test = mat['HT']

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train[0:size_of_trainingset]
x_val = x_val[0:size_of_validationset]

np.random.shuffle(x_train)
np.random.shuffle(x_val)

print(x_train.shape)
print(x_val.shape)

x_train = np.reshape(x_train, [len(x_train), img_channels, img_height, img_width])
x_val = np.reshape(x_val, [len(x_val), img_channels, img_height, img_width])
x_test = np.reshape(x_test, [len(x_test), img_channels, img_height, img_width])


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses_train = []
        self.losses_val = []

    def on_batch_end(self, batch, logs={}):
        self.losses_train.append(logs.get('loss'))
        
    def on_epoch_end(self, epoch, logs={}):
        self.losses_val.append(logs.get('val_loss'))
        

file = 'MRNet-1R_'+'dim'+str(encoded_dim)+'_FT-Outdoor'
path = 'result/TensorBoard_%s' % file

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

callbacks = [history, TensorBoard(log_dir = path)]

autoencoder.fit(x_train, x_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_val, x_val),
                callbacks=callbacks)


outfile = 'result/%s_model.h5' % file
autoencoder.save_weights(outfile)

# Testing data
autoencoder.load_weights(outfile)

x_hat = autoencoder.predict(x_test)

x_test_real = np.reshape(x_test[:, 0, :, :], (len(x_test), -1))
x_test_imag = np.reshape(x_test[:, 1, :, :], (len(x_test), -1))
x_test_C = x_test_real-0.5 + 1j*(x_test_imag-0.5)
x_hat_real = np.reshape(x_hat[:, 0, :, :], (len(x_test), -1))
x_hat_imag = np.reshape(x_hat[:, 1, :, :], (len(x_test), -1))
x_hat_C = x_hat_real-0.5 + 1j*(x_hat_imag-0.5)

power = np.sum(abs(x_test_C)**2, axis=1)

mse = np.sum(abs(x_test_C-x_hat_C)**2, axis=1)

print("In "+envir+" environment")
print("When dimension is", encoded_dim)
print("NMSE is ", 10*math.log10(np.mean(mse/power)))





















