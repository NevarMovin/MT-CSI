# -*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU, ReLU, Lambda, dot
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation
import scipy.io as sio
import numpy as np
import math
import time
import os

# Enable GPU
tf.compat.v1.reset_default_graph()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True   
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)


# image params
img_height = 32
img_width = 32
img_channels = 2
img_total = img_height * img_width * img_channels  # 2048
# network params
# envir = 'indoor'  # 'indoor' or 'outdoor'
encoded_dim1 = 256  # compress rate=1/8
encoded_dim2 = 128  # compress rate=1/16
encoded_dim3 = 64  # compress rate=1/32
encoded_dim4 = 32   # compress rate=1/64

batchsize = 200
limit1 = 50
limit2 = 100
limit3 = 150


class UnfoldLayer(tf.keras.layers.Layer):
    def __init__(self, Kh, Kw, stride=1, **kwargs):
        self.Kh = Kh
        self.Kw = Kw
        self.stride = stride
        super(UnfoldLayer, self).__init__(**kwargs)

    def call(self, x, **kwargs):
        return self.Unfold(x)

    def Unfold(self, x):
        _, C, H, W = x.shape
        Co = C * self.Kw * self.Kh
        x_out = x[:, :, 0:self.Kh, 0:self.Kw]
        x_out = tf.reshape(x_out, [-1, Co, 1])
        for i in range(H - (self.Kh - 1)):
            for j in range(W - (self.Kw - 1)):
                Hstart = i * self.stride
                Hend = Hstart + self.Kh
                Wstart = j * self.stride
                Wend = Wstart + self.Kw
                xi = x[:, :, Hstart:Hend, Wstart:Wend]
                xi = tf.reshape(xi, [-1, Co, 1])
                x_out = tf.concat([x_out, xi], axis=2)
        x_out = x_out[:, :, 1:]
        return x_out

    def get_config(self):
        base_config = super(UnfoldLayer, self).get_config()
        base_config['Kw'] = self.Kw
        return base_config


def SA_block(y, k, rel_planes, mid_planes, out_planes):
    # pre
    shortcut = y
    y = BatchNormalization()(y)
    y = ReLU()(y)
    # y1
    y1 = Conv2D(rel_planes, (1, 1), padding='same', use_bias=False, data_format='channels_first',
                kernel_initializer='he_normal', name='SA_y1conv')(y)
    y1 = Reshape((rel_planes, 1, -1,))(y1)

    # y2
    y2 = Conv2D(rel_planes, (1, 1), padding='same', use_bias=False, data_format='channels_first',
                kernel_initializer='he_normal', name='SA_y2conv')(y)
    y2 = Lambda(lambda e: tf.pad(e, [[0, 0], [0, 0], [1, 1], [1, 1]], mode='SYMMETRIC'))(y2)
    y2 = UnfoldLayer(Kh=k, Kw=k, stride=1)(y2)
    y2 = Lambda(lambda e: tf.expand_dims(e, axis=2))(y2)

    # y3
    y3 = Conv2D(mid_planes, (1, 1), padding='same', use_bias=False, data_format='channels_first',
                kernel_initializer='he_normal', name='SA_y3conv')(y)
    y3 = BatchNormalization()(y3)
    y3 = ReLU()(y3)

    # cat y12
    y12 = keras.layers.concatenate([y1, y2], axis=1)
    # convW
    y12 = BatchNormalization()(y12)
    y12 = ReLU()(y12)
    y12 = Conv2D(mid_planes, (1, 1), padding='same', use_bias=False, data_format='channels_first',
                 kernel_initializer='he_normal', name='SA_conw1')(y12)
    y12 = BatchNormalization()(y12)
    y12 = ReLU()(y12)
    y12 = Conv2D(mid_planes, (1, 1), padding='same', use_bias=False, data_format='channels_first',
                 kernel_initializer='he_normal', name='SA_conw2')(y12)
    y12 = Reshape((mid_planes, y3.shape[2], y3.shape[3],))(y12)

    # multi
    y = tf.multiply(y12, y3)
    y = BatchNormalization()(y)
    y = ReLU()(y)
    y = Conv2D(out_planes, (3, 3), padding='same', use_bias=False, data_format='channels_first',
               kernel_initializer='he_normal', name='SA_multi_conv')(y)
    y = add([shortcut, y])
    y = BatchNormalization()(y)
    y = ReLU()(y)
    return y


def residual_block_decoded(y):
    y = Conv2D(32, (3, 3), padding='same', data_format="channels_first")(y)
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)

    y = Conv2D(16, (3, 3), padding='same', data_format="channels_first")(y)
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)

    y = Conv2D(8, (3, 3), padding='same', data_format="channels_first")(y)
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)
    return y


def DenseRefine(x):
    ip = x
    x1 = ip
    x = residual_block_decoded(ip)
    x2 = x
    x = keras.layers.concatenate([x1, x2], axis=1)

    x = residual_block_decoded(x)
    x3 = x
    x = keras.layers.concatenate([x1, x2, x3], axis=1)

    x = residual_block_decoded(x)
    x4 = x
    x = keras.layers.concatenate([x1, x2, x3, x4], axis=1)

    x = residual_block_decoded(x)
    x5 = x
    x = keras.layers.concatenate([x1, x2, x3, x4, x5], axis=1)

    x = residual_block_decoded(x)
    x6 = x
    x = keras.layers.concatenate([x1, x2, x3, x4, x5, x6], axis=1)
    return x


# Bulid the autoencoder model of SALDR
def residual_network(x):
    ip = x
    # encoder Net
    x1 = Conv2D(16, (1, 1), padding='same', use_bias=False, data_format='channels_first',
                kernel_initializer='he_normal', name='conv00')(ip)
    x1 = BatchNormalization(name='encoder_bn0')(x1)
    x1 = LeakyReLU(name='encoder_leakyrulu0')(x1)
    x1 = Conv2D(16, (1, 1), padding='same', use_bias=False, data_format='channels_first',
                kernel_initializer='he_normal', name='conv01')(x1)       #
    x1 = SA_block(x1, k=3, rel_planes=4, mid_planes=8, out_planes=16)

    x1 = Conv2D(32, (3, 3), padding='same', data_format="channels_first", name='encoder_conv1')(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU()(x1)

    x1 = Conv2D(16, (3, 3), padding='same', data_format="channels_first", name='encoder_conv2')(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU()(x1)

    x1 = Conv2D(8, (3, 3), padding='same', data_format="channels_first", name='encoder_conv3')(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU()(x1)

    x1 = Conv2D(2, (3, 3), padding='same', data_format="channels_first", name='encoder_conv4')(x1)  #
    # x = tf.add(x1, ip)         # concate or add
    x = keras.layers.concatenate([x1, ip], axis=1)
    x = Conv2D(2, (1, 1), padding='same', data_format="channels_first", name='encoder_conv5')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Reshape((img_total,), name='encoder_reshape1')(x)

    encoded1 = Dense(encoded_dim1, activation='linear', name='encoder_cr8_dense')(x)
    # sigmoid: Limit the output to [0, 1] for quantization
    encoded1 = Activation('sigmoid')(encoded1)
    encoded2 = Dense(encoded_dim2, activation='linear', name='encoder_cr16_dense')(encoded1)
    encoded2 = Activation('sigmoid')(encoded2)
    encoded3 = Dense(encoded_dim3, activation='linear', name='encoder_cr32_dense')(encoded2)
    encoded3 = Activation('sigmoid')(encoded3)
    encoded4 = Dense(encoded_dim4, activation='linear', name='encoder_cr64_dense')(encoded3)
    encoded4 = Activation('sigmoid')(encoded4)

    # decoder
    x1 = Dense(img_total, activation='linear', name='decoder_cr8_dense')(encoded1)
    x2 = Dense(img_total, activation='linear', name='decoder_cr16_dense')(encoded2)
    x3 = Dense(img_total, activation='linear', name='decoder_cr32_dense')(encoded3)
    x4 = Dense(img_total, activation='linear', name='decoder_cr64_dense')(encoded4)

    # decoder Net
    # decoder Net for CR:8
    x1 = Reshape((img_channels, img_height, img_width,))(x1)  # reshape to real channel and im channel
    x1 = DenseRefine(x1)
    x1 = Conv2D(2, (3, 3), activation='sigmoid', padding='same', data_format="channels_first")(x1)

    # decoder Net for CR:16
    x2 = Reshape((img_channels, img_height, img_width,))(x2)  # reshape to real channel and im channel
    x2 = DenseRefine(x2)
    x2 = Conv2D(2, (3, 3), activation='sigmoid', padding='same', data_format="channels_first")(x2)

    # decoder Net for CR:32
    x3 = Reshape((img_channels, img_height, img_width,))(x3)  # reshape to real channel and im channel
    x3 = DenseRefine(x3)
    x3 = Conv2D(2, (3, 3), activation='sigmoid', padding='same', data_format="channels_first")(x3)
    # decoder Net for CR:32
    x4 = Reshape((img_channels, img_height, img_width,))(x4)  # reshape to real channel and im channel
    x4 = DenseRefine(x4)
    x4 = Conv2D(2, (3, 3), activation='sigmoid', padding='same', data_format="channels_first")(x4)

    x = keras.layers.concatenate([x1, x2, x3, x4], axis=1)
    return x

# Total loss function
def SM_loss(y_actual, y_pred):    
    y_pred1 = y_pred[:, 0:2, :, :]
    y_pred2 = y_pred[:, 2:4, :, :]
    y_pred3 = y_pred[:, 4:6, :, :]
    y_pred4 = y_pred[:, 6:8, :, :]

    loss_c8 = tf.reduce_mean(tf.square(y_actual-y_pred1))
    loss_c16 = tf.reduce_mean(tf.square(y_actual - y_pred2))
    loss_c32 = tf.reduce_mean(tf.square(y_actual - y_pred3))
    loss_c64 = tf.reduce_mean(tf.square(y_actual - y_pred4))

    sm_loss = 3*loss_c64 + 4*loss_c32 + 7*loss_c16 + 25*loss_c8
    sm_loss = sm_loss/39
    return sm_loss


image_tensor = Input(shape=(img_channels, img_height, img_width))
network_output = residual_network(image_tensor)
autoencoder = Model(inputs=[image_tensor], outputs=[network_output])
adam = keras.optimizers.Adam(lr=0.001)
autoencoder.compile(optimizer=adam, loss=SM_loss)
print(autoencoder.summary())

mat_1 = sio.loadmat('dataset/DATA_Htrainin.mat')
x_train_1 = mat_1['HT']  # array
mat_2 = sio.loadmat('dataset/DATA_Hvalin.mat')
x_val_1 = mat_2['HT']  # array

mat_3 = sio.loadmat('dataset/DATA_Htrainout.mat')
x_train_2 = mat_3['HT']  # array
mat_4 = sio.loadmat('dataset/DATA_Hvalout.mat')
x_val_2 = mat_4['HT']  # array

size_of_trainingset = 50000
size_of_validationset = 15000

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

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')


x_train = np.reshape(x_train, (
    len(x_train), img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format
x_val = np.reshape(x_val, (
    len(x_val), img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format


reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.6, patience=10, mode='auto',
                                              verbose=0, min_delta=3e-6, cooldown=0, min_lr=0)
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=80, mode='auto',
                                               verbose=0, min_delta=3e-6)

ckfile = 'SALDR_Pre-train'
filepath = "result/ck_%s.h5" % ckfile
ck = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto',
                                     save_weights_only=True)

autoencoder.fit(x_train, x_train,
                epochs=1000,
                batch_size=batchsize,
                shuffle=True,
                validation_data=(x_val, x_val),
                callbacks=[ck, early_stopping])

print("Pre-training training finish.")
