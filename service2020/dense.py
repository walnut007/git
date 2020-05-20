# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D, BatchNormalization, AveragePooling2D, MaxPooling2D, Dropout, Concatenate
from tensorflow.keras import Model, Input
growth_rate = 12
inpt = Input(shape=(5, 5, 1))

def DenseLayer(x, nb_filter, bn_size=4, alpha=0.0, drop_rate=0.2):
    # Bottleneck layers
    x = BatchNormalization(axis=3)(x)
    x = tf.nn.leaky_relu(x, alpha=0.2)
    x = Conv2D(bn_size * nb_filter, (1, 1), strides=(1, 1), padding='same')(x)
    # Composite function
    x = BatchNormalization(axis=3)(x)
    x = tf.nn.leaky_relu(x, alpha=0.2)
    x = Conv2D(nb_filter, (3, 3), strides=(1, 1), padding='same')(x)
    if drop_rate:
        x = Dropout(drop_rate)(x)
    return x

def DenseBlock(x, nb_layers, growth_rate, drop_rate=0.2):
    for ii in range(nb_layers):
        conv = DenseLayer(x, nb_filter=growth_rate, drop_rate=drop_rate)
        x = Concatenate()([x, conv])
    return x


def TransitionLayer(x, compression=0.5, alpha=0.0, is_max=0):
    nb_filter = int(x.shape.as_list()[-1] * compression)
    x = BatchNormalization(axis=3)(x)
    x = tf.nn.leaky_relu(x, alpha=0.2)
    x = Conv2D(nb_filter, (1, 1), strides=(1, 1), padding='same')(x)
    if is_max != 0:
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    else:
        x = AveragePooling2D(pool_size=(2, 2), strides=2)(x)
    return x


# growth_rate = 12
# inpt = Input(shape=(32,32,3))
def densenet():
    x = Conv2D(growth_rate * 2, (3, 3), strides=1, padding='same')(inpt)
    x = BatchNormalization(axis=3)(x)
    x = tf.nn.leaky_relu(x, alpha=0.1)
    x = DenseBlock(x, 12, growth_rate, drop_rate=0.2)
    x = TransitionLayer(x)
    x = DenseBlock(x, 12, growth_rate, drop_rate=0.2)
    x = TransitionLayer(x)
    x = DenseBlock(x, 12, growth_rate, drop_rate=0.2)
    x = BatchNormalization(axis=3)(x)
    x = GlobalAveragePooling2D()(x)

    x = Dense(2, activation='softmax')(x)
    model = Model(inpt, x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model