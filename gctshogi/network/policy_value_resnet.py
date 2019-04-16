import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization, Add
from tensorflow.keras.layers import Input, Activation, Flatten
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers

import shogi

from gctshogi.common import *

# RES_BLOCKS = 18
RES_BLOCKS = 5
FILTERS = 192
FCL_UNITS = 192

def conv_layer(inputs,
               filters,
               kernel_size=3,
               activation='relu',
               use_bias=True):

    x =  Conv2D(filters,
                kernel_size=kernel_size,
                padding='same',
                data_format='channels_first',
                kernel_initializer='he_normal',
                use_bias=use_bias)(inputs)
    x = BatchNormalization(axis=1)(x)
    if activation is not None:
        x = Activation(activation)(x)
    return x

class PolicyValueResnet():
    def __init__(self):
        self.model = self._build_model()
    
    def predict(self, x):
        return self.model.predict(x)
    
    def _build_model(self):
        inputs = Input(shape=(104, 9, 9))
        
        # layer1
        x = conv_layer(inputs, filters=FILTERS, use_bias=False)

        # layer2 - 39
        for i in range(RES_BLOCKS):
            y = conv_layer(x, filters=FILTERS, use_bias=False)
            y = conv_layer(y, filters=FILTERS, use_bias=False, activation=None)
            x = Add()([x, y])
            x = Activation('relu')(x)
        
        # policy network
        # layer40
        ph = conv_layer(x, filters=FILTERS)
        ph = Conv2D(MOVE_DIRECTION_LABEL_NUM,
                    kernel_size=3,
                    padding='same',
                    data_format='channels_first',
                    kernel_initializer='he_normal')(ph)
        ph = Flatten(name='policy_head')(ph)

        # value network
        # layer13
        vh = conv_layer(x, filters=1, kernel_size=1)
        vh = Flatten()(vh)
        vh = Dense(FCL_UNITS, activation='relu', kernel_initializer='he_normal')(vh)
        vh = Dense(1, activation="tanh", kernel_initializer='he_normal', name='value_head')(vh)

        model = Model(inputs=inputs, outputs=[ph, vh])
        
        return model

if __name__ == '__main__':
    network = PolicyValueResnet()
    model = network.model
