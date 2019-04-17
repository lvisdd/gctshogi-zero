import pyximport; pyximport.install()
# import numpy as np
# cimport numpy as np
# cimport cython

import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Activation, Flatten
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD

from gctshogi.common import *
from gctshogi.network.policy_value_resnet import *
from gctshogi.features import *
from gctshogi.read_kifu import *

import shogi
import shogi.CSA

import random
import copy

import os
import pickle

# mini batch
def mini_batch(positions, i, batchsize):
    mini_batch_data = []
    mini_batch_move = []
    mini_batch_win = []
    for b in range(batchsize):
        features, move, win = make_features(positions[i + b])
        mini_batch_data.append(features)
        mini_batch_move.append(move)
        mini_batch_win.append(win)

    return (np.array(mini_batch_data, dtype=np.float32),
            to_categorical(mini_batch_move, NUM_CLASSES),
            np.array(mini_batch_win, dtype=np.int32).reshape((-1, 1)))

# data generator
def datagen(positions):
    while True:
        positions_shuffled = random.sample(positions, len(positions))
        for i in range(0, len(positions_shuffled) - args.batchsize, args.batchsize):
            x, t1, t2 = mini_batch(positions_shuffled, i, args.batchsize)
            yield (x, {'policy_head': t1, 'value_head': t2})

def categorical_crossentropy(y_true, y_pred):
    return tf.keras.backend.categorical_crossentropy(y_true, y_pred, from_logits=True)

def categorical_accuracy(y_true, y_pred):
    return tf.keras.metrics.categorical_accuracy(y_true, tf.nn.softmax(y_pred))

def binary_accuracy(y_true, y_pred):
    return tf.keras.metrics.binary_accuracy(tf.keras.backend.round((y_true + 1) / 2), y_pred, threshold=0)

def compile(model, lr, weight_decay):

    # add weight decay
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(tf.keras.regularizers.l2(weight_decay)(layer.kernel))

    model.compile(optimizer=tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9),
                  loss={'policy_head': categorical_crossentropy, 'value_head': 'mse'},
                  metrics={'policy_head': categorical_accuracy, 'value_head': binary_accuracy})

def train(positions_train, positions_test, model, batchsize, steps, test_steps, window_size):
    # model.fit_generator(datagen(positions_train), steps,
    #           validation_data=datagen(positions_test), validation_steps=test_steps)
    model.fit_generator(datagen(positions_train), int(len(positions_train) / batchsize),
              validation_data=datagen(positions_test), validation_steps=int(len(positions_test) / batchsize))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deep Learning Shogi')
    parser.add_argument('train_kifu', type=str, help='train kifu hcpe')
    parser.add_argument('test_kifu', type=str, help='test kifu hcpe')
    parser.add_argument('model', default='model.h5')
    parser.add_argument('--resume', '-r')
    parser.add_argument('--batchsize', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--test_steps', type=int, default=1000)
    parser.add_argument('--window_size', type=int, default=1000000)
    parser.add_argument('--weight_decay', type=int, default=1e-4)
    parser.add_argument('--use_tpu', '-t', action='store_true', help='Use TPU model instead of CPU')
    args = parser.parse_args()

    if args.resume is not None:
        model = load_model(args.resume)
    else:
        network = PolicyValueResnet()
        model = network.model

    if args.use_tpu:
        # TPU
        import tensorflow as tf
        from tensorflow.contrib.tpu.python.tpu import keras_support
        tpu_grpc_url = "grpc://"+os.environ["COLAB_TPU_ADDR"]
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu_grpc_url)
        strategy = keras_support.TPUDistributionStrategy(tpu_cluster_resolver)
        model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)

    compile(model, args.lr, args.weight_decay)

    positions_test = read_kifu_from_hcpe(args.test_kifu)

    for i in range(2):
        positions_train = read_kifu_from_hcpe(args.train_kifu, i+1)

        train(positions_train,
              positions_test,
              model,
              args.batchsize,
              args.steps,
              args.test_steps,
              args.window_size
              )

        model.save(args.model)
        positions_train.clear()
