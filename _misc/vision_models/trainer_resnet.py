from functools import partial
import tensorflow_datasets as tfds
import tensorflow as tf
import sklearn.metrics
import numpy as np
import pathlib


class Params:

    def __init__(self):

        # dataset
        self.SPLIT = ['train[:10%]', 'train[10%:25%]', 'train[25%:]']
        self.DS_NAME = 'tf_flowers'

        # model
        self.INPUT_SHAPE = [224, 224, 3]
        self.CONV1_PARAMS = {
            'filters': 64,
            'kernel_size': 7,
            'strides': 2,
            'input_shape': [224, 224, 3]
        }
        self.POOL1_PARAMS = {
            'pool_size': [3, 3],
            'strides': [2, 2],
            'padding': 'same'
        }
        self.CONV_FILTERS = [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3
        self.SOFTMAX_PARAMS = {
            'units': 5,  # flowers
            'activation': 'softmax'
        }

        # training
        self.N_EPOCHS = 25
        self.BATCH_SIZE = 32
        self.LOSS = 'sparse_categorical_crossentropy'
        self.METRICS = ['accuracy']
        self.OPT_ADAM = 'adam'
        self.EARLY_STOP_PARAMS = {'monitor': 'val_loss',
                                  'min_delta': 1e-3,
                                  'patience': 5,
                                  'verbose': 1}


params = Params()


DefaultConv2D = partial(tf.keras.layers.Conv2D,
                        kernel_size=3,
                        strides=1,
                        padding='same',
                        use_bias=False)


class ResidualUnit(tf.keras.layers.Layer):

    def __init__(self, filters=64, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.layers.ReLU()
        self.main_layers = [
            # DefaultConv2D -> ReLU -> BN -> DefaultConv2D -> ReLU -> BN

            # filters: we double filters in each block starting from 64
            # strides: we don't use max pooling, instead we use stride=2 in the beginning of a new block
            DefaultConv2D(filters=filters, strides=strides),
            tf.keras.layers.BatchNormalization(),
            # we need to put batch norm before activation, so we can't specify it in Conv2D
            self.activation,

            DefaultConv2D(filters=filters),
            tf.keras.layers.BatchNormalization()
            # no activation here - activation goes after we add skip connection
        ]

        # skip layers contain 1x1 conv - we need it only in the beginning of the new block
        # when stride == 2; we need to translate the input of the previous block to the
        # current block so we use a) strides == 2 to halve the size of feature map and
        # b) filters from the current block to double number of channels from previous block
        self.skip_layers = []
        if strides > 1:
            self.skip_layers.append(DefaultConv2D(filters=filters, strides=strides, kernel_size=1))
            self.skip_layers.append(tf.keras.layers.BatchNormalization())
            # still no activation here


    def call(self, inputs, **kwargs):
        # we need to move inputs through main layers and
        # through skip connection if necessary or just add it to the output of main layer
        # finally we need to add one more activation
        x, y = inputs, inputs
        for layer in self.main_layers:
            x = layer(x)

        for layer in self.skip_layers:
            y = layer(y)

        return self.activation(x + y)


class Resnet34:

    def __init__(self):
        # model
        self.model = tf.keras.models.Sequential()
        self._get_model()

    def _get_top(self):
        self.model.add(DefaultConv2D(**params.CONV1_PARAMS))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.ReLU())
        self.model.add(tf.keras.layers.MaxPool2D(**params.POOL1_PARAMS))

    def _get_conv_block(self):
        prev_filters = params.CONV_FILTERS[0]
        for filters in params.CONV_FILTERS:
            if filters != prev_filters: strides = 2
            else: strides = 1
            self.model.add(ResidualUnit(filters=filters, strides=strides))
            prev_filters = filters

    def _get_tail(self):
        self.model.add(tf.keras.layers.GlobalAvgPool2D())
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(**params.SOFTMAX_PARAMS))

    def _get_model(self):

        self._get_top()
        self._get_conv_block()
        self._get_tail()



