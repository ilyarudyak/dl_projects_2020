from functools import partial
import tensorflow_datasets as tfds
import tensorflow as tf
import sklearn.metrics
import numpy as np
import pathlib
import functools


class ConvWithBatchNorm(tf.keras.layers.Conv2D):
    """ Convolutional layer with batch normalization"""

    def __init__(self,
                 activation='relu',
                 name='convbn',
                 **kwargs):
        """
        Initialize the layer.
        :param activation:   Activation function (name or callable)
        :param name:         Name suffix for the sub-layers.
        :param kwargs:       Mandatory and optional parameters of tf.keras.layers.Conv2D
        """

        self.activation = tf.keras.layers.Activation(
            activation, name=name + '_act') if activation is not None else None

        super().__init__(activation=None, name=name + '_c', **kwargs)

        self.batch_norm = tf.keras.layers.BatchNormalization(axis=-1,
                                                             name=name + '_bn')

    def call(self, inputs, training=None):
        """
        Call the layer.
        :param inputs:         Input tensor to process
        :param training:       Flag to let TF knows if it is a training iteration or not
                               (this will affect the behavior of BatchNorm)
        :return:               Convolved tensor
        """
        x = super().call(inputs)
        x = self.batch_norm(x, training=training)
        if self.activation is not None:
            x = self.activation(x)
        return x


class SkipConnection(tf.keras.layers.Layer):
    """ Layer to merge the original tensor and the residual one in residual blocks"""

    def __init__(self, name='block', **kwargs):
        """
        Initialize the layer.
        :param activation:   Activation function (name or callable)
        :param name:         Name suffix for the sub-layers.
        :param kwargs:       Optional parameters of tf.keras.layers.Conv2D
        """

        super().__init__(name=name)
        self.shortcut = None
        self.kwargs = kwargs

    def build(self, input_shape):
        x_shape = input_shape[0]
        x_residual_shape = input_shape[1]
        if x_shape[1] == x_residual_shape[1] and \
                x_shape[2] == x_residual_shape[2] and \
                x_shape[3] == x_residual_shape[3]:
            self.shortcut = functools.partial(tf.identity, name=self.name + '_shortcut')
        else:
            strides = (
                int(round(x_shape[1] / x_residual_shape[1])),  # vertical stride
                int(round(x_shape[2] / x_residual_shape[2]))  # horizontal stride
            )
            x_residual_channels = x_residual_shape[3]
            self.shortcut = ConvWithBatchNorm(
                filters=x_residual_channels, kernel_size=(1, 1), strides=strides,
                activation=None, name=self.name + '_shortcut_c', **self.kwargs)

    def call(self, inputs):
        """
        Call the layer.
        :param inputs:         Tuple of two input tensors to merge
        :return:               Merged tensor
        """
        x, x_residual = inputs

        x_shortcut = self.shortcut(x)
        x_merge = x_shortcut + x_residual
        return x_merge


class BottleneckBlock(tf.keras.Model):
    """ Residual block with bottleneck, recommended for deep ResNets (depth > 34)"""

    def __init__(self, filters=16, kernel_size=1, strides=1, activation='relu',
                 kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                 name='res_basic', **kwargs):
        """
        Initialize the block.
        :param filters:                 Number of filters
        :param kernel_size:             Kernel size
        :param strides:                 Convolution strides
        :param activation:              Activation function (name or callable)
        :param kernel_initializer:      Kernel initialisation method name
        :param kernel_regularizer:      Kernel regularizer
        :param name:                    Name suffix for the sub-layers.
        :param kwargs:                  Optional parameters of tf.keras.layers.Conv2D
        """
        super().__init__(name=name)

        self.conv_0 = ConvWithBatchNorm(
            filters=filters, kernel_size=1, activation=activation, padding='valid',
            kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
            strides=1, name=name + '_cb_0', **kwargs)

        self.conv_1 = ConvWithBatchNorm(
            filters=filters, kernel_size=kernel_size, activation=activation, padding='same',
            kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
            strides=strides, name=name + '_cb_1', **kwargs)

        self.conv_2 = ConvWithBatchNorm(
            filters=4 * filters, kernel_size=1, activation=None, padding='valid',
            kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
            strides=1, name=name + '_cb_2', **kwargs)

        self.merge = SkipConnection(
            kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
            name=name)

        self.activation = tf.keras.layers.Activation(activation, name=name + '_act')

    def call(self, inputs, training=None):
        """
        Call the layer.
        :param inputs:         Input tensor to process
        :param training:       Flag to let TF knows if it is a training iteration or not
                               (this will affect the behavior of BatchNorm)
        :return:               Block output tensor
        """
        x = inputs
        # Residual path:
        x_residual = self.conv_0(x, training=training)
        x_residual = self.conv_1(x_residual, training=training)
        x_residual = self.conv_2(x_residual, training=training)

        # Merge residual result with original tensor:
        x_merge = self.merge([x, x_residual])
        x_merge = self.activation(x_merge)
        return x_merge


# as we may see it subclasses a model, not a layer
# and the explanation of why it's even possible we
# may find in Geron book: models can be defined and
# used exactly like layers.
class ResidualUnit(tf.keras.models.Sequential):
    """ Macro-block, chaining multiple residual blocks (as a Sequential model)"""

    def __init__(self, block_class=BottleneckBlock, repetitions=3,
                 filters=16, kernel_size=1, strides=1, activation='relu',
                 kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                 name='res_macroblock', **kwargs):
        """
        Initialize the block.
        :param block_class:             Block class to be used.
        :param repetitions:             Number of times the block should be repeated inside.
        :param filters:                 Number of filters
        :param kernel_size:             Kernel size
        :param strides:                 Convolution strides
        :param activation:              Activation function (name or callable)
        :param kernel_initializer:      Kernel initialisation method name
        :param kernel_regularizer:      Kernel regularizer
        :param name:                    Name suffix for the sub-layers.
        :param kwargs:                  Optional parameters of tf.keras.layers.Conv2D
        """
        super().__init__(
            [block_class(
                filters=filters, kernel_size=kernel_size, activation=activation,
                strides=strides if i == 0 else 1, name="{}_{}".format(name, i),
                kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
                for i in range(repetitions)],
            name=name)


# noinspection PyListCreation
class ResNet(tf.keras.models.Sequential):
    """ ResNet model for classification"""

    def __init__(self,
                 input_shape,
                 num_classes=1000,
                 block_class=BottleneckBlock,
                 repetitions=None,
                 kernel_initializer='he_normal',
                 kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                 name=None):
        """
        Initialize a ResNet model for classification.
        :param input_shape:             Input shape (e.g. (224, 224, 3))
        :param num_classes:             Number of classes to predict
        :param block_class:             Block class to be used
        :param repetitions:             List of repetitions for each macro-blocks the network should contain
        :param kernel_initializer:      Kernel initialisation method name
        :param kernel_regularizer:      Kernel regularizer
        :param name:                    Model's name
        :return:                        ResNet model.
        """

        def _get_layers():
            filters = 64
            strides = 2

            layers = [tf.keras.layers.Input(shape=input_shape,
                                            name='input')]
            layers.append(ConvWithBatchNorm(filters=filters,
                                            kernel_size=7,
                                            activation='relu',
                                            padding='same',
                                            strides=strides,
                                            kernel_initializer=kernel_initializer,
                                            kernel_regularizer=kernel_regularizer,
                                            name='conv'))
            layers.append(tf.keras.layers.MaxPooling2D(pool_size=3,
                                                       strides=strides,
                                                       padding='same',
                                                       name='max_pool'))

            residual_units = [ResidualUnit(block_class=block_class,
                                           repetitions=repetition,
                                           filters=min(filters * (2 ** i), 1024),
                                           kernel_size=3,
                                           activation='relu',
                                           strides=strides if i != 0 else 1,
                                           name=f'conv{i+1}',
                                           kernel_initializer=kernel_initializer,
                                           kernel_regularizer=kernel_regularizer)
                              for i, repetition in enumerate(repetitions)
                              ]
            layers += residual_units

            layers.append(tf.keras.layers.GlobalAveragePooling2D(name='avg_pool'))
            layers.append(tf.keras.layers.Dense(units=num_classes,
                                                kernel_initializer=kernel_initializer,
                                                name='predictions',
                                                activation='softmax'))
            return layers

        super().__init__(layers=_get_layers(), name=name)


class ResNet50(ResNet):
    def __init__(self,
                 input_shape,
                 name='resnet50'):
        # the only 2 arguments we need to provide:
        # name and repetitions; but we saved input_shape
        # here just in case
        super().__init__(input_shape=input_shape,
                         name=name,
                         repetitions=(3, 4, 6, 3))
