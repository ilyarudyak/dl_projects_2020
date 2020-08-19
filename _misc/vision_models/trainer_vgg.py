import tensorflow as tf
import sklearn.metrics
import numpy as np
import pathlib

# imports for mini VGG class
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, \
    Activation, Flatten, Dropout, Dense


class Params:

    def __init__(self):
        # model
        self.INPUT_SHAPE = [28, 28, 1]
        self.INITIAL_FILTERS = 32
        self.DROPOUT_CONV = .25
        self.DROPOUT_DENSE = .5
        self.POOL_SIZE = [2, 2]
        self.DENSE_PARAMS = {
            'units': 512,
            'activation': 'relu'
        }
        self.SOFTMAX_PARAMS = {
            'units': 10,
            'activation': 'softmax'
        }

        # training
        self.N_EPOCHS = 25
        self.BATCH_SIZE = 32
        self.LOSS = 'sparse_categorical_crossentropy'
        self.METRICS = ['accuracy']
        self.OPT_ADAM = 'adam'
        self.OPT_SGD = 'sgd'
        self.SGD_PARAMS = {'lr': 1e-2,
                           'momentum': 0.9,
                           'decay': 1e-2 / self.N_EPOCHS}
        self.ES_CALLBACK_PARAMS = {'monitor': 'val_loss',
                                   'min_delta': 1e-3,
                                   'patience': 7,
                                   'verbose': 1}

    # convolution params
    def get_conv_params(self, filters, is_first=False):
        conv_params = {
            'filters': filters,
            'kernel_size': (3, 3),
            'padding': 'same',
            'activation': 'relu'
        }

        if is_first:
            conv_params['input_shape'] = self.INPUT_SHAPE

        return conv_params


params = Params()


class FashionMNIST:

    @staticmethod
    def get_data(is_small=True):
        ((x_train, y_train), (x_val, y_val)) = tf.keras.datasets.fashion_mnist.load_data()

        x_train = x_train[..., np.newaxis] / 255.
        x_train = x_train.astype(np.float32)

        x_val = x_val[..., np.newaxis] / 255.
        x_val = x_val.astype(np.float32)

        x_train_small = x_train[:1000]
        y_train_small = y_train[:1000]

        x_val_small = x_val[:1000]
        y_val_small = y_val[:1000]

        if is_small:
            return x_train_small, y_train_small, x_val_small, y_val_small
        else:
            return x_train, y_train, x_val, y_val


class MiniVGG:

    def __init__(self,
                 is_batch_norm=True,
                 is_dropout=True,
                 is_gap=False):

        # net params
        self.is_batch_norm = is_batch_norm
        self.is_dropout = is_dropout
        self.is_gap = is_gap

        # model 
        self.model = Sequential()
        self._get_model()

    def _get_conv_block(self, n_block=1):

        conv_params_first = params.get_conv_params(
            filters=params.INITIAL_FILTERS * n_block,
            is_first=True)

        conv_params = params.get_conv_params(
            filters=params.INITIAL_FILTERS * n_block)

        # 1st conv sub block
        if n_block == 1:
            self.model.add(tf.keras.layers.Conv2D(**conv_params_first))
        else:
            self.model.add(tf.keras.layers.Conv2D(**conv_params))

        if self.is_batch_norm:
            self.model.add(tf.keras.layers.BatchNormalization())

        # 2nd conv sub block
        self.model.add(tf.keras.layers.Conv2D(**conv_params))

        if self.is_batch_norm:
            self.model.add(tf.keras.layers.BatchNormalization())

        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=params.POOL_SIZE))

        if self.is_dropout:
            self.model.add(tf.keras.layers.Dropout(rate=params.DROPOUT_CONV))

    def _get_top(self):

        # first (and only) set of FC => RELU layers
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(**params.DENSE_PARAMS))

        if self.is_batch_norm:
            self.model.add(tf.keras.layers.BatchNormalization())

        if self.is_dropout:
            self.model.add(tf.keras.layers.Dropout(rate=params.DROPOUT_DENSE))

        # softmax classifier
        self.model.add(tf.keras.layers.Dense(**params.SOFTMAX_PARAMS))

    def _get_top_gap(self):

        self.model.add(tf.keras.layers.GlobalAveragePooling2D())
        self.model.add(tf.keras.layers.Dense(**params.SOFTMAX_PARAMS))

    def _get_model(self):

        # first CONV => RELU => CONV => RELU => POOL layer set
        self._get_conv_block(n_block=1)

        # second CONV => RELU => CONV => RELU => POOL layer set
        self._get_conv_block(n_block=2)

        # first (and only) set of FC => RELU layers and classifier
        if self.is_gap:
            self._get_top_gap()
        else:
            self._get_top()


class MiniVGGOriginal:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # first CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(32, (3, 3), padding="same",
                         input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # second CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model


class Trainer:

    def __init__(self,
                 is_batch_norm=True,
                 is_dropout=True,
                 is_gap=False,
                 opt_type='adam',
                 is_small=True,
                 experiment_name='original-model',
                 seed=42):

        # trainer params
        self.is_batch_norm = is_batch_norm
        self.is_dropout = is_dropout
        self.is_gap = is_gap
        self.opt_type = opt_type
        self.is_small = is_small
        self.seed = seed

        # model
        self.model = None
        self._get_compiled_model()

        # callbacks
        self.log_dir = pathlib.Path('logs') / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.callbacks = [
            tf.keras.callbacks.EarlyStopping(**params.ES_CALLBACK_PARAMS),
            tf.keras.callbacks.TensorBoard(log_dir=self.log_dir)
        ]

    def _get_compiled_model(self):
        tf.keras.backend.clear_session()
        tf.random.set_seed(self.seed)

        self.model = MiniVGG(is_batch_norm=self.is_batch_norm,
                             is_dropout=self.is_dropout,
                             is_gap=self.is_gap).model

        if self.opt_type == params.OPT_ADAM:
            optimizer = tf.keras.optimizers.Adam()
        elif self.opt_type == params.OPT_SGD:
            optimizer = tf.keras.optimizers.SGD(**params.SGD_PARAMS)
        else:
            raise ValueError(f'wrong optimizer type: {self.opt_type}')

        self.model.compile(loss=params.LOSS,
                           optimizer=optimizer,
                           metrics=params.METRICS)

    def train(self):

        x_train, y_train, x_val, y_val = FashionMNIST.get_data(is_small=self.is_small)

        history = self.model.fit(x=x_train,
                                 y=y_train,
                                 validation_data=(x_val, y_val),
                                 batch_size=params.BATCH_SIZE,
                                 epochs=params.N_EPOCHS,
                                 callbacks=self.callbacks)
        return history


if __name__ == '__main__':
    vgg = MiniVGG().model
    vgg.summary()
