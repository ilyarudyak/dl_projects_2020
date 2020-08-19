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
        # model ### UPDATED ###
        self.INPUT_SHAPE = [28, 28, 1]
        self.CONV1_FILTERS = 20
        self.CONV2_FILTERS = 50
        self.POOL_PARAMS = {
            'pool_size': [2, 2],
            'strides': [2, 2]
        }
        self.DENSE_PARAMS = {
            'units': 512,
            'activation': 'relu'
        }
        self.SOFTMAX_PARAMS = {
            'units': 10,
            'activation': 'softmax'
        }

        # training ### UPDATED ###
        self.N_EPOCHS = 25
        self.BATCH_SIZE = 32
        self.LOSS = 'sparse_categorical_crossentropy'
        self.METRICS = ['accuracy']
        self.OPT_ADAM = 'adam'
        self.EARLY_STOP_PARAMS = {'monitor': 'val_loss',
                                  'min_delta': 1e-3,
                                  'patience': 5,
                                  'verbose': 1}

    # convolution params ### UPDATED ###
    def get_conv_params(self, is_first=False):
        conv_params = {
            'kernel_size': (5, 5),
            'padding': 'same',
            'activation': 'relu'
        }

        if is_first:
            conv_params['filters'] = self.CONV1_FILTERS
            conv_params['input_shape'] = self.INPUT_SHAPE
        else:
            conv_params['filters'] = self.CONV2_FILTERS

        return conv_params


params = Params()


class MNIST:

    @staticmethod
    def get_data(is_small=True):
        ((x_train, y_train), (x_val, y_val)) = tf.keras.datasets.mnist.load_data()

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


class LeNet:

    def __init__(self):
        # model
        self.model = Sequential()
        self._get_model()

    def _get_conv_block(self, is_first=False):
        conv_params = params.get_conv_params(is_first=is_first)
        self.model.add(tf.keras.layers.Conv2D(**conv_params))
        self.model.add(tf.keras.layers.MaxPooling2D(**params.POOL_PARAMS))

    def _get_top(self):
        # first (and only) set of FC => RELU layers
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(**params.DENSE_PARAMS))

        # softmax classifier
        self.model.add(tf.keras.layers.Dense(**params.SOFTMAX_PARAMS))

    def _get_model(self):
        # first CONV => RELU => CONV => RELU => POOL layer set
        self._get_conv_block(is_first=True)

        # second CONV => RELU => CONV => RELU => POOL layer set
        self._get_conv_block()

        # first (and only) set of FC => RELU layers and classifier
        self._get_top()


class Trainer:

    def __init__(self,
                 experiment_name='original-model',
                 is_small=True,
                 seed=42):

        # trainer params
        self.is_small = is_small
        self.seed = seed

        # model
        self.model = None
        self._get_compiled_model()

        # callbacks
        self.log_dir = pathlib.Path('logs') / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.callbacks = [
            tf.keras.callbacks.EarlyStopping(**params.EARLY_STOP_PARAMS),
            tf.keras.callbacks.TensorBoard(log_dir=self.log_dir)
        ]

    def _get_compiled_model(self):
        tf.keras.backend.clear_session()
        tf.random.set_seed(self.seed)

        self.model = LeNet().model
        optimizer = tf.keras.optimizers.Adam()
        self.model.compile(loss=params.LOSS,
                           optimizer=optimizer,
                           metrics=params.METRICS)

    def train(self):

        x_train, y_train, x_val, y_val = MNIST.get_data(is_small=self.is_small)

        history = self.model.fit(x=x_train,
                                 y=y_train,
                                 validation_data=(x_val, y_val),
                                 batch_size=params.BATCH_SIZE,
                                 epochs=params.N_EPOCHS,
                                 callbacks=self.callbacks)
        return history

    def predict_image(self, image):
        image_batch = image[np.newaxis, ...]
        probs = np.squeeze(self.model.predict(image_batch))
        return np.argmax(probs)

    def predict_batch(self, image_batch):
        probs = self.model.predict(image_batch)
        return np.argmax(probs, axis=1)
