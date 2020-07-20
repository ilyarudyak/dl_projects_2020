import pickle
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from pathlib import Path
import matplotlib.pyplot as plt
from functools import partial

import tensorflow as tf

AUTO = tf.data.experimental.AUTOTUNE


class Params:

    LEARNING_RATE = 0.001
    BATCH_SIZE = 128
    EPOCHS = 5

    #parameters for dataset
    CLASSES_BITS = [b'daisy', b'dandelion', b'roses', b'sunflowers', b'tulips']
    CLASSES_STR = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    FILE_SPLITS = [13, 2, 1]
    MAIN_SHUFFLE_BUFFER = 3072
    TRAIN_SHUFFLE_BUFFER = 2048

    # parameters for the model
    IMAGE_SIZE = [192, 192]
    N_CHANNELS = 3
    WEIGHTS = 'imagenet'
    N_CLASSES = 5
    ACTIVATION = 'softmax'

    # parameters for the trainer
    LOSS = 'sparse_categorical_crossentropy'
    METRICS = ['accuracy']
    SPLITS = [2990, 460, 220]
    DATASET_SIZE = 3670

    SEED = 42


params = Params()


# about this file:
# - this file is a preparation for our chexnet training on TPU;
# - to train on TPU we need TFRecords files - otherwise training will be very slow;
#   and we need to read them into tf.data.Dataset (no way we may use keras generator);
# - this files is based on TPU colab:
# https://codelabs.developers.google.com/codelabs/keras-flowers-tpu/

# TODO add parallel readings from files
# TODO where do we shuffle data?
class FlowerUtils:

    @staticmethod
    def plot_sample_from_dataset(dataset, is_xception=True):
        plt.figure(figsize=(8, 8))
        subplot = 331
        for i, (image, label) in enumerate(dataset):
            plt.subplot(subplot)
            plt.axis('off')

            image = image.numpy()
            if is_xception: image = image / 2 + .5
            plt.imshow(image)

            label = params.CLASSES_STR[label.numpy()]
            plt.title(label, fontsize=12)

            subplot += 1
            if i == 8: break
        # plt.tight_layout()
        plt.subplots_adjust(wspace=0.1, hspace=0.1)

    @staticmethod
    def plot_sample_from_batch(batched_dataset):
        dataset = batched_dataset.take(1).unbatch()
        FlowerUtils.plot_sample_from_dataset(dataset)

    @staticmethod
    def save_history(history, trainer):
        filename = trainer.history_file
        with open(filename, 'wb') as f:
            pickle.dump(history.history, f)

    @staticmethod
    def load_history(trainer):
        filename = trainer.history_file
        with open(filename, "rb") as f:
            history = pickle.load(f)
        return history

    @staticmethod
    def plot_metric(metric, trainer):
        history = FlowerUtils.load_history(trainer)
        plt.plot(history[metric], label=metric)
        val_metric = f'val_{metric}'
        plt.plot(history[val_metric], label=val_metric)
        plt.legend()
        plt.title(metric)

    @staticmethod
    # TODO correct plotting of history with continued epochs
    # TODO 2 history and weight files
    def train_with_fine_tuning(metric='accuracy'):
        ft = FlowerTrainer()
        ft.train()
        FlowerUtils.plot_metric(metric=metric, trainer=ft)
        params.LEARNING_RATE /= 10
        params.EPOCHS = 20
        ft = FlowerTrainer(is_fine_tune=True)
        ft.train()
        FlowerUtils.plot_metric(metric=metric, trainer=ft)
        ft.evaluate()


class FlowerTFRecordReader:

    IMAGE = 'image'
    CLASS_NUM = 'class'
    FEATURES = {
        # tf.string = bytestring (not text string)
        IMAGE: tf.io.FixedLenFeature([], tf.string),
        # shape [] means scalar
        CLASS_NUM: tf.io.FixedLenFeature([], tf.int64),
    }

    @staticmethod
    def _get_image(parsed_example):
        # parsed_example is a dictionary {'image': tensor, 'class': tensor}
        image = parsed_example[FlowerTFRecordReader.IMAGE]
        # we decode byte string into tensor that contains numpy array
        # it seems it already has [192, 192, 3] shape
        # it has type uint-8
        image = tf.image.decode_jpeg(image, channels=params.N_CHANNELS)
        image = tf.reshape(image, [*params.IMAGE_SIZE, params.N_CHANNELS])

        return image

    @staticmethod
    def read_tfrecord(example):
        # example is tensor that contains byte string both for image and class
        parsed_example = tf.io.parse_single_example(example, FlowerTFRecordReader.FEATURES)
        image = FlowerTFRecordReader._get_image(parsed_example)
        class_num = parsed_example[FlowerTFRecordReader.CLASS_NUM]
        return image, class_num


# TODO check how google credentials work on kaggle
# TODO add more image sizes
class FlowerDataSet:
    """
    this class:
     - creates TFRecord files with specified size;
     - reads TFRecord files into train and val datasets;

     notes:
     - we use here tf.data.Dataset;
     - we split images into train and val datasets by splitting
     list of files (no build-in mechanism in tf to do this);
     - we use kaggle platform to process files; in the future we're going to use
     google cloud storage directly;
    """

    def __init__(self):
        self.GCS_OUT_PATTERN = 'gs://dl-projects-2020-bucket-1/flowers/tfrecords-jpeg-192x192/flowers'
        self.GCS_PATTERN = f'{self.GCS_OUT_PATTERN}*.tfrec'

        # main datasets
        self.dataset = None
        self._get_dataset()

        # train / val splits
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self._get_splits()

    # @staticmethod
    # def _read_tfrecord(example):
    #     features = {
    #         "image": tf.io.FixedLenFeature([], tf.string),  # tf.string = bytestring (not text string)
    #         "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means scalar
    #     }
    #     # decode the TFRecord
    #     example = tf.io.parse_single_example(example, features)
    #
    #     # FixedLenFeature fields are now ready to use: example['size']
    #     # VarLenFeature fields require additional sparse_to_dense decoding
    #
    #     image = tf.image.decode_jpeg(example['image'], channels=params.N_CHANNELS)
    #     image = tf.reshape(image, [*params.IMAGE_SIZE, params.N_CHANNELS])
    #
    #     # pre-process image to Xception
    #     image = tf.cast(image, tf.float32)
    #     image = tf.keras.applications.xception.preprocess_input(image)
    #
    #     class_num = example['class']
    #
    #     return image, class_num

    def _get_dataset(self):
        filenames = tf.io.gfile.glob(self.GCS_PATTERN)
        self.dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
        self.dataset = self.dataset.map(FlowerTFRecordReader.read_tfrecord, num_parallel_calls=AUTO)
        self.dataset = self.dataset.shuffle(params.MAIN_SHUFFLE_BUFFER)

    @staticmethod
    def _get_split(filenames, prep_fun=None, is_train=False):
        """
        order of operations:
        -> read
        -> pre-process (augmentation + normalization)
        -> cache (if small)
        -> repeat -> shuffle (if train)
        -> batch -> prefetch
        """
        dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)\
            .map(FlowerTFRecordReader.read_tfrecord)\
            .map(partial(FlowerDataSet._preprocess, is_train=is_train)) \
            .cache()  # this dataset is small enough to fit in memory

        if is_train:
            dataset = dataset.repeat().shuffle(params.TRAIN_SHUFFLE_BUFFER)

        dataset = dataset.batch(params.BATCH_SIZE).prefetch(AUTO)
        return dataset

    @staticmethod
    def _preprocess(image, class_num, is_train=False):

        #augment data if train dataset
        if is_train:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_saturation(image, 0, 2)

        image = tf.cast(image, tf.float32)
        image = tf.keras.applications.xception.preprocess_input(image)

        return image, class_num

    def _get_filenames(self):
        filenames = tf.io.gfile.glob(self.GCS_PATTERN)
        train_split, val_split, _ = params.FILE_SPLITS
        train_filenames = filenames[:train_split]
        val_filenames = filenames[train_split:train_split + val_split]
        test_filenames = filenames[train_split + val_split:]
        return train_filenames, val_filenames, test_filenames

    def _get_splits(self):

        train_filenames, val_filenames, test_filenames = self._get_filenames()

        self.train_ds = FlowerDataSet._get_split(train_filenames, is_train=True)
        self.val_ds = FlowerDataSet._get_split(val_filenames)
        self.test_ds = FlowerDataSet._get_split(test_filenames)


class FlowerNet:

    def __init__(self,
                 is_freeze=True,
                 freeze_to=None):

        tf.random.set_seed(seed=params.SEED)

        self.base_model = None
        self.model = None
        self._build_model()

        # freeze all layers or layers up to
        # specified layer (not including)
        if freeze_to is not None:
            assert is_freeze is False
            self.freeze_to = freeze_to
            self._freeze_to()

        if is_freeze:
            self._freeze()

    def _build_model(self):
        # TODO can we add dropout?
        # self.base_model = tf.keras.applications.densenet.DenseNet121(weights=params.WEIGHTS, include_top=False)
        self.base_model = tf.keras.applications.Xception(input_shape=[*params.IMAGE_SIZE, params.N_CHANNELS],
                                                         weights=params.WEIGHTS, include_top=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(self.base_model.output)
        output = tf.keras.layers.Dense(units=params.N_CLASSES, activation=params.ACTIVATION)(x)
        self.model = tf.keras.models.Model(inputs=self.base_model.input, outputs=output)

    def _freeze(self):
        for layer in self.base_model.layers:
            layer.trainable = False

    def _freeze_to(self):
        for layer in self.base_model.layers:
            if layer.name == self.freeze_to:
                break
            layer.trainable = False


class FlowerTrainer:

    def __init__(self,
                 is_fine_tune=False,
                 freeze_to=None,
                 ):

        tf.keras.backend.clear_session()

        # TPU strategy and batch size
        self.tpu = None
        self.strategy = None
        self._get_strategy()

        ##### directories and files
        self.working_dir = Path('.')
        self.weight_file = self.working_dir / 'weights.h5'
        self.history_file = self.working_dir / 'history.pickle'

        # datasets
        self.fds = FlowerDataSet()
        self.train_ds = self.fds.train_ds
        self.val_ds = self.fds.val_ds
        self.test_ds = self.fds.test_ds

        # create model in strategy scope
        # if we use freeze_to set is_fine_tune MANUALLY in constructor
        if freeze_to: assert is_fine_tune is True
        self.freeze_to = freeze_to

        if is_fine_tune: is_freeze, is_load_weights = False, True
        else: is_freeze, is_load_weights = True, False
        self.is_freeze = is_freeze
        self.is_load_weights = is_load_weights

        self.net = None
        self.model = None
        self.optimizer = None
        # noinspection PyUnresolvedReferences
        with self.strategy.scope():
            self._get_model()

        ##### callbacks
        self.callbacks = [
            tf.keras.callbacks.ModelCheckpoint(str(self.weight_file),
                                               save_weights_only=True,
                                               # this is from the paper:
                                               # "pick the model with the lowest validation loss"
                                               monitor='val_loss',
                                               save_best_only=True,
                                               verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',  # from the paper
                                                 factor=.1,  # from the paper
                                                 # TODO tune this parameter
                                                 patience=5,  # not quite clear what did they use ?
                                                 verbose=1
                                                 ),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                             # min delta is 0 by default - that's strange
                                             min_delta=1e-3,
                                             patience=10,
                                             verbose=1)
        ]

        # steps per epoch
        self.train_steps = 0
        self.val_steps = 0
        self.test_steps = 0
        self._get_steps()

    def _get_model(self):

        # net and model
        self.net = FlowerNet(is_freeze=self.is_freeze,
                             freeze_to=self.freeze_to)
        self.model = self.net.model
        if self.is_load_weights:
            self.model.load_weights(str(self.weight_file))

        # optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=params.LEARNING_RATE)

        # metrics and loss
        self.model.compile(loss=params.LOSS,
                           optimizer=self.optimizer,
                           metrics=params.METRICS)

    def _get_strategy(self, is_verbose=True):
        try:  # detect TPUs
            self.tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
            tf.config.experimental_connect_to_cluster(self.tpu)
            tf.tpu.experimental.initialize_tpu_system(self.tpu)
            self.strategy = tf.distribute.experimental.TPUStrategy(self.tpu)
        except ValueError:  # detect GPUs
            # self.strategy = tf.distribute.MirroredStrategy()  # for GPU or multi-GPU machines
            self.strategy = tf.distribute.get_strategy()  # default strategy that works on CPU and single GPU
            # strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy() # for clusters of multi-GPU machines

        if is_verbose:
            print(f'number of accelerators: {self.strategy.num_replicas_in_sync}')

    def _get_steps(self):
        # unpack some params
        dataset_size = params.DATASET_SIZE
        batch_size = params.BATCH_SIZE
        train_split, val_split, test_split = np.array(params.SPLITS) / params.DATASET_SIZE

        self.train_steps = int(train_split * dataset_size // batch_size)
        self.val_steps = int(val_split * dataset_size // batch_size)
        self.test_steps = int(test_split * dataset_size // batch_size)

    def train(self):
        # we use here fit() instead of deprecated fit_generator()
        history = self.model.fit(self.train_ds,
                                 steps_per_epoch=self.train_steps,
                                 validation_data=self.val_ds,
                                 validation_steps=self.val_steps,
                                 epochs=params.EPOCHS,
                                 callbacks=self.callbacks)
        FlowerUtils.save_history(history, self)
        return history

    def evaluate(self, is_verbose=False):
        pass


if __name__ == '__main__':
    print(Params.LEARNING_RATE)

