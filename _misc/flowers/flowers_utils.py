import json
import pickle
import math
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from pathlib import Path
import matplotlib.pyplot as plt

import tensorflow as tf

AUTO = tf.data.experimental.AUTOTUNE


class Params:

    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    EPOCHS = 5

    #parameters for dataset
    FILE_SPLITS = [13, 2, 1]
    TRAIN_SHUFFLE_BUFFER = 2048

    # parameters for the model
    IMAGE_SIZE = [192, 192]
    N_CHANNELS = 3
    WEIGHTS = 'imagenet'
    N_CLASSES = 5
    ACTIVATION = 'softmax'

    # parameters for the trainer
    LOSS = 'categorical_crossentropy'
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


class FlowerUtils:

    @staticmethod
    def display_9_images_from_dataset(dataset, is_class_num=False, fds=None):
        plt.figure(figsize=(8, 8))
        subplot = 331
        for i, (image, label) in enumerate(dataset):
            plt.subplot(subplot)
            plt.axis('off')
            plt.imshow(image.numpy().astype(np.uint8))

            if is_class_num:
                label = fds.CLASSES[label.numpy()]
            else:
                label = label.numpy()

            plt.title(label.decode("utf-8"), fontsize=12)
            subplot += 1
            if i == 8:
                break
        # plt.tight_layout()
        plt.subplots_adjust(wspace=0.1, hspace=0.1)

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


class FlowersDataSet:
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

    def __init__(self, with_splits=True):
        self.GCS_IN_PATTERN = 'gs://flowers-public/*/*.jpg'

        # self.GCS_OUT_DIR = Path('tfrecords-jpeg-192x192')
        # self.GCS_OUT_PATTERN = 'tfrecords-jpeg-192x192/flowers'
        self.GCS_OUT_PATTERN = 'gs://dl-projects-2020-bucket-1/flowers/tfrecords-jpeg-192x192/flowers'
        self.GCS_PATTERN = f'{self.GCS_OUT_PATTERN}*.tfrec'

        self.SHARDS = 16
        self.TARGET_SIZE = params.IMAGE_SIZE
        self.CLASSES = [b'daisy', b'dandelion', b'roses', b'sunflowers', b'tulips']
        self.N_IMAGES = len(tf.io.gfile.glob(self.GCS_IN_PATTERN))
        self.SHARD_SIZE = math.ceil(1.0 * self.N_IMAGES / self.SHARDS)  # 230
        self.SHUFFLE_BUFFER = 300

        # main datasets
        self.dataset_from_files = None
        self.dataset_from_tfr = None

        # train / val splits
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        if with_splits:
            self._get_splits()

    def _get_dataset_to_write(self):

        def _resize_and_crop_image(image):
            # Resize and crop using "fill" algorithm:
            # always make sure the resulting image
            # is cut out from the source image so that
            # it fills the TARGET_SIZE entirely with no
            # black bars and a preserved aspect ratio.
            w = tf.shape(image)[0]
            h = tf.shape(image)[1]
            tw = self.TARGET_SIZE[1]
            th = self.TARGET_SIZE[0]
            resize_crit = (w * th) / (h * tw)
            image = tf.cond(resize_crit < 1,
                            lambda: tf.image.resize(image, [w * tw / w, h * tw / w]),  # if true
                            lambda: tf.image.resize(image, [w * th / h, h * th / h])  # if false
                            )
            nw = tf.shape(image)[0]
            nh = tf.shape(image)[1]
            image = tf.image.crop_to_bounding_box(image, (nw - tw) // 2, (nh - th) // 2, tw, th)
            return image

        def _recompress_image(image):
            image = tf.cast(image, tf.uint8)
            image = tf.image.encode_jpeg(image, optimize_size=True, chroma_downsampling=False)
            return image

        def _get_image_label_str(filename):
            # read file and decode image
            bits = tf.io.read_file(filename)
            image = tf.image.decode_jpeg(bits)

            # resize image
            image = _resize_and_crop_image(image)

            # decompress image
            image = _recompress_image(image)

            # parse flower name from containing directory
            label = tf.strings.split(tf.expand_dims(filename, axis=-1), sep='/')
            label_str = label.values[-2]

            return image, label_str

        filenames = tf.data.Dataset.list_files(self.GCS_IN_PATTERN, seed=params.SEED)
        self.dataset_from_files = filenames.map(_get_image_label_str, num_parallel_calls=AUTO)
        self.dataset_from_files = self.dataset_from_files.batch(self.SHARD_SIZE)

    def _to_tfrecord(self, img_bytes, label):

        def _bytestring_feature(list_of_bytestrings):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))

        def _int_feature(list_of_ints):  # int64
            return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))

        class_num = np.argmax(np.array(self.CLASSES) == label)  # 'roses' => 2 (order defined in CLASSES)

        feature = {
            "image": _bytestring_feature([img_bytes]),  # one image in the list
            "class": _int_feature([class_num]),  # one class in the list
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def write_tfrecord(self):

        if self.dataset_from_files is None:
            self._get_dataset_to_write()

        # if parents is true, any missing parents of this path are created as needed.
        # if exist_ok is true, FileExistsError exceptions will be ignored.
        # self.GCS_OUT_DIR.mkdir(parents=True, exist_ok=True)

        print("===> writing TFRecords ...")
        for shard, (image, label) in enumerate(self.dataset_from_files):

            # batch size used as shard size here
            shard_size = image.numpy().shape[0]

            # good practice to have the number of records in the filename
            filename = f'{self.GCS_OUT_PATTERN}{shard:02d}-{shard_size}.tfrec'

            with tf.io.TFRecordWriter(filename) as out_file:
                for i in range(shard_size):
                    example = self._to_tfrecord(image.numpy()[i],  # re-compressed image: already a byte string
                                                label.numpy()[i])
                    out_file.write(example.SerializeToString())
                print(f'{filename} with {shard_size} records DONE')

    @staticmethod
    def _read_tfrecord(example):
        features = {
            "image": tf.io.FixedLenFeature([], tf.string),  # tf.string = bytestring (not text string)
            "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means scalar
        }
        # decode the TFRecord
        example = tf.io.parse_single_example(example, features)

        # FixedLenFeature fields are now ready to use: example['size']
        # VarLenFeature fields require additional sparse_to_dense decoding

        image = tf.image.decode_jpeg(example['image'], channels=params.N_CHANNELS)
        image = tf.reshape(image, [*params.IMAGE_SIZE, params.N_CHANNELS])

        class_num = example['class']

        return image, class_num

    def read_tfrecord(self):
        filenames = tf.io.gfile.glob(self.GCS_PATTERN)
        self.dataset_from_tfr = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
        self.dataset_from_tfr = self.dataset_from_tfr.map(FlowersDataSet._read_tfrecord, num_parallel_calls=AUTO)
        self.dataset_from_tfr = self.dataset_from_tfr.shuffle(self.SHUFFLE_BUFFER)

    def _get_splits(self):

        if self.dataset_from_tfr is None:
            self.read_tfrecord()

        # get filenames
        filenames = tf.io.gfile.glob(self.GCS_PATTERN)
        train_split, val_split, _ = params.FILE_SPLITS
        train_filenames = filenames[:train_split]
        val_filenames = filenames[train_split:train_split + val_split]
        test_filenames = filenames[train_split + val_split]

        # get train dataset
        def _data_augment(image, class_num):
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_saturation(image, 0, 2)
            return image, class_num

        self.train_ds = tf.data.TFRecordDataset(train_filenames, num_parallel_reads=AUTO)\
            .map(FlowersDataSet._read_tfrecord).map(_data_augment)\
            .repeat().shuffle(params.TRAIN_SHUFFLE_BUFFER).batch(params.BATCH_SIZE).prefetch(AUTO)

        # get val dataset
        self.val_ds = tf.data.TFRecordDataset(val_filenames, num_parallel_reads=AUTO)\
            .map(FlowersDataSet._read_tfrecord)\
            .batch(params.BATCH_SIZE).prefetch(AUTO)

        # get test dataset
        self.test_ds = tf.data.TFRecordDataset(test_filenames, num_parallel_reads=AUTO)\
            .map(FlowersDataSet._read_tfrecord)\
            .batch(params.BATCH_SIZE).prefetch(AUTO)


class FlowersDataSetWithIntermediateDS:
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
        self.GCS_PATTERN = 'gs://flowers-public/*/*.jpg'
        self.GCS_OUTPUT = 'tfrecords-jpeg-192x192/flowers'
        self.SHARDS = 16
        self.TARGET_SIZE = [192, 192]
        self.CLASSES = [b'daisy', b'dandelion', b'roses', b'sunflowers', b'tulips']
        self.N_IMAGES = len(tf.io.gfile.glob(self.GCS_PATTERN))
        self.SHARD_SIZE = math.ceil(1.0 * self.N_IMAGES / self.SHARDS)

        # datasets
        self.raw_dataset = None
        self.resized_dataset = None
        self.recomp_dataset = None

    def get_raw_dataset(self):

        def _decode_jpeg_and_label(filename):
            bits = tf.io.read_file(filename)
            image = tf.image.decode_jpeg(bits)
            # parse flower name from containing directory
            label = tf.strings.split(tf.expand_dims(filename, axis=-1), sep='/')
            label = label.values[-2]
            return image, label

        filenames = tf.data.Dataset.list_files(self.GCS_PATTERN, seed=params.SEED)
        self.raw_dataset = filenames.map(_decode_jpeg_and_label, num_parallel_calls=AUTO)

    def get_resized_dataset(self):

        def resize_and_crop_image(image, label):
            # Resize and crop using "fill" algorithm:
            # always make sure the resulting image
            # is cut out from the source image so that
            # it fills the TARGET_SIZE entirely with no
            # black bars and a preserved aspect ratio.
            w = tf.shape(image)[0]
            h = tf.shape(image)[1]
            tw = self.TARGET_SIZE[1]
            th = self.TARGET_SIZE[0]
            resize_crit = (w * th) / (h * tw)
            image = tf.cond(resize_crit < 1,
                            lambda: tf.image.resize(image, [w * tw / w, h * tw / w]),  # if true
                            lambda: tf.image.resize(image, [w * th / h, h * th / h])  # if false
                            )
            nw = tf.shape(image)[0]
            nh = tf.shape(image)[1]
            image = tf.image.crop_to_bounding_box(image, (nw - tw) // 2, (nh - th) // 2, tw, th)
            return image, label

        if self.raw_dataset is None:
            self.get_raw_dataset()

        self.resized_dataset = self.raw_dataset.map(resize_and_crop_image, num_parallel_calls=AUTO)

    def get_recomp_dataset(self):

        def recompress_image(image, label):
            image = tf.cast(image, tf.uint8)
            image = tf.image.encode_jpeg(image, optimize_size=True, chroma_downsampling=False)
            return image, label

        if self.resized_dataset is None:
            self.get_resized_dataset()

        self.recomp_dataset = self.resized_dataset.map(recompress_image, num_parallel_calls=AUTO)
        self.recomp_dataset = self.recomp_dataset.batch(self.SHARD_SIZE)

    def _to_tfrecord(self, img_bytes, label):

        def _bytestring_feature(list_of_bytestrings):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))

        def _int_feature(list_of_ints):  # int64
            return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))

        class_num = np.argmax(np.array(self.CLASSES) == label)  # 'roses' => 2 (order defined in CLASSES)

        feature = {
            "image": _bytestring_feature([img_bytes]),  # one image in the list
            "class": _int_feature([class_num]),  # one class in the list
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def write_tfrecord(self):

        if self.recomp_dataset is None:
            self.get_recomp_dataset()

        print("===> writing TFRecords ...")
        for shard, (image, label) in enumerate(self.recomp_dataset.take(1)):

            # batch size used as shard size here
            shard_size = image.numpy().shape[0]

            # good practice to have the number of records in the filename
            filename = f'{self.GCS_OUTPUT}{shard:02d}-{shard_size}.tfrec'

            with tf.io.TFRecordWriter(filename) as out_file:
                for i in range(shard_size):
                    example = self._to_tfrecord(image.numpy()[i],  # re-compressed image: already a byte string
                                                label.numpy()[i])
                    out_file.write(example.SerializeToString())
                print(f'{filename} with {shard_size} records DONE')


class FlowersNet:

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
            is_freeze = False
            self.freeze_to = freeze_to
            self._freeze_to()

        if is_freeze:
            self._freeze()

    def _build_model(self):
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


class FlowersTrainer:

    def __init__(self,
                 is_freeze=True,
                 freeze_to=None,
                 is_load_weights=False,
                 is_local=True
                 ):

        tf.keras.backend.clear_session()

        # TPU strategy
        self.tpu = None
        self.strategy = None
        self._get_strategy()

        ##### directories and files
        #TODO remove local
        if is_local:
            self.working_dir = Path('.')
        else:
            self.working_dir = Path('/kaggle/working')
        self.weight_file = self.working_dir / 'weights.h5'
        self.history_file = self.working_dir / 'history.pickle'

        # datasets
        self.fds = FlowersDataSet()
        self.train_ds = self.fds.train_ds
        self.val_ds = self.fds.val_ds
        self.test_ds = self.fds.test_ds

        # create model in strategy scope
        self.is_freeze = is_freeze
        self.freeze_to = freeze_to
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
        self.net = FlowersNet(is_freeze=self.is_freeze,
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
                                 batch_size=params.BATCH_SIZE,
                                 steps_per_epoch=self.train_steps,
                                 validation_data=self.val_ds,
                                 validation_steps=self.val_steps,
                                 epochs=params.EPOCHS,
                                 validation_batch_size=params.BATCH_SIZE,
                                 callbacks=self.callbacks)
        FlowerUtils.save_history(history, self)
        return history

    def evaluate(self, is_verbose=False):
        pass


if __name__ == '__main__':
    print(Params.LEARNING_RATE)

