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


class FlowersDataSetWithWriter:
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
        self.CLASSES = params.CLASSES_BITS
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
        self.dataset_from_tfr = self.dataset_from_tfr.map(FlowersDataSetWithWriter._read_tfrecord, num_parallel_calls=AUTO)
        self.dataset_from_tfr = self.dataset_from_tfr.shuffle(self.SHUFFLE_BUFFER)

    def _get_splits(self):

        if self.dataset_from_tfr is None:
            self.read_tfrecord()

        # get filenames
        filenames = tf.io.gfile.glob(self.GCS_PATTERN)
        train_split, val_split, _ = params.FILE_SPLITS
        train_filenames = filenames[:train_split]
        val_filenames = filenames[train_split:train_split + val_split]
        test_filenames = filenames[train_split + val_split:]

        # get train dataset
        def _data_augment(image, class_num):
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_saturation(image, 0, 2)
            return image, class_num

        self.train_ds = tf.data.TFRecordDataset(train_filenames, num_parallel_reads=AUTO)\
            .map(FlowersDataSetWithWriter._read_tfrecord).map(_data_augment)\
            .repeat().shuffle(params.TRAIN_SHUFFLE_BUFFER).batch(params.BATCH_SIZE).prefetch(AUTO)

        # get val dataset
        self.val_ds = tf.data.TFRecordDataset(val_filenames, num_parallel_reads=AUTO)\
            .map(FlowersDataSetWithWriter._read_tfrecord)\
            .batch(params.BATCH_SIZE).prefetch(AUTO)

        # get test dataset
        self.test_ds = tf.data.TFRecordDataset(test_filenames, num_parallel_reads=AUTO)\
            .map(FlowersDataSetWithWriter._read_tfrecord)\
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



