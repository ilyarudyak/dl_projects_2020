import json
import pickle
import math
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from pathlib import Path
import matplotlib.pyplot as plt
from argparse import Namespace

import tensorflow as tf

AUTO = tf.data.experimental.AUTOTUNE

params = Namespace(
    LEARNING_RATE=0.001,
    BATCH_SIZE=16,
    EPOCHS=5,

    SEED=42
)


# about this file:
# - this file is a preparation for our chexnet training on TPU;
# - to train on TPU we need TFRecords files - otherwise training will be very slow;
#   and we need to read them into tf.data.Dataset (no way we may use keras generator);
# - this files is based on TPU colab:
# https://codelabs.developers.google.com/codelabs/keras-flowers-tpu/


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
    plt.show()


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

    def __init__(self):
        self.GCS_IN_PATTERN = 'gs://flowers-public/*/*.jpg'
        self.GCS_OUT_DIR = Path('tfrecords-jpeg-192x192')
        self.GCS_OUT_PATTERN = 'tfrecords-jpeg-192x192/flowers'
        self.SHARDS = 16
        self.TARGET_SIZE = [192, 192]
        self.CLASSES = [b'daisy', b'dandelion', b'roses', b'sunflowers', b'tulips']
        self.N_IMAGES = len(tf.io.gfile.glob(self.GCS_IN_PATTERN))
        self.SHARD_SIZE = math.ceil(1.0 * self.N_IMAGES / self.SHARDS)  # 230
        self.SHUFFLE_BUFFER = 300

        # datasets
        self.dataset_from_files = None
        self.dataset_from_tfr = None

    def _get_dataset(self):

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
            self._get_dataset()

        # if parents is true, any missing parents of this path are created as needed.
        # if exist_ok is true, FileExistsError exceptions will be ignored.
        self.GCS_OUT_DIR.mkdir(parents=True, exist_ok=True)

        print("===> writing TFRecords ...")
        for shard, (image, label) in enumerate(self.dataset_from_files.take(1)):

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

    def _read_tfrecord(self, example):
        features = {
            "image": tf.io.FixedLenFeature([], tf.string),  # tf.string = bytestring (not text string)
            "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means scalar
        }
        # decode the TFRecord
        example = tf.io.parse_single_example(example, features)

        # FixedLenFeature fields are now ready to use: example['size']
        # VarLenFeature fields require additional sparse_to_dense decoding

        image = tf.image.decode_jpeg(example['image'], channels=3)
        image = tf.reshape(image, [*self.TARGET_SIZE, 3])

        class_num = example['class']

        return image, class_num

    def read_tfrecord(self):
        filenames = tf.io.gfile.glob(self.GCS_OUT_PATTERN + "*.tfrec")
        self.dataset_from_tfr = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
        self.dataset_from_tfr = self.dataset_from_tfr.map(self._read_tfrecord, num_parallel_calls=AUTO)
        self.dataset_from_tfr = self.dataset_from_tfr.shuffle(self.SHUFFLE_BUFFER)


class FlowersDataSet2:
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
