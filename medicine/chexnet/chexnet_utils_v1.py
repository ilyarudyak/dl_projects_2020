import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
from pathlib import Path
import matplotlib.pyplot as plt
from functools import partial

import tensorflow as tf
from sklearn.model_selection import train_test_split

AUTO = tf.data.experimental.AUTOTUNE


class Params:
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    EPOCHS = 5

    # parameters for dataset
    CLASSES = [
        'Atelectasis',
        'Cardiomegaly',
        'Consolidation',
        'Edema',
        'Effusion',
        'Emphysema',
        'Fibrosis',
        'Hernia',
        'Infiltration',
        'Mass',
        # original label 'No Finding'
        # we don't include it in our labels
        # 'No_Finding',
        'Nodule',
        'Pleural_Thickening',
        'Pneumonia',
        'Pneumothorax'
    ]
    NO_FINDING = 'No_Finding'
    PNEUMONIA = 'Pneumonia'
    NOT_PNEUMONIA = 'not ' + PNEUMONIA
    SHUFFLE_BUFFER = 2048
    READER_SHUFFLE_BUFFER_TRAIN = 3000
    READER_SHUFFLE_BUFFER_VAL = 1000

    # parameters for the model
    IMAGE_SIZE = [224, 224]
    CHANNELS = 3

    # parameters for the trainer
    TRAIN_SPLIT = .8
    VAL_SPLIT = .1
    TEST_SPLIT = .1

    SEED = 42


params = Params()


class ChexnetUtils:

    @staticmethod
    def plot_sample_from_dataset(dataset, is_prep=True, is_pneumonia=True):
        dataset = dataset.take(9)
        plt.figure(figsize=(8, 8))
        subplot = 331
        for i, (image, label) in enumerate(dataset):
            plt.subplot(subplot)
            plt.axis('off')

            image = image.numpy()
            if is_prep: image = image / 2 + .5
            plt.imshow(image)

            label = label.numpy()
            if is_pneumonia:
                label = ChexnetUtils.get_label_pneumonia(label)
            else:
                label = ChexnetUtils.get_label_full(label)
            plt.title(label, fontsize=12)

            subplot += 1
            if i == 8: break
        # plt.tight_layout()
        plt.subplots_adjust(wspace=0.1, hspace=0.1)

    @staticmethod
    def get_label_pneumonia(label):
        return params.PNEUMONIA if label else params.NOT_PNEUMONIA

    @staticmethod
    def get_label_full(labels_arr):
        labels = [params.CLASSES[i] for i, j in enumerate(labels_arr) if j]
        if len(labels) == 0:
            return params.NO_FINDING
        return '|'.join(labels)

    @staticmethod
    def plot_sample_from_batch(batched_dataset, is_prep=True, is_pneumonia=True):
        dataset = batched_dataset.take(1).unbatch()
        ChexnetUtils.plot_sample_from_dataset(dataset, is_prep=is_prep, is_pneumonia=is_pneumonia)

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
        history = ChexnetUtils.load_history(trainer)
        plt.plot(history[metric], label=metric)
        val_metric = f'val_{metric}'
        plt.plot(history[val_metric], label=val_metric)
        plt.legend()
        plt.title(metric)

    # @staticmethod
    # # TODO correct plotting of history with continued epochs
    # # TODO 2 history and weight files
    # def train_with_fine_tuning(metric='accuracy'):
    #     ft = FlowerTrainer()
    #     ft.train()
    #     FlowerUtils.plot_metric(metric=metric, trainer=ft)
    #     params.LEARNING_RATE /= 10
    #     params.EPOCHS = 20
    #     ft = FlowerTrainer(is_fine_tune=True)
    #     ft.train()
    #     FlowerUtils.plot_metric(metric=metric, trainer=ft)
    #     ft.evaluate()


class CheXNetDataset:
    """
    This class writes TFRecord files for NIH
    """

    def __init__(self, is_pneumonia=True):

        # directories and files
        self.DATA_DIR = Path('/kaggle/input/data')
        self.WORKING_DIR = Path('/kaggle/working')
        self.DF_LABELS_PATH = self.DATA_DIR / 'Data_Entry_2017.csv'
        self.DF_FILES_PATH = self.WORKING_DIR / 'df_files.csv'

        # dataframes columns
        self.FILE_PATH = 'file_path'
        self.PATIENT_ID = 'patient_id'
        self.SPLIT = 'split'
        self.LABELS = 'labels'
        self.LABELS_ARRAY = 'labels_array'
        self.LABEL_PNEUMONIA = 'label_pneumonia'
        self.PNEUMONIA = 'Pneumonia'

        # main dataframes and dictionaries
        self.df_original = None
        self.labels_dict = None
        self._get_labels_dict()
        self.df_files = None
        self._get_df_files()

        # dataframes for splits
        self.SPLIT_TRAIN = 'train'
        self.SPLIT_VAl = 'val'
        self.SPLIT_TEST = 'test'
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self._get_patient_splits()

        # datasets
        self.is_pneumonia = is_pneumonia
        self.ds_train = None
        self.ds_val = None
        self.ds_test = None
        self._get_datasets()

    def _get_labels_dict(self):
        print(f'===> getting label_dict ... ')
        self.df_original = pd.read_csv(self.DF_LABELS_PATH)
        # change 'No Finding' to 'No_Finding'
        self.df_original.loc[self.df_original['Finding Labels'] == 'No Finding', 'Finding Labels'] = 'No_Finding'
        # get mapping of image name to its labels
        self.labels_dict = pd.Series(self.df_original['Finding Labels'].values,
                                     index=self.df_original['Image Index']).to_dict()
        print(f'===> getting label_dict DONE ')

    def _get_df_files(self):

        print(f'===> getting df_files ... ')

        # get list of paths for all image files
        if self.DF_FILES_PATH.is_file():
            files_str = list(pd.read_csv('df_files.csv').file_path.values)
        else:
            files_str = [str(fp) for fp in self.DATA_DIR.glob('**/**/*.png')]

        # create df that contain all image files
        self.df_files = pd.DataFrame(list(files_str), columns=[self.FILE_PATH], dtype='string')

        # add column with patient_id
        def get_patient(file_path):
            return str(file_path).split('/')[-1][:-8]

        self.df_files[self.PATIENT_ID] = self.df_files[self.FILE_PATH].apply(get_patient).astype('string')

        # add column with labels from original df provided in dataset
        def get_label(file_path):
            return self.labels_dict[file_path.split('/')[-1]]

        self.df_files[self.LABELS] = self.df_files[self.FILE_PATH].apply(get_label).astype('string')

        # add labels as array for full 14 disease problem
        def get_labels_as_array(labels):
            labels = labels.split('|')
            labels_arr = np.zeros(len(params.CLASSES))
            for label in labels:
                if label != params.NO_FINDING:
                    index = params.CLASSES.index(label)
                    labels_arr[index] = 1
            return labels_arr

        self.df_files[self.LABELS_ARRAY] = self.df_files[self.LABELS].apply(get_labels_as_array)

        # add label for pneumonia
        self.df_files[self.LABEL_PNEUMONIA] = self.df_files[self.LABELS].str.contains(self.PNEUMONIA).astype(int)

        # safe list of files to disc
        if not self.DF_FILES_PATH.is_file():
            self.df_files.file_path.to_csv(self.DF_FILES_PATH, index=False)

        print(f'===> getting df_files DONE ')

    def _get_patient_splits(self):

        print(f'===> getting patient_splits ... ')

        # get unique patient ids
        patients = self.df_files[self.PATIENT_ID].unique()

        # split patients using sklearn
        patients_train, patients_val = train_test_split(list(patients),
                                                        train_size=params.TRAIN_SPLIT,
                                                        test_size=1 - params.TRAIN_SPLIT,
                                                        random_state=params.SEED)
        patients_val, patients_test = train_test_split(patients_val,
                                                       train_size=.5,
                                                       test_size=.5,
                                                       random_state=params.SEED)
        patients_train, patients_val, patients_test = set(patients_train), set(patients_val), set(patients_test)

        # add split to main dataframe
        def get_split(patient_id):
            if patient_id in patients_train:
                return self.SPLIT_TRAIN
            elif patient_id in patients_val:
                return self.SPLIT_VAl
            elif patient_id in patients_test:
                return self.SPLIT_TEST
            else:
                raise ValueError('wrong patient_id')

        self.df_files[self.SPLIT] = self.df_files[self.PATIENT_ID].apply(get_split).astype('string')

        # get split dataframes
        self.df_train = self.df_files[self.df_files[self.SPLIT] == self.SPLIT_TRAIN]
        self.df_val = self.df_files[self.df_files[self.SPLIT] == self.SPLIT_VAl]
        self.df_test = self.df_files[self.df_files[self.SPLIT] == self.SPLIT_TEST]

        print(f'===> getting patient_splits DONE ')

    def _get_dataset(self, df, is_train=False):

        def _prepocess(file_path, label, _is_train=False):
            image = _decode_image(file_path)
            if _is_train:
                image = tf.image.random_flip_left_right(image)
            image = tf.cast(image, tf.float32)
            image = tf.keras.applications.xception.preprocess_input(image)
            return image, label

        def _decode_image(file_path):
            image_bytes = tf.io.read_file(file_path)
            image = tf.image.decode_jpeg(image_bytes, channels=params.CHANNELS)
            image = tf.image.resize(image, params.IMAGE_SIZE)
            return image

        # load slices of dataframe into dataset
        if self.is_pneumonia:
            labels = df[self.LABEL_PNEUMONIA].values
        else:
            labels = np.stack(df[self.LABELS_ARRAY].values, axis=0)
        files = df[self.FILE_PATH].values
        list_ds = tf.data.Dataset.from_tensor_slices((files, labels))

        # preprocess dataset
        # dataset is small enough to fit in memory
        dataset = list_ds.map(partial(_prepocess, _is_train=is_train)).cache()

        if is_train:
            dataset = dataset.shuffle(params.SHUFFLE_BUFFER).repeat()

        dataset = dataset.batch(params.BATCH_SIZE).prefetch(AUTO)

        return dataset

    def _get_datasets(self):
        self.ds_train = self._get_dataset(df=self.df_train, is_train=True)
        self.ds_val = self._get_dataset(df=self.df_val)
        self.ds_test = self._get_dataset(df=self.df_test)


class TFRecordWriter:
    IMAGE = 'image'
    LABEL = 'label'
    FILE_PREFIX = 'chexnet'
    WORKING_DIR = Path('/kaggle/working/')

    @staticmethod
    def _to_tfrecord(img_bytes, label_arr):

        def _bytestring_feature(list_of_bytes):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytes))

        def _float_feature(list_of_floats):  # float32
            return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))

        feature = {
            # one image in the list
            TFRecordWriter.IMAGE: _bytestring_feature([img_bytes]),
            TFRecordWriter.LABEL: _float_feature(label_arr.tolist()),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    @staticmethod
    def write_tfrecords(ds, split='train'):

        print(f'===> writing TFRecords split:{split} ...')
        for shard, (image_batch, label_batch) in enumerate(ds):

            # batch size used as shard size here
            shard_size = image_batch.numpy().shape[0]

            # good practice to have the number of records in the filename
            image_size = params.IMAGE_SIZE[0]
            out_dir = TFRecordWriter.WORKING_DIR / f'tfrecords-jpeg-{image_size}x{image_size}'
            out_dir = out_dir / split
            out_dir.mkdir(parents=True, exist_ok=True)
            out_pattern = f'{str(out_dir)}/chexnet'
            filename = f'{out_pattern}-{shard:02d}-{shard_size}.tfrec'

            with tf.io.TFRecordWriter(filename) as out_file:
                for i in range(shard_size):
                    example = TFRecordWriter._to_tfrecord(image_batch.numpy()[i],
                                                          label_batch.numpy()[i])
                    out_file.write(example.SerializeToString())
                print(f'{filename}')
        print(f'===> writing TFRecords split:{split} DONE')


class CheXNetTFRecordWriter:

    def __init__(self):

        # dataframes columns
        self.FILE_PATH = 'file_path'
        self.PATIENT_ID = 'patient_id'
        self.SPLIT = 'split'
        self.LABELS = 'labels'
        self.LABELS_ARRAY = 'labels_array'
        self.LABEL_PNEUMONIA = 'label_pneumonia'
        self.PNEUMONIA = 'Pneumonia'

        # dataframes
        self.df = None
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self._get_dataframes()

        # directories and files
        self.WORKING_DIR = Path('/kaggle/working')

        # params for dataset
        self.N_IMAGES = 112120
        self.SHARD_SIZE_TRAIN = 3000
        self.SHARD_SIZE_VAL = 1000

        # datasets to write
        self.SPLIT_TRAIN = 'train'
        self.SPLIT_VAL = 'val'
        self.SPLIT_TEST = 'test'
        self.ds_train = None
        self.ds_val = None
        self.ds_test = None
        self._get_datasets_to_write()

    def _get_dataframes(self):
        cds = CheXNetDataset(is_pneumonia=False)
        self.df = cds.df_files
        self.df_train = cds.df_train
        self.df_val = cds.df_val
        self.df_test = cds.df_test

    def _get_dataset_to_write(self, df, split='train'):
        """
        We prepare dataset to write into TFRecord:
        - decode images;
        - resize them to given size (key operation);
        - encode again into byte string;
        No pre-processing, it will be performed after reading
        a TFRecord file.
        """

        def _decode_image(file_name):
            bits = tf.io.read_file(file_name)
            image = tf.image.decode_jpeg(bits, channels=params.CHANNELS)
            image = tf.image.resize(image, params.IMAGE_SIZE)
            return image

        def _encode_image(image):
            image = tf.cast(image, tf.uint8)
            image = tf.image.encode_jpeg(image, optimize_size=True, chroma_downsampling=False)
            return image

        def _process(filename, label, split=self.SPLIT_TRAIN):
            image = _decode_image(filename)
            image = _encode_image(image)
            label = tf.cast(label, dtype=tf.float32)
            return image, label

        files = df[self.FILE_PATH].values
        # we have to stack a sequence of arrays to get 2D array
        labels = np.stack(df[self.LABELS_ARRAY].values, axis=0)
        # we need to supply a tuple to get supervised dataset
        if split == self.SPLIT_TRAIN: shard_size = self.SHARD_SIZE_TRAIN
        else: shard_size = self.SHARD_SIZE_VAL
        ds = tf.data.Dataset.from_tensor_slices((files, labels)) \
            .map(_process, num_parallel_calls=AUTO)\
            .batch(shard_size)
        return ds

    def _get_datasets_to_write(self):
        self.ds_train = self._get_dataset_to_write(self.df_train, split=self.SPLIT_TRAIN)
        self.ds_val = self._get_dataset_to_write(self.df_val, split=self.SPLIT_VAL)
        self.ds_test = self._get_dataset_to_write(self.df_test, split=self.SPLIT_TEST)

    def write_tfrecord_datasets(self):
        TFRecordWriter.write_tfrecords(self.ds_train.take(2), split=self.SPLIT_TRAIN)
        TFRecordWriter.write_tfrecords(self.ds_val.take(2), split=self.SPLIT_VAL)
        TFRecordWriter.write_tfrecords(self.ds_test.take(2), split=self.SPLIT_TEST)


class TFRecordReader:

    IMAGE = 'image'
    LABEL = 'label'
    FILE_PREFIX = 'chexnet'
    WORKING_DIR = Path('/kaggle/working/')
    FEATURES = {
        # tf.string = bytestring (not text string)
        IMAGE: tf.io.FixedLenFeature([], tf.string),
        # a certain number of floats
        LABEL: tf.io.VarLenFeature(tf.float32)
    }

    @staticmethod
    def read_tfrecord(example):
        parsed_example = tf.io.parse_single_example(example, features=TFRecordReader.FEATURES)
        image = parsed_example[TFRecordReader.IMAGE]
        image = tf.image.decode_jpeg(image, channels=params.CHANNELS)
        image = tf.reshape(image, [*params.IMAGE_SIZE, params.CHANNELS])
        label = parsed_example[TFRecordReader.LABEL]
        label = tf.sparse.to_dense(label)
        return image, label


class CheXNetTFRecordReader:


    def __init__(self):

        # patterns on GCS
        # self.GCS_OUT_PATTERN = 'gs://dl-projects-2020-bucket-1/flowers/tfrecords-jpeg-192x192/flowers'
        size = params.IMAGE_SIZE[0]
        self.GCS_DATA_DIR = Path(f'/kaggle/working/tfrecords-jpeg-{size}x{size}')
        self.SPLIT_TRAIN = 'train'
        self.SPLIT_VAl = 'val'
        self.SPLIT_TEST = 'test'

        # datasets for all splits
        self.ds_train = None
        self.ds_val = None
        self.ds_test = None
        self._get_datasets()

    def _get_pattern(self, split):
        split_dir = self.GCS_DATA_DIR / split
        return f'{str(split_dir)}/{TFRecordWriter.FILE_PREFIX}*.tfrec'

    def _read_dataset(self, split):
        """
        This function performs the following operations:
        - list all TFRecord files based on a pattern for a given split;
        - create a TFRecordDataset dataset;
        - read files using TFRecordReader;
        - shuffle files using a buffer based on a given split;
        """
        split_pattern = self._get_pattern(split)
        filenames = tf.io.gfile.glob(split_pattern)
        ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
        ds = ds.map(TFRecordReader.read_tfrecord, num_parallel_calls=AUTO)
        if split == self.SPLIT_TRAIN: shuffle_buffer = params.READER_SHUFFLE_BUFFER_TRAIN
        else: shuffle_buffer = params.READER_SHUFFLE_BUFFER_VAL
        ds = ds.shuffle(shuffle_buffer)
        return ds

    @staticmethod
    def _process(image, label, split):
        #augment data if train dataset
        if split == 'train': image = tf.image.random_flip_left_right(image)
        image = tf.cast(image, tf.float32)
        image = tf.keras.applications.xception.preprocess_input(image)
        return image, label

    def _process_dataset(self, split):
        """
        order of operations:
        -> read
        -> pre-process (augmentation + normalization)
        -> cache (if small)
        -> repeat -> shuffle (if train)
        -> batch -> prefetch
        """
        ds = self._read_dataset(split)\
            .map(partial(CheXNetTFRecordReader._process, split=split)) \
            .cache()

        if split == self.SPLIT_TRAIN:
            ds = ds.repeat().shuffle(params.SHUFFLE_BUFFER)

        ds = ds.batch(params.BATCH_SIZE).prefetch(AUTO)

        return ds

    def _get_datasets(self):
        self.ds_train = self._process_dataset(split=self.SPLIT_TRAIN)
        self.ds_val = self._process_dataset(split=self.SPLIT_VAl)
        self.ds_test = self._process_dataset(split=self.SPLIT_TEST)

