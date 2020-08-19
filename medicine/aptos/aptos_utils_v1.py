import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
from pathlib import Path
import matplotlib.pyplot as plt
from functools import partial
from google.cloud import storage
import seaborn as sns

import tensorflow as tf
from sklearn.model_selection import train_test_split

AUTO = tf.data.experimental.AUTOTUNE
sns.set()


#################################################################
### version description:
### - v.1: simple model;
### - v.2:
### - v.3:
### - v.4:
#################################################################


class Params:

    def __init__(self):
        # hyper-parameters
        self.LEARNING_RATE = 0.001
        self.BATCH_SIZE = 32
        self.EPOCHS = 5
        self.IMAGE_SIZE = [224, 224]

        # train / val split
        self.VAL_SIZE = 0.2
        self.PREP_FUN = tf.keras.applications.resnet.preprocess_input

        # classes
        self.CLASSES = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferative_DR']

        # seed
        self.SEED = 42


params = Params()


class AptosUtils:

    @staticmethod
    def plot_sample_from_gen(gen, is_prep=True):
        image_batch, label_batch = next(gen)
        plt.figure(figsize=(8, 8))
        subplot = 331
        for image, label in zip(image_batch[:9], label_batch[:9]):
            plt.subplot(subplot)
            plt.axis('off')

            if is_prep: image = image / 2 + .5
            plt.imshow(image)
            plt.title(params.CLASSES[int(label)], fontsize=12)

            subplot += 1
        plt.subplots_adjust(wspace=0.1, hspace=0.1)


class AptosDataset:

    def __init__(self):

        # directories and files
        self.DATA_DIR = Path('/kaggle/input/aptos2019-blindness-detection/')
        self.IMAGES_TRAIN = self.DATA_DIR / 'train_images'
        self.IMAGES_TEST = self.DATA_DIR / 'test_images'
        self.CSV_TRAIN = self.DATA_DIR / 'train.csv'
        self.CSV_TEST = self.DATA_DIR / 'test.csv'
        self.WORKING_DIR = Path('/kaggle/working')

        # splits
        self.TRAIN = 'train'
        self.VAL = 'val'
        self.TEST = 'test'
        self.SPLITS = ['train', 'val', 'test']

        # dataframes
        self.dfs = {self.TRAIN: None,
                    self.VAL: None,
                    self.TEST: None}
        self.FILE_PATH = 'file_path'
        self.ID_CODE = 'id_code'
        self.DIAGNOSIS = 'diagnosis'
        self._get_dfs()

        # data generators
        self.gens = {self.TRAIN: None,
                     self.VAL: None,
                     self.TEST: None}
        self.RESCALE = 1 / 255.
        self._get_gens()

    def _get_dfs(self):
        self.dfs[self.TRAIN], self.dfs[self.VAL] = train_test_split(pd.read_csv(self.CSV_TRAIN),
                                                                    test_size=params.VAL_SIZE,
                                                                    random_state=params.SEED)
        self.dfs[self.TEST] = pd.read_csv(self.CSV_TEST)
        self._add_file_path()

    def _add_file_path(self):

        def get_file_path(id_code):
            return f'{id_code}.png'

        for split in self.SPLITS:
            df = self.dfs[split]
            df[self.FILE_PATH] = df[self.ID_CODE].apply(get_file_path)

    def _get_gens(self):
        idg_train = tf.keras.preprocessing.image.ImageDataGenerator(rescale=self.RESCALE,
                                                                    preprocessing_function=params.PREP_FUN)
        idg_val = tf.keras.preprocessing.image.ImageDataGenerator(rescale=self.RESCALE,
                                                                  preprocessing_function=params.PREP_FUN)
        idgs = {self.TRAIN: idg_train,
                self.VAL: idg_val,
                self.TEST: idg_val}

        def get_gen(_split):

            idg_params = {
                'dataframe': self.dfs[_split],
                'directory': self.IMAGES_TRAIN,
                'x_col': self.FILE_PATH,
                'y_col': self.DIAGNOSIS,
                'target_size': params.IMAGE_SIZE,
                'batch_size': params.BATCH_SIZE,
                'class_mode': 'raw',
                'seed': params.SEED
            }

            if _split == self.TEST:
                idg_params['directory'] = self.IMAGES_TEST
                idg_params['y_col'] = None
                idg_params['class_mode'] = None

            return idgs[_split].flow_from_dataframe(**idg_params)

        for split in self.SPLITS:
            self.gens[split] = get_gen(split)
