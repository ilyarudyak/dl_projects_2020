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
    BATCH_SIZE = 128
    EPOCHS = 5

    #parameters for dataset
    CLASSES_STR = [
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
        'No_Finding',  # original label 'No Finding'
        'Nodule',
        'Pleural_Thickening',
        'Pneumonia',
        'Pneumothorax'
    ]

    # parameters for the model
    IMAGE_SIZE = [224, 224]

    # parameters for the trainer
    TRAIN_SPLIT = .8
    VAL_SPLIT = .1
    TEST_SPLIT = .1

    SEED = 42


params = Params()


class CheXNetDataset:
    """
    This class writes TFRecord files for NIH
    """

    def __init__(self):

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

        # main dataframes and dictionaries
        self.df_labels = None
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
        self.ds_train = None
        self.ds_val = None
        self.ds_test = None

    def _get_labels_dict(self):
        print(f'===> getting label_dict ... ')
        self.df_labels = pd.read_csv(self.DF_LABELS_PATH)
        # change 'No Finding' to 'No_Finding'
        self.df_labels .loc[self.df_labels['Finding Labels'] == 'No Finding', 'Finding Labels'] = 'No_Finding'
        # get mapping of image name to its labels
        self.labels_dict = pd.Series(self.df_labels['Finding Labels'].values,
                                     index=self.df_labels['Image Index']).to_dict()
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
        def get_patient(file_path): return str(file_path).split('/')[-1][:-8]
        self.df_files[self.PATIENT_ID] = self.df_files[self.FILE_PATH].apply(get_patient).astype('string')

        # add column with labels from original df provided in dataset
        def get_label(file_path): return self.labels_dict[file_path.split('/')[-1]]
        self.df_files['labels'] = self.df_files['file_path'].apply(get_label).astype('string')

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
                                                        test_size=1-params.TRAIN_SPLIT,
                                                        random_state=params.SEED)
        patients_val, patients_test = train_test_split(patients_val,
                                                       train_size=.5,
                                                       test_size=.5,
                                                       random_state=params.SEED)
        patients_train, patients_val, patients_test = set(patients_train), set(patients_val), set(patients_test)

        # add split to main dataframe
        def get_split(patient_id):
            if patient_id in patients_train: return self.SPLIT_TRAIN
            elif patient_id in patients_val: return self.SPLIT_VAl
            elif patient_id in patients_test: return self.SPLIT_TEST
            else: raise ValueError('wrong patient_id')

        self.df_files[self.SPLIT] = self.df_files[self.PATIENT_ID].apply(get_split).astype('string')

        # get split dataframes
        self.df_train = self.df_files[self.df_files[self.SPLIT] == self.SPLIT_TRAIN]
        self.df_val = self.df_files[self.df_files[self.SPLIT] == self.SPLIT_VAl]
        self.df_test = self.df_files[self.df_files[self.SPLIT] == self.SPLIT_TEST]

        print(f'===> getting patient_splits DONE ')

    def _get_datasets(self):
        pass
