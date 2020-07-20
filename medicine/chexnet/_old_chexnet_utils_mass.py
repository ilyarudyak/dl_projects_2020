import json
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

from pathlib import Path

import tensorflow as tf
from tensorflow.keras import backend as K

import matplotlib.pyplot as plt


#################### about ####################
# in this file we use custom loss function AND
# class_weights in model.fit

# we solve another binary classification problem here:
# instead of pneumonia we are looking into 2
# diseases: mass and nodule (combined) - that's
# 10% of dataset (the same as in the main dataset)
# instead of 1% for pneumonia

#################### utils ####################


class Params:
    """Class that loads hyper-parameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


def save_history(history, trainer):
    filename = trainer.history_file
    with open(filename, 'wb') as f:
        pickle.dump(history.history, f)


def load_history(trainer):
    filename = trainer.history_file
    with open(filename, "rb") as f:
        history = pickle.load(f)
    return history


def plot_metric(metric, trainer):
    history = load_history(trainer)
    plt.plot(history[metric], label=metric)
    val_metric = f'val_{metric}'
    plt.plot(history[val_metric], label=val_metric)
    plt.legend()
    plt.title(metric)


def check_for_leakage(df1, df2, patient_col):
    """
    Return True if there any patients are in both df1 and df2.

    Args:
        df1 (dataframe): dataframe describing first dataset
        df2 (dataframe): dataframe describing second dataset
        patient_col (str): string name of column with patient IDs

    Returns:
        leakage (bool): True if there is leakage, otherwise False
    """

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

    df1_patients_unique = set(df1[patient_col])
    df2_patients_unique = set(df2[patient_col])

    patients_in_both_groups = df1_patients_unique.intersection(df2_patients_unique)

    # leakage contains true if there is patient overlap, otherwise false.
    leakage = len(patients_in_both_groups) > 0  # boolean (true if there is at least 1 patient in both groups)

    ### END CODE HERE ###

    return leakage


def print_sample_batch(data_gen, m=3, n=3, shape=(224, 224), label='Pneum'):
    image_batch, labels_batch = next(data_gen)
    image_batch = image_batch[:m * n, :, :, 0].reshape(m, n, shape[0], shape[1])
    labels_batch = labels_batch[:m * n].reshape(m, n)
    fig, ax = plt.subplots(m, n)
    for i in range(m):
        for j in range(n):
            ax[i, j].imshow(image_batch[i, j].reshape(*shape), cmap='gray')
            ax[i, j].set_axis_off()
            ax[i, j].set_title(f'{label}: {bool(labels_batch[i, j])}')


def get_weighted_loss(pos_weight, neg_weight, epsilon=1e-7):
    def weighted_loss(y_true, y_pred):
        """
        compute weighted loss - we multiply log_likelihood by weight
        :param y_true: shape (batch_size,)
        :param y_pred: shape (batch_size, 2) so we have probabilities for each class
        :return:
        """
        y_true = tf.cast(y_true, tf.float32)
        log_likelihood = -pos_weight * y_true * K.log(y_pred[:, 1] + epsilon) \
                         - neg_weight * (1 - y_true) * K.log(y_pred[:, 0] + epsilon)
        loss = K.mean(log_likelihood)
        return loss

    return weighted_loss


def run_full_training_with_unfrozen_conv5():
    trainer = Trainer()
    trainer.train()
    trainer = Trainer(is_fine_tune=True,
                      freeze_to=trainer.CONV5_BLOCK)
    trainer.train()
    trainer.evaluate()


#################### *data* ####################


class CheXNetDataGen:

    def __init__(self,
                 params,
                 is_local=False,
                 is_toy=True,
                 is_pneumonia=False,
                 is_mass_or_nodule=True):

        self.params = params

        tf.random.set_seed(seed=self.params.seed)

        self.is_pneumonia = is_pneumonia

        # directories and csv file
        if is_toy:
            if is_local:
                self.DATA_DIR = Path('sample')
                self.CSV_PATH = self.DATA_DIR / 'sample_labels.csv'
                self.IMAGE_PATH = self.DATA_DIR / 'images'
            else:
                self.DATA_DIR = Path('/kaggle/input/sample')
                self.CSV_PATH = self.DATA_DIR / 'sample_labels.csv'
                self.IMAGE_PATH = self.DATA_DIR / 'sample/images'

        # labels
        self.PNEUMONIA = 'Pneumonia'
        self.MASS = 'Mass'
        self.NODULE = 'Nodule'
        self.MASS_OR_NODULE = 'Mass_or_Nodule'
        self.IMAGE_INDEX = 'Image_Index'
        self.FINDING_LABELS = 'Finding_Labels'
        self.PATIENT_ID = 'Patient_ID'
        if is_pneumonia:
            self.LABELS = [self.PNEUMONIA]
        elif is_mass_or_nodule:
            self.LABELS = [self.MASS, self.NODULE]
        else:
            self.LABELS = [
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

        self.df = pd.read_csv(self.CSV_PATH)
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self._get_dfs()

        self.gen_train = None
        self.gen_val = None
        self.gen_test = None
        self._get_datagens()

    def _clean_df(self):
        # change 'No Finding' to 'No_Finding'
        self.df.loc[self.df['Finding Labels'] == 'No Finding', 'Finding Labels'] = 'No_Finding'

        # drop unused columns
        drop_columns = ['Follow-up #',
                        'Patient Age', 'Patient Gender', 'View Position', 'OriginalImageWidth',
                        'OriginalImageHeight', 'OriginalImagePixelSpacing_x',
                        'OriginalImagePixelSpacing_y']
        self.df = self.df.drop(columns=drop_columns)

        # change column names
        columns_new = {'Image Index': self.IMAGE_INDEX,
                       'Finding Labels': self.FINDING_LABELS,
                       'Patient ID': self.PATIENT_ID}
        self.df = self.df.rename(columns=columns_new)

        # add columns with labels
        is_mass = self.df[self.FINDING_LABELS].str.contains(self.MASS)
        is_nodule = self.df[self.FINDING_LABELS].str.contains(self.NODULE)
        self.df[self.MASS_OR_NODULE] = (is_mass | is_nodule).astype(np.int)

    def _split_patients(self):

        np.random.seed(self.params.seed)

        patients_set = set(self.df[self.PATIENT_ID])
        size = len(patients_set)
        split_train, split_val, _ = self.params.splits

        patients_train = np.random.choice(list(patients_set), size=int(split_train * size), replace=False)
        patients_set = patients_set.difference(set(patients_train))

        patients_val = np.random.choice(list(patients_set), size=int(split_val * size), replace=False)

        patients_test = patients_set.difference(set(patients_val))

        return list(patients_train), list(patients_val), list(patients_test)

    def _split_patients_pneumonia(self):

        np.random.seed(self.params.seed)

        patients_with_pn = list(self.df[self.PATIENT_ID][self.df[self.PNEUMONIA] == 1])
        pwp_train, pwp_val, pwp_test = patients_with_pn[:42], patients_with_pn[42:52], patients_with_pn[52:]

        patients_set = set(self.df[self.PATIENT_ID])
        patients_set = patients_set.difference(patients_with_pn)
        size = len(patients_set)
        split_train, split_val, _ = self.params.splits

        patients_train = np.random.choice(list(patients_set), size=int(split_train * size), replace=False)
        patients_train = set(patients_train).union(set(pwp_train))
        patients_set = patients_set.difference(set(patients_train))

        patients_val = np.random.choice(list(patients_set), size=int(split_val * size), replace=False)
        patients_val = set(patients_val).union(set(pwp_val))

        patients_test = patients_set.difference(set(patients_val))
        patients_test = set(patients_test).union(pwp_test)

        return list(patients_train), list(patients_val), list(patients_test)

    def _get_dfs(self):

        self._clean_df()
        patients_train, patients_val, patients_test = self._split_patients()

        # split dataframe into train/val/test
        self.df_train = self.df[self.df['Patient_ID'].isin(patients_train)]
        self.df_val = self.df[self.df['Patient_ID'].isin(patients_val)]
        self.df_test = self.df[self.df['Patient_ID'].isin(patients_test)]

    def _get_datagens(self, class_mode='raw', horizontal_flip=True, shuffle=True):

        # get standard ImageDataGenerator
        idg_train = tf.keras.preprocessing.image.ImageDataGenerator(

            # the input pixels values are scaled between 0 and 1 and
            # each channel is normalized with respect to the ImageNet dataset
            # TODO check what is this function actually doing and do we need RGB images?
            preprocessing_function=tf.keras.applications.densenet.preprocess_input,

            # from the *paper*: "We also augment the training data with
            # random horizontal flipping.";
            # horizontal_flip is random here
            horizontal_flip=horizontal_flip
        )
        # no augmentation for val and test generators
        idg_val = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=tf.keras.applications.densenet.preprocess_input
        )

        if self.is_pneumonia:
            y_col = self.PNEUMONIA
        elif self.MASS_OR_NODULE:
            y_col = self.MASS_OR_NODULE
        else:
            raise NotImplemented
        idg_args = {
            'directory': self.IMAGE_PATH,

            # column in dataframe that contains the filenames
            'x_col': self.IMAGE_INDEX,

            # string or list, column/s in dataframe that has the target data
            'y_col': y_col,

            # if class_mode is "raw" or "multi_output" it should contain
            # the columns specified in y_col
            # TODO check 'raw' mode
            'class_mode': class_mode,

            'batch_size': self.params.batch_size,
            'shuffle': shuffle,
            'seed': self.params.seed,
            'target_size': self.params.input_shape
        }

        self.gen_train = idg_train.flow_from_dataframe(
            dataframe=self.df_train, **idg_args)

        self.gen_val = idg_val.flow_from_dataframe(
            dataframe=self.df_val, **idg_args)

        self.gen_test = idg_val.flow_from_dataframe(
            dataframe=self.df_test, **idg_args)


#################### *model* ####################


class CheXNet:

    def __init__(self,
                 params,
                 is_freeze=True,
                 # conv5_block1_0_bn
                 freeze_to=None,
                 is_dropout=False):

        self.params = params
        tf.random.set_seed(seed=self.params.seed)

        self.is_dropout = is_dropout

        self.base_model = None
        self.model = None
        self._build_model()
        if is_freeze:
            self._freeze()

        if freeze_to is not None:
            self.freeze_to = freeze_to
            self._freeze_to()

    def _build_model(self, weights='imagenet'):

        self.base_model = tf.keras.applications.densenet.DenseNet121(weights=weights,
                                                                     include_top=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(self.base_model.output)

        if self.is_dropout:
            # one more dense layer and dropout layer
            x = tf.keras.layers.Dense(self.params.dict['n_units'])(x)
            x = tf.keras.layers.Dropout(self.params.dict['dropout'])(x)

        output = tf.keras.layers.Dense(self.params.n_classes, activation=self.params.activation)(x)
        self.model = tf.keras.models.Model(inputs=self.base_model.input, outputs=output)

    def _freeze(self):
        for layer in self.base_model.layers:
            layer.trainable = False

    def _freeze_to(self):
        for layer in self.base_model.layers:
            if layer.name == self.freeze_to:
                break
            layer.trainable = False


#################### *trainer* ####################


class Trainer:

    def __init__(self,
                 experiment_dir=Path('/kaggle/input/chexnet-experiments/experiments'),
                 is_toy=True,
                 is_freeze=True,
                 is_fine_tune=False,
                 is_load_weights=False,
                 is_pneumonia=False,
                 is_mass_or_nodule=True,
                 freeze_to=None,
                 is_dropout=False,
                 is_custom_loss=True
                 ):
        tf.keras.backend.clear_session()

        ##### parameters and directories
        if is_fine_tune:
            is_freeze = False
            is_load_weights = True
            self.experiment_dir = experiment_dir / 'mass_base_model_ft'
        else:
            self.experiment_dir = experiment_dir / 'mass_base_model'
        self.params = Params(self.experiment_dir / 'params.json')
        self.working_dir = Path('/kaggle/working')

        ##### weight and history file
        self.is_freeze = is_freeze
        self.weight_file = self.working_dir / 'weights.h5'
        self.history_file = self.working_dir / 'history.pickle'

        ##### net and model
        self.CONV5_BLOCK = 'conv5_block1_0_bn'
        self.net = CheXNet(params=self.params,
                           is_freeze=is_freeze,
                           freeze_to=freeze_to,
                           is_dropout=is_dropout)
        self.model = self.net.model
        if is_load_weights:
            self.model.load_weights(str(self.weight_file))

        ##### data generator
        self.data_gen = CheXNetDataGen(params=self.params,
                                       is_toy=is_toy,
                                       is_pneumonia=is_pneumonia,
                                       is_mass_or_nodule=is_mass_or_nodule)

        ### optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.params.dict['learning_rate'])

        ##### metrics and loss
        self.class_weights = {1: self.params.dict['weight_negative'],
                              0: self.params.dict['weight_positive']}
        self.is_custom_loss = is_custom_loss
        if is_custom_loss:
            loss = get_weighted_loss(pos_weight=self.params.dict['weight_positive'],
                                     neg_weight=self.params.dict['weight_negative'])
        else:
            loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.model.compile(loss=loss,
                           optimizer=self.optimizer,
                           metrics=["accuracy"])

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

        ##### steps per epoch
        self.steps_per_epoch = 0
        self.validation_steps = 0
        self.test_steps = 0
        self._get_steps()

    def train(self, is_plot=True):
        # we use here fit() instead of deprecated fit_generator()
        if self.is_custom_loss:
            weights = None
        else:
            weights = self.class_weights
        history = self.model.fit(self.data_gen.gen_train,
                                 batch_size=self.params.dict['batch_size'],
                                 steps_per_epoch=self.steps_per_epoch,
                                 validation_data=self.data_gen.gen_val,
                                 validation_steps=self.validation_steps,
                                 epochs=self.params.dict['epochs'],
                                 validation_batch_size=self.params.dict['batch_size'],
                                 callbacks=self.callbacks,
                                 class_weight=weights)

        save_history(history, self)

        if is_plot:
            plot_metric('loss', self)

        return history

    def _get_steps(self):
        # unpack some params
        dataset_size = self.params.dict['dataset_size']
        batch_size = self.params.dict['batch_size']
        train_split, val_split, test_split = self.params.dict['splits']

        self.steps_per_epoch = train_split * dataset_size // batch_size
        self.validation_steps = val_split * dataset_size // batch_size
        self.test_steps = test_split * dataset_size // batch_size

    def evaluate(self, is_verbose=False):

        # it returns array of shape (bs, 2) in case of Pneumonia task
        # so we need to transform it into the correct shape (bs,) -
        # we use sparse labels here: [0, 1, 0, ... ]
        y_pred = self.model.predict(self.data_gen.gen_test,
                                    batch_size=self.params.dict['batch_size'],
                                    steps=self.test_steps)
        y_pred = np.argmax(y_pred, axis=1)

        # we may get true labels using this attribute
        y_true = self.data_gen.gen_test.labels

        f1 = f1_score(y_true=y_true, y_pred=y_pred)
        conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)

        if is_verbose:
            print(f'f1_score={f1:.4f}')
            print(f'confusion matrix:\n{conf_matrix}')

        return f1, conf_matrix

    def train_and_evaluate(self):
        self.train()
        self.evaluate()
