## 01-pneumonia-downsampling-1

- we down sample the dataset to get the equal number of pneumonia 
and non-pneumonia training examples; so `DOWN_SAMPLING_RATIO = 1`; 
full list of parameters is below;

- results are not very promising - we have very low `val_accuracy` (~.6) 
and big gap between training and validation `accuracy` (over fitting);

```python
import tensorflow as tf
class Params:

    # hyper-parameters
    LEARNING_RATE = 0.0001 # fine-tuning
    BATCH_SIZE = 32
    EPOCHS = 20 # fine-tuning

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

    # down sampling params
    DOWN_SAMPLING_RATIO = 1

    # parameters for the model
    IMAGE_SIZE = [224, 224]
    CHANNELS = 3
    WEIGHTS = 'imagenet'
    N_CLASSES_PNEUMONIA = 1
    N_CLASSES = len(CLASSES)
    ACTIVATION = 'sigmoid'

    # parameters for the trainer
    LOSS = 'binary_crossentropy'
    METRICS = [tf.keras.metrics.BinaryAccuracy(),
               tf.keras.metrics.Precision(),
               tf.keras.metrics.Recall()]
    TRAIN_SPLIT = .8
    VAL_SPLIT = .1
    TEST_SPLIT = .1

    SEED = 42
```