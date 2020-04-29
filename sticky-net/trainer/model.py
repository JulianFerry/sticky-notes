# Tensorflow
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, Dropout, MaxPool2D, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
# TF extensions
from tensorboard.plugins.hparams import api as hp
import hypertune
# Python
import os
import json
from google.cloud import storage


# Add before any TF calls - initializes the keras global outside of any tf.functions
temp = tf.zeros([4, 32, 32, 3])
preprocess_input(temp)
AUTOTUNE = tf.data.experimental.AUTOTUNE
# strategy = tf.distribute.MirroredStrategy()
# print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


def load_metadata(tfrecord_path):
    """
    Load METADATA.json file from the tfrecord parent directory
    """
    data_dir = os.path.dirname(tfrecord_path)
    if data_dir.startswith('gs://'):
        # Extract bucket and blob name from path
        bucket_name = data_dir[5:].split('/')[0]
        data_dir = data_dir.replace(f'gs://{bucket_name}/', '')
        # Load metadata
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blob = bucket.get_blob(os.path.join(data_dir, 'METADATA.json'))
        metadata = blob.download_as_string()
        metadata = json.loads(metadata)
    else:
        metadata = json.load(open(os.path.join(data_dir, 'METADATA.json')))
    return metadata


def parse_image(tfrecord, tfrecord_feature_description, image_shape):
    """
    Parse image, label and bounding box from a tfrecord example

    Arguments:
    - tfrecord - tf.data.TFRecordDataset - loaded tfrecord dataset containing training examples
    - tfrecord_feature_description - dict - dict mapping used to parse tf.Example features
    - image_shape - tuple - should be parsed from the METADATA file, with format (height, width, channels)
    """
    # Parse single example
    tf_example = tf.io.parse_single_example(tfrecord, tfrecord_feature_description)
    # Decode and preprocess image
    image = tf.io.decode_raw(tf_example['image_raw'], tf.uint8)
    image = tf.reshape(image, image_shape)
    image = preprocess_input(image)
    # Decode label
    label = tf_example['label']
    label = (label == 'stickie')
    #bbox = tf_example['bbox']
    return image, label


def read_dataset(tfrecord_path, batch_size=32):
    """
    Read tfrecord dataset of images, labels and bounding boxes from storage
    """
    metadata = load_metadata(tfrecord_path)

    # Determine how many steps to run per epoch from the metadata
    split = tfrecord_path.split('/')[-1].split('.')[0]  # returns train/val/test
    num_examples = metadata['num_examples'][split]
    num_steps = num_examples // batch_size

    # Arguments for data parsing
    image_shape = (
        metadata['dimensions']['height'],
        metadata['dimensions']['width'],
        metadata['dimensions']['channels']
    )
    tfrecord_feature_description = {
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
        'bbox': tf.io.FixedLenFeature([], tf.string)
    }

    # Load and parse data
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(
        lambda x: parse_image(x, tfrecord_feature_description, image_shape),
        num_parallel_calls=AUTOTUNE
    )
    # Repeat, shuffle, batch and prefetch
    dataset = dataset.repeat(None).shuffle(num_examples).batch(batch_size).prefetch(AUTOTUNE)

    return dataset, num_steps


# Custom callbacks
class AIplatformMetricCallback(tf.keras.callbacks.Callback):
    """
    Metric callback for AI platform hyperparameter tuning
    Eager execution mode only (there might be a way to use @tf.function)
    """
    def __init__(self, metric):
        self.metric = metric
        self.hpt = hypertune.HyperTune()

    def on_epoch_end(self, epoch, logs):
        self.hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag=self.metric,
            metric_value=logs[self.metric],
            global_step=epoch
        )


class HparamsMetricCallback(tf.keras.callbacks.Callback):
    """
    Metric callback for Hparams dashboard
    Eager execution mode only (there might be a way to use @tf.function)
    """
    def __init__(self, metric, log_dir):
        """
        Arguments:
        - metric - str - validation metric (should correspond to a metric used in `model.compile`)
        - log_dir - str - log directory to store the metric (should be same dir as Tensorboard)
        
        Example:
        ```
        model.compile(..., metrics=['accuracy'])
        tensorboard_cb = Tensorboard(log_dir=log_dir)
        hparams_metric_cb = HparamsMetricCallback(metric='val_accuracy', log_dir=log_dir)
        ```
        """
        self.metric = metric
        self.log_dir = log_dir

    def on_epoch_end(self, epoch, logs):
        """
        This function will automatically be called during a model.fit() call
        Creates a tf.summary from the validation metric stored in the training logs
        """
        with tf.summary.create_file_writer(self.log_dir).as_default():
            tf.summary.scalar(self.metric, logs[self.metric], epoch)


def create_hparams_callbacks(log_dir, opt_metric, hparams, args):
    """
    Create the two callbacks necessary to use hparams in Tensorboard
    """
    # Hparams metric callback to log the validation score
    hparams_metric_cb = HparamsMetricCallback(
        metric=opt_metric,
        log_dir=log_dir
    )
    # Hparams callback to log the hyperparameter values
    with tf.summary.create_file_writer(log_dir).as_default():
        hp.hparams_config(
            hparams=[hp.HParam(hparam)for hparam in hparams],
            metrics=[hp.Metric(opt_metric)]
        )
    hparams_cb = hp.KerasCallback(
        writer=log_dir,
        hparams={hparam: args[hparam] for hparam in hparams}
    )
    return hparams_metric_cb, hparams_cb


def create_model(args, metrics):
    """
    Create trainable model initialised from VGG-16 pretrained on ImageNet
    """
    # Pre-trained model
    vgg = VGG16(weights='imagenet', input_tensor=Input(shape=(224, 224, 3)), include_top=False)
    vgg.trainable = False
    for layer in vgg.layers:
        layer.trainable = False

    # Add trainable output layer
    flatten_layer = Flatten()
    output_layer = Dense(1, activation='sigmoid', kernel_regularizer=l2(l=args['l2_regularisation']))
    output = vgg.layers[-1].output
    output = output_layer(flatten_layer(output))
    model = Model(vgg.input, output)

    # Compile
    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=args['learning_rate']),
        metrics=metrics
    )

    return model


def train_and_evaluate(args):
    """
    Main training function
    Training logs and model checkpoints will be stored in args['job_dir']

    Arguments:
    - args - dict - Training parameters.
      Should contain:
        - 'learning_rate'     - float - initial learning rate for training
        - 'l2_regularisation' - float - regularisation used for dense (fully connected) layers
        - 'batch_size'        - int   - mini-batch size used using training (Adam optimisation)
        - 'epochs'            - int   - number of training epochs
        - 'job_dir'           - str   - job directory used to store the logs and model checkpoints
    """
    # Training parameters
    metrics = ['accuracy']
    opt_metric = 'val_accuracy'
    hparams = ['learning_rate', 'l2_regularisation', 'batch_size']
    log_dir = os.path.join(args['job_dir'], 'training-logs')
    model_dir = os.path.join(args['job_dir'], 'model-weights.tf')

    # Model definition
    model = create_model(args, metrics)

    # Callback definition
    tensorboard_cb = TensorBoard(
        log_dir=log_dir
    )
    checkpoint_cb = ModelCheckpoint(
        filepath=model_dir,
        save_format='tf',
        monitor=opt_metric,
        mode='max',
        save_freq='epoch',
        save_weights_only=True,
        save_best_only=True,
        verbose=0
    )
    ai_platform_metric_cb = AIplatformMetricCallback(
        metric=opt_metric
    )
    hparams_metric_cb, hparams_cb = create_hparams_callbacks(log_dir, opt_metric, hparams, args)
    callbacks = [tensorboard_cb, checkpoint_cb, ai_platform_metric_cb, hparams_metric_cb, hparams_cb]

    # Load data
    train_tfrecord_path = os.path.join(args['data_dir'], 'train.tfrecord')
    val_tfrecord_path = os.path.join(args['data_dir'], 'val.tfrecord')
    train_dataset, train_steps = read_dataset(train_tfrecord_path, args['batch_size'])
    val_dataset, val_steps = read_dataset(val_tfrecord_path, args['batch_size'])

    # Train model
    model.fit(
        train_dataset,
        epochs=args['epochs'],
        steps_per_epoch=train_steps,
        validation_data=val_dataset,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1
    )
    return model
