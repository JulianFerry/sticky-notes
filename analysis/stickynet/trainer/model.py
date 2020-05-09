# Tensorflow
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
# TF extensions
from tensorboard.plugins.hparams import api as hp
import hypertune
# Python
import os
import math
import json
from google.cloud import storage

# Add before any TF calls - initializes the keras global outside of any tf.functions
temp = tf.zeros([4, 32, 32, 3])
preprocess_input(temp)

AUTOTUNE = tf.data.experimental.AUTOTUNE


# Data loading functions

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
        tfrecord - tf.data.TFRecordDataset - loaded tfrecord dataset containing training examples
        tfrecord_feature_description - dict - dict mapping used to parse tf.Example features
        image_shape - tuple (height, width, channels) - parse this from the METADATA file
    """
    # Parse single example
    tf_example = tf.io.parse_single_example(tfrecord, tfrecord_feature_description)
    # Decode and preprocess image
    image = tf.io.decode_raw(tf_example['image/encoded'], tf.uint8)
    image = tf.reshape(image, image_shape)
    image = preprocess_input(image)
    # Decode label
    label = tf_example['image/object/class/text']
    label = (label == 'stickie')
    return image, label


def read_dataset(tfrecord_path, batch_size=32, repeat=None, **kwargs):
    """
    Read tfrecord dataset of images, labels and bounding boxes from storage
    """
    metadata = load_metadata(tfrecord_path)

    # Determine how many steps to run per epoch from the metadata
    split = tfrecord_path.split('/')[-1].split('.')[0]  # returns 'train', 'val' or 'test'
    num_examples = metadata['num_examples'][split]
    num_steps = num_examples // batch_size

    # Arguments for data parsing
    image_shape = (
        metadata['dimensions']['height'],
        metadata['dimensions']['width'],
        metadata['dimensions']['channels']
    )
    tfrecord_feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/class/text': tf.io.FixedLenFeature([], tf.string)
    }

    # Load and parse data
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(
        lambda x: parse_image(x, tfrecord_feature_description, image_shape),
        num_parallel_calls=AUTOTUNE
    )
    # Repeat, shuffle, batch and prefetch
    dataset = (dataset.repeat(repeat)
                      .shuffle(num_examples, **kwargs)
                      .batch(batch_size)
                      .prefetch(AUTOTUNE))

    return dataset, num_steps


# Custom callbacks

class AIplatformMetricCallback(tf.keras.callbacks.Callback):
    """
    Metric callback for AI platform hyperparameter tuning
    Eager execution mode only (there might be a way to use @tf.function)
    """
    def __init__(self, metric):
        self.metric = 'val_' + metric
        self.hpt = hypertune.HyperTune()

    def on_epoch_end(self, epoch, logs):
        self.hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag=self.metric,
            metric_value=logs[self.metric],
            global_step=epoch
        )


class TensorBoardExtended(TensorBoard):
    """
    Wrapper around TensorBoard to allow epoch counting to start at defined number
    """
    def __init__(self, first_epoch=None, steps_per_epoch=None,
                 validation_data=None, batch_val_steps=None,
                 *args, **kwargs):
        # Run TensorBoard.__init__() - this creates self.update_freq
        super().__init__(*args, **kwargs)
        # For on_epoch_end()
        if first_epoch is None:
            first_epoch = 0
        self.first_epoch = first_epoch
        self.current_epoch = first_epoch
        # For on_train_batch_end()
        self.steps_per_epoch = steps_per_epoch
        self.validation_data = validation_data
        self.batch_val_steps = batch_val_steps

    def on_epoch_end(self, epoch, logs=None):
        """
        Starts counting epochs at `self.first_epoch`
        """
        self.current_epoch = self.first_epoch + epoch
        super().on_epoch_end(self.current_epoch, logs)

    def on_train_batch_end(self, batch, logs={}):
        """
        Starts counting batches based on `self.first_epoch`
        Evaluates the model on validation data after `self.update_freq` training batches
        """
        current_batch = self.current_epoch * self.steps_per_epoch + batch
        if self.validation_data:
            if isinstance(self.update_freq, int) & (batch % self.update_freq == 0):
                batch_loss, batch_accuracy = self.model.evaluate(
                    self.validation_data,
                    steps=self.batch_val_steps,
                    verbose=0
                )
                logs['batch'] = current_batch
                logs['val_accuracy'] = batch_accuracy
                logs['val_loss'] = batch_loss
        super().on_train_batch_end(current_batch, logs)


def create_hparams_callback(log_dir, opt_metric, hparams, args):
    """
    Set up Hprams plugin config and callback for Tensorboard
    """
    hparams_dir = os.path.join(log_dir, 'validation')
    opt_metric = 'epoch_' + opt_metric

    # Hparams callback to log the hyperparameter values
    with tf.summary.create_file_writer(hparams_dir).as_default():
        hp.hparams_config(
            hparams=[hp.HParam(hparam)for hparam in hparams],
            metrics=[hp.Metric(opt_metric)]
        )
    hparams_cb = hp.KerasCallback(
        writer=hparams_dir,
        hparams={hparam: args[hparam] for hparam in hparams}
    )
    return hparams_cb


# Model definition

def create_model(args, metrics):
    """
    Create trainable model initialised from VGG-16 pretrained on ImageNet
    """
    # Load pre-trained model
    weights = None
    if args.get('initial_weights_path') is None:
        weights = 'imagenet'
    vgg = VGG16(weights=weights, input_tensor=Input(shape=(224, 224, 3)), include_top=False)
    # Freeze the model from training, except for the last n convolutional blocks
    vgg.trainable = False
    if args['trainable_blocks'] > 0:
        trainable_layers = args['trainable_blocks'] * 4
        for layer in vgg.layers[-trainable_layers:]:
            layer.trainable = True

    # Add trainable output layer
    flatten_layer = Flatten()
    dropout_layer = Dropout(args['dropout_rate'])
    dense_layer = Dense(1, kernel_regularizer=l2(l=args['l2_regularisation']))
    sigmoid_layer = Activation('sigmoid')
    # Apply layers to output using Keras functional API
    output = vgg.layers[-1].output
    output = flatten_layer(output)
    output = dropout_layer(output)
    output = dense_layer(output)
    output = sigmoid_layer(output)
    model = Model(vgg.input, output)

    # Load weights (from gcloud or local storage)
    weights_path = args.get('initial_weights_path')
    if weights_path is not None:
        print('Initialising model with weights from:', weights_path)
        model.load_weights(weights_path)

    # Compile
    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=args['learning_rate']),
        metrics=metrics
    )

    return model


# Training

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
    opt_metric = 'accuracy'
    hparams = ['learning_rate', 'l2_regularisation', 'dropout_rate', 'trainable_blocks']
    log_dir = os.path.join(args['job_dir'], 'training-logs')
    model_dir = os.path.join(args['job_dir'], 'model-weights.tf')

    # Load data
    train_tfrecord_path = os.path.join(args['data_dir'], 'train.tfrecord')
    val_tfrecord_path = os.path.join(args['data_dir'], 'val.tfrecord')
    train_dataset, train_steps = read_dataset(train_tfrecord_path, args['batch_size'])
    val_dataset, val_steps = read_dataset(val_tfrecord_path, args['batch_size'])

    # Model definition
    model = create_model(args, metrics)

    # Callback definition
    tensorboard_cb = TensorBoardExtended(
        log_dir=log_dir,
        update_freq=math.ceil(train_steps / 5),
        first_epoch=args['first_epoch'],
        steps_per_epoch=train_steps,
        validation_data=val_dataset,
        batch_val_steps=math.ceil(val_steps / 5)
    )
    checkpoint_cb = ModelCheckpoint(
        filepath=model_dir,
        save_format='tf',
        monitor='val_' + opt_metric,
        mode='max',
        save_freq='epoch',
        save_weights_only=True,
        save_best_only=True,
        verbose=0
    )
    ai_platform_metric_cb = AIplatformMetricCallback(
        metric=opt_metric
    )
    hparams_cb = create_hparams_callback(log_dir, opt_metric, hparams, args)
    callbacks = [
        tensorboard_cb,
        checkpoint_cb,
        ai_platform_metric_cb,
        hparams_cb
    ]

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
