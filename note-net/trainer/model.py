
# Tensorflow
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPool2D, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
# TF extensions
import tensorflow_datasets as tfds
import hypertune
from tensorboard.plugins.hparams import api as hp
# Python
import os
from functools import partial

# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'credentials/mnist-keras-mle-gcs-access.json'


# Create an input function that loads data as a Tensorflow dataset
def read_dataset(split, train_val_ratio, batch_size):

    # Load data
    tfds_splits = {
        'train': f'train[:{train_val_ratio}%]',
        'validation': f'train[{train_val_ratio}%:]',
        'test': 'test'
    }
    dataset, info = tfds.load(
        data_dir='gs://tfds-data/datasets',
        name='mnist',
        split=tfds_splits[split],
        as_supervised=True,
        with_info=True
    )

    # Prepare data using tf.data.dataset API
    if split == 'train':
        dataset = dataset.shuffle(buffer_size=batch_size*100) # noqa
        num_epochs = None   # unlimited
    else:
        num_epochs = 1      # end-of-input after this

    def preprocess_fn(image, label):
        """Transformation function to preprocess raw data into trainable input"""
        image = tf.cast(image, tf.float32)
        return image, label
    dataset = dataset.map(preprocess_fn)
    dataset = dataset.repeat(num_epochs).batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    num_steps = int(info.splits[tfds_splits[split]].num_examples / batch_size)
    return dataset, num_steps


# Define model using subclasing API
class MyNet(tf.keras.Model):
    def __init__(self, args, name='mynet'):
        super(MyNet, self).__init__(name=name)
        DenseReg = partial(Dense, kernel_regularizer=l2(l=args['l2_regularisation']))
        self.conv2d_1 = Conv2D(filters=32, kernel_size=5, input_shape=(28, 28, 1))
        self.relu = Activation('relu')
        self.max_pool = MaxPool2D()
        self.conv2d_2 = Conv2D(filters=64, kernel_size=5)
        self.flatten = Flatten()
        self.dense = DenseReg(128)
        self.dropout = Dropout(0.4)
        self.out = DenseReg(10, activation='softmax')

    def call(self, inputs):
        x = self.conv2d_1(inputs)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.out(x)


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
        self.metric = metric
        self.log_dir = log_dir

    def on_epoch_end(self, epoch, logs):
        with tf.summary.create_file_writer(self.log_dir).as_default():
            tf.summary.scalar(self.metric, logs[self.metric], epoch)


def train_and_evaluate(args):

    # Training parameters
    metrics = ['accuracy']
    opt_metric = 'val_accuracy'
    hparams = ['learning_rate', 'l2_regularisation']
    log_dir = os.path.join(args['job_dir'], 'training-logs')
    model_dir = os.path.join(args['job_dir'], 'model-weights.tf')

    # Model definition
    model = MyNet(args)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(learning_rate=args['learning_rate']),
        metrics=metrics
    )

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
        verbose=1
    )
    ai_platform_metric_cb = AIplatformMetricCallback(
        metric=opt_metric
    )
    hparams_metric_cb = HparamsMetricCallback(
        metric=opt_metric,
        log_dir=log_dir
    )
    with tf.summary.create_file_writer(log_dir).as_default():
        hp.hparams_config(
            hparams=[hp.HParam(hparam)for hparam in hparams],
            metrics=[hp.Metric(opt_metric)]
        )
    hparams_cb = hp.KerasCallback(
        writer=log_dir,
        hparams={hparam: args[hparam] for hparam in hparams}
    )
    callbacks = [
        tensorboard_cb, checkpoint_cb, ai_platform_metric_cb, hparams_metric_cb, hparams_cb]

    # Load data
    train_val_ratio = 80  # percentage
    train_dataset, train_steps = read_dataset('train', train_val_ratio, args['batch_size'])
    val_dataset, val_steps = read_dataset('validation', train_val_ratio, args['batch_size'])

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
