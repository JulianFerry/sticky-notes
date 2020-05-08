import argparse
from . import model


def get_args():
    """
    Argument parser

    Returns:
      Dictionary of arguments.
    """

    parser = argparse.ArgumentParser()

    # Training arguments
    parser.add_argument(
        '--epochs',
        help = 'Epochs to run the training job for',
        type = int,
        default = 1
    )
    parser.add_argument(
        '--batch_size',
        help = 'Batch size for training steps',
        type = int,
        default = 32
    )
    # Hyperparameters
    parser.add_argument(
        '--learning_rate',
        help = 'Learning rate to use for optimization',
        type = float,
        default = 0.001
    )
    parser.add_argument(
        '--l2_regularisation',
        help = 'Regularisation rate to use for dense layers',
        type = float,
        default = 0.01
    )
    parser.add_argument(
        '--dropout_rate',
        help = 'Dropout rate to use for dense layers',
        type = float,
        default = 0
    )
    parser.add_argument(
        '--trainable_blocks',
        help = 'Number of VGG convolutional blocks to train at the top of the network',
        type = int,
        default = 0
    )

    # Checkpoint arguments
    parser.add_argument(
        '--checkpoint_epochs',
        help = 'Create a checkpoint every n epochs',
        default = 1,
        type = int
    )
    parser.add_argument(
        '--first_epoch',
        help = 'Epoch at which to start counting for training logs',
        default = 0,
        type = int
    )

    # Eval arguments
    parser.add_argument(
        '--eval_steps',
        help = 'Number of steps to run evalution for at each checkpoint',
        type = int,
        default = 10
    )
    parser.add_argument(
        '--eval_delay_secs',
        help = 'How long to wait before running first evaluation',
        default = 10,
        type = int
    )
    parser.add_argument(
        '--throttle_secs',
        help = 'Seconds between evaluations',
        default = 300,
        type = int
    )

    # Paths
    parser.add_argument(
        '--job-dir',
        help = 'GCS location to write logs and checkpoint model weights',
    )
    parser.add_argument(
        '--data_dir',
        help = 'GCS location to fetch data from',
    )
    parser.add_argument(
        '--initial_weights_path',
        help = 'Checkpointed model weights to load at the start of training',
        default = None
    )

    # Parse
    args = parser.parse_args()
    args = args.__dict__
    return args


if __name__ == '__main__':
    # Parse command-line arguments
    args = get_args()
    # Run the training job
    model.train_and_evaluate(args)
