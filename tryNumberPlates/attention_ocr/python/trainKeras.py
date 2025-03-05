# Import necessary libraries
import collections
import logging

import tensorflow as tf

from tensorflow.keras import layers, models

from absl import app
from absl import flags
# from tensorflow import flags

import data_provider
import common_flags

# Define FLAGS
FLAGS = flags.FLAGS
common_flags.define()

# Flag definitions
flags.DEFINE_integer('task', 0, 'The Task ID.')
flags.DEFINE_integer('ps_tasks', 0, 'The number of parameter servers.')
flags.DEFINE_integer('save_summaries_secs', 60, 'The frequency of saving summaries.')
flags.DEFINE_integer('save_interval_secs', 600, 'Frequency in seconds of saving the model.')
flags.DEFINE_integer('max_number_of_steps', int(1e10), 'The maximum number of gradient steps.')
flags.DEFINE_string('checkpoint_inception', '', 'Checkpoint for inception weights.')
flags.DEFINE_float('clip_gradient_norm', 2.0, 'Gradient clipping norm.')
flags.DEFINE_bool('sync_replicas', False, 'Synchronize replicas during training.')
flags.DEFINE_integer('replicas_to_aggregate', 1, 'The number of gradient updates before updating params.')
flags.DEFINE_integer('total_num_replicas', 1, 'Total number of worker replicas.')
flags.DEFINE_integer('startup_delay_steps', 15, 'Number of training steps between replicas startup.')
flags.DEFINE_boolean('reset_train_dir', False, 'Reset the training directory.')
flags.DEFINE_boolean('show_graph_stats', False, 'Output model size stats.')

# Hyperparameters
TrainingHParams = collections.namedtuple('TrainingHParams', [
    'learning_rate', 'optimizer', 'momentum', 'use_augment_input'])


def get_training_hparams():
    return TrainingHParams(
        learning_rate=FLAGS.learning_rate,
        optimizer=FLAGS.optimizer,
        momentum=FLAGS.momentum,
        use_augment_input=FLAGS.use_augment_input)


def create_optimizer(hparams):
    """Creates optimizer based on the specified flags."""
    if hparams.optimizer == 'momentum':
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=hparams.learning_rate, momentum=hparams.momentum)
    elif hparams.optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=hparams.learning_rate)
    elif hparams.optimizer == 'adadelta':
        optimizer = tf.keras.optimizers.Adadelta(learning_rate=hparams.learning_rate)
    elif hparams.optimizer == 'adagrad':
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=hparams.learning_rate)
    elif hparams.optimizer == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=hparams.learning_rate, momentum=hparams.momentum)
    return optimizer


def train(model, dataset, hparams):
    """Runs the training loop."""
    optimizer = create_optimizer(hparams)

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        if FLAGS.clip_gradient_norm > 0:
            gradients = [tf.clip_by_norm(g, FLAGS.clip_gradient_norm) for g in gradients]
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    # Prepare the training loop
    for step, (images, labels) in enumerate(dataset):
        loss = train_step(images, labels)
        if step % FLAGS.save_summaries_secs == 0:
            logging.info(f'Step {step}, Loss: {loss.numpy()}')
        if step % FLAGS.save_interval_secs == 0:
            model.save_weights(f'{FLAGS.train_log_dir}/model_checkpoint_{step}')


def prepare_training_dir():
    if not tf.io.gfile.exists(FLAGS.train_log_dir):
        logging.info(f'Create a new training directory {FLAGS.train_log_dir}')
        tf.io.gfile.makedirs(FLAGS.train_log_dir)
    else:
        if FLAGS.reset_train_dir:
            logging.info(f'Reset the training directory {FLAGS.train_log_dir}')
            tf.io.gfile.rmtree(FLAGS.train_log_dir)
            tf.io.gfile.makedirs(FLAGS.train_log_dir)
        else:
            logging.info(f'Use already existing training directory {FLAGS.train_log_dir}')


def main(_):
    prepare_training_dir()

    dataset = common_flags.create_dataset(split_name=FLAGS.split_name)
    model = common_flags.create_model(dataset.num_char_classes,
                                      dataset.max_sequence_length,
                                      dataset.num_of_views, dataset.null_code)
    hparams = get_training_hparams()

    # Prepare the dataset
    data = data_provider.get_data(
        dataset,
        FLAGS.batch_size,
        augment=hparams.use_augment_input,
        central_crop_size=common_flags.get_crop_size())

    # Compile the model
    model.compile(optimizer=create_optimizer(hparams),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Train the model
    train(model, data, hparams)

if __name__ == '__main__':
    tf.compat.v1.app.run()
