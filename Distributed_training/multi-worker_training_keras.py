# Multi-worker training with Keras

# Setup
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow_datasets as tfds
import tensorflow as tf
import os
import json

tfds.disable_progress_bar()

# Preparing dataset
BUFFER_SIZE = 10000
BATCH_SIZE = 64


def make_datasets_unbatched():
    # Scaling MNIST data from (0, 255] to (0., 1.]
    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label

    datasets, info = tfds.load(name='mnist',
                               with_info=True,
                               as_supervised=True)

    return datasets['train'].map(scale).cache().shuffle(BUFFER_SIZE)


train_datasets = make_datasets_unbatched().batch(BATCH_SIZE)


# Build the Keras model
def build_and_compile_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        metrics=['accuracy'])
    return model


single_worker_model = build_and_compile_cnn_model()
single_worker_model.fit(x=train_datasets, epochs=3)

# Multi-worker Configuration
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["localhost:12345", "localhost:23456"]
    },
    'task': {'type': 'worker', 'index': 0}
})

# Choose the right strategy
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

# Train the model with MultiWorkerMirroredStrategy
NUM_WORKERS = 2
# Here the batch size scales up by number of workers since
# `tf.data.Dataset.batch` expects the global batch size. Previously we used 64,
# and now this becomes 128.
GLOBAL_BATCH_SIZE = 64 * NUM_WORKERS
with strategy.scope():
    # Creation of dataset, and model building/compiling need to be within
    # `strategy.scope()`.
    train_datasets = make_datasets_unbatched().batch(GLOBAL_BATCH_SIZE)
    multi_worker_model = build_and_compile_cnn_model()
multi_worker_model.fit(x=train_datasets, epochs=3)

# Dataset sharding and batch size
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
train_datasets_no_auto_shard = train_datasets.with_options(options)

# ModelCheckpoint callback
# Replace the `filepath` argument with a path in the file system
# accessible by all workers.
callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='/tmp/keras-ckpt')]
with strategy.scope():
    multi_worker_model = build_and_compile_cnn_model()
multi_worker_model.fit(x=train_datasets, epochs=3, callbacks=callbacks)
