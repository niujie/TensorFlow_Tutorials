# Custom training with tf.distribute.Strategy

from __future__ import absolute_import, division, print_function, unicode_literals

# Import TensorFlow
import tensorflow as tf

# Helper libraries
import numpy as np
import os

print(tf.__version__)

# Download the fashion MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Adding a dimension to the array -> new shape == (28, 28, 1)
# We are doing this because the first layer in our model is a convolutional
# layer and it requires a 4D input (batch_size, height, width, channels).
# batch_size dimension will be added later on.
train_images = train_images[..., None]
test_images = test_images[..., None]

# Getting the images in [0, 1] range.
train_images = train_images / np.float32(255)
test_images = test_images / np.float32(255)

# Create a strategy to distribute the variables and the graph
# If the list of devices is not specified in the
# `tf.distribute.MirroredStrategy` constructor, it will be auto-detected.
strategy = tf.distribute.MirroredStrategy()

print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Setup input pipeline
BUFFER_SIZE = len(train_images)

BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

EPOCHS = 10

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)) \
    .shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)) \
    .batch(GLOBAL_BATCH_SIZE)

train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)


# Create the model
def create_model():
    _model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    return _model


# Create a checkpoint directory to store the checkpoints.
checkpoint_dir = '.\\training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

# Define the loss function
with strategy.scope():
    # Set reduction to `none` so we can do the reduction afterwards and divide by
    # global batch size.
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE)

    # or loss_fn = tf.keras.losses.sparse_categorical_crossentropy
    def compute_loss(_labels, predictions):
        per_example_loss = loss_object(_labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

# Define the metrics to track loss and accuracy
with strategy.scope():
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='test_accuracy')

# Training loop
# model and optimizer must be created under `strategy.scope`.
with strategy.scope():
    model = create_model()

    optimizer = tf.keras.optimizers.Adam()

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

with strategy.scope():
    def train_step(inputs):
        _images, _labels = inputs

        with tf.GradientTape() as tape:
            predictions = model(_images, training=True)
            loss = compute_loss(_labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_accuracy.update_state(_labels, predictions)
        return loss


    def test_step(inputs):
        _images, _labels = inputs

        predictions = model(_images, training=False)
        t_loss = loss_object(_labels, predictions)

        test_loss.update_state(t_loss)
        test_accuracy.update_state(_labels, predictions)

with strategy.scope():
    # `experimental_run_v2` replicates the provided computation and runs it
    # with the distributed input.
    @tf.function
    def distributed_train_step(dataset_inputs):
        per_replica_losses = strategy.experimental_run_v2(train_step,
                                                          args=(dataset_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                               axis=None)


    @tf.function
    def distributed_test_step(dataset_inputs):
        return strategy.experimental_run_v2(test_step, args=(dataset_inputs,))


    for epoch in range(EPOCHS):
        # TRAIN LOOP
        total_loss = 0.0
        num_batches = 0
        for x in train_dist_dataset:
            total_loss += distributed_train_step(x)
            num_batches += 1
        train_loss = total_loss / num_batches

        # TEST LOOP
        for x in test_dist_dataset:
            distributed_test_step(x)

        if epoch % 2 == 0:
            checkpoint.save(checkpoint_prefix)

        template = ("Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, "
                    "Test Accuracy: {}")
        print(template.format(epoch + 1, train_loss,
                              train_accuracy.result() * 100, test_loss.result(),
                              test_accuracy.result() * 100))

        test_loss.reset_states()
        train_accuracy.reset_states()
        test_accuracy.reset_states()

# Restore the latest checkpoint and test
eval_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='eval_accuracy')

new_model = create_model()
new_optimizer = tf.keras.optimizers.Adam()

test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(GLOBAL_BATCH_SIZE)


@tf.function
def eval_step(_images, _labels):
    predictions = new_model(_images, training=False)
    eval_accuracy(_labels, predictions)


checkpoint = tf.train.Checkpoint(optimizer=new_optimizer, model=new_model)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

for images, labels in test_dataset:
    eval_step(images, labels)

print('Accuracy after restoring the saved model without strategy: {}'.format(
    eval_accuracy.result() * 100))

# Alternate ways of iterating over a dataset
# Using iterators
with strategy.scope():
    for _ in range(EPOCHS):
        total_loss = 0.0
        num_batches = 0
        train_iter = iter(train_dist_dataset)

        for _ in range(10):
            total_loss += distributed_train_step(next(train_iter))
            num_batches += 1
        average_train_loss = total_loss / num_batches

        template = "Epoch {}, Loss: {}, Accuracy: {}"
        print(template.format(epoch + 1, average_train_loss, train_accuracy.result() * 100))
        train_accuracy.reset_states()

# Iterating inside a tf.function
with strategy.scope():
    @tf.function
    def distributed_train_epoch(dataset):
        _total_loss = 0.0
        _num_batches = 0
        for _x in dataset:
            per_replica_losses = strategy.experimental_run_v2(train_step,
                                                              args=(_x,))
            _total_loss += strategy.reduce(
                tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
            _num_batches += 1
        return _total_loss / tf.cast(_num_batches, dtype=tf.float32)


    for epoch in range(EPOCHS):
        train_loss = distributed_train_epoch(train_dist_dataset)

        template = "Epoch {}, Loss: {}, Accuracy: {}"
        print(template.format(epoch + 1, train_loss, train_accuracy.result() * 100))

        train_accuracy.reset_states()