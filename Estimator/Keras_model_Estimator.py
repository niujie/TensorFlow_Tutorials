# Create an Estimator from a Keras model

# Setup
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import numpy as np
import tensorflow_datasets as tfds

# Create a simple Keras model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()


# Create an input function
def input_fn():
    split = tfds.Split.TRAIN
    dataset = tfds.load('iris', split=split, as_supervised=True)
    dataset = dataset.map(lambda features, labels: ({'dense_input': features}, labels))
    dataset = dataset.batch(32).repeat()
    return dataset


for features_batch, labels_batch in input_fn().take(1):
    print(features_batch)
    print(labels_batch)

# Create an Estimator from the tf.keras model
model_dir = "/tmp/tfkeras_example/"
keras_estimator = tf.keras.estimator.model_to_estimator(
    keras_model=model, model_dir=model_dir)

keras_estimator.train(input_fn=input_fn, steps=25)
eval_result = keras_estimator.evaluate(input_fn=input_fn, steps=10)
print('Eval result: {}'.format(eval_result))
