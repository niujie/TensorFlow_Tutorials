# Load a pandas.DataFrame

# Read data using pandas
from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
import tensorflow as tf

# Download the csv file containing the heart dataset
csv_file = tf.keras.utils.get_file('heart.csv', 'https://storage.googleapis.com/applied-dl/heart.csv')

# Read the csv file using pandas
df = pd.read_csv(csv_file)
print(df.head())
print(df.dtypes)

df['thal'] = pd.Categorical(df['thal'])
df['thal'] = df.thal.cat.codes
print(df.head())

# Load data using tf.data.Dataset
target = df.pop('target')

dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))

for feat, targ in dataset.take(5):
    print('Features: {}, Target: {}'.format(feat, targ))

print(tf.constant(df['thal']))

# Shuffle and batch the dataset
train_dataset = dataset.shuffle(len(df)).batch(1)


# Create and train a model
def get_compiled_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


model = get_compiled_model()
model.fit(train_dataset, epochs=15)

# Alternative to feature columns
inputs = {key: tf.keras.layers.Input(shape=(), name=key) for key in df.keys()}
x = tf.stack(list(inputs.values()), axis=-1)

x = tf.keras.layers.Dense(10, activation='relu')(x)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model_func = tf.keras.Model(inputs=inputs, outputs=output)

model_func.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

dict_slices = tf.data.Dataset.from_tensor_slices((df.to_dict('list'), target.values)).batch(16)

for dict_slice in dict_slices.take(1):
    print(dict_slice)

model_func.fit(dict_slices, epochs=15)
