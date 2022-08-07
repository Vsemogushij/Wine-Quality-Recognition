import tensorflow as tf
import pandas as pd
import numpy as np

# Reading data from csv
wine_train = pd.read_csv('winequality-red.csv', skiprows=[0], names=['fixed acidity',
                                                                     'volatile acidity',
                                                                     'citric acid',
                                                                     'residual sugar',
                                                                     'chlorides',
                                                                     'free sulfur dioxide',
                                                                     'total sulfur dioxide',
                                                                     'density',
                                                                     'pH',
                                                                     'sulphates',
                                                                     'alcohol',
                                                                     'quality'])

# Getting qualities from whole table and normalizing it
wine_qualities = wine_train.pop('quality') * 0.1
# Copying other wine features
wine_features = wine_train.copy()

wine_features = np.array(wine_features)

model = tf.keras.models.Sequential([
    tf.keras.layers.Normalization(),
    tf.keras.layers.Dense(32),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam())

model.fit(wine_features, wine_qualities, epochs=100)

print(model.evaluate(wine_features[200:205], wine_qualities[200:205], verbose=2))
