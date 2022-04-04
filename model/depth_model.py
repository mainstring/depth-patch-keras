from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

import os

PROJECT_ROOT_DIR = os.path.dirname(find_dotenv())

from tensorflow import keras


# import os, inspect


def get_uncompiled_model():

    pool1 = keras.Sequential()
    pool1.add(keras.layers.Conv2D(64, 3, input_shape=(128,128,3)))
    pool1.add(keras.layers.LeakyReLU())
    pool1.add(keras.layers.Conv2D(64, 3))
    pool1.add(keras.layers.LeakyReLU())
    pool1.add(keras.layers.Conv2D(128, 3))
    pool1.add(keras.layers.LeakyReLU())
    pool1.add(keras.layers.BatchNormalization())
    pool1.add(keras.layers.MaxPool2D(strides=2))

    pool2 = keras.Sequential()
    pool2.add(keras.layers.Conv2D(128, 3))
    pool2.add(keras.layers.LeakyReLU())
    pool2.add(keras.layers.Conv2D(256, 3))
    pool2.add(keras.layers.LeakyReLU())
    pool2.add(keras.layers.Conv2D(160, 3))
    pool2.add(keras.layers.LeakyReLU())
    pool2.add(keras.layers.BatchNormalization())
    pool2.add(keras.layers.MaxPool2D(strides=2))

    upsample1 = keras.Sequential()
    upsample1.add(keras.layers.Conv2D(128, 3))
    upsample1.add(keras.layers.Conv2DTranspose(128, 6))

    upsample2 = keras.Sequential()
    upsample2.add(keras.layers.Conv2D(128, 3))
    upsample2.add(keras.layers.Conv2DTranspose(128, 6))

    upsample3 = keras.Sequential()
    upsample3.add(keras.layers.Conv2D(160, 3))
    upsample3.add(keras.layers.Conv2DTranspose(160, 6))

    upsample4 = keras.Sequential()
    upsample4.add(keras.layers.Conv2D(320, 3))
    upsample4.add(keras.layers.Conv2DTranspose(320, 6))

    output = keras.Sequential()
    output.add(keras.layers.Conv2D(1, 3))

    model = keras.Sequential()
    model.add(pool1)
    model.add(pool2)
    model.add(upsample1)
    model.add(upsample2)
    model.add(upsample3)
    model.add(upsample4)
    model.add(output)
    return model


def get_compiled_model():
    model = get_uncompiled_model()
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy', 'categorical_accuracy'])
    return model


cwd = os.path.abspath(os.path.dirname(__file__))
print(cwd)
