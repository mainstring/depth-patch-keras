import tensorflow as tf
import numpy as np
# from tensorflow_core.python.keras.preprocessing.image import ImageDataGenerator

from data_utils.video_separation import get_project_root
from data_utils.patch_generation import PATCH_HEIGHT, PATCH_WIDTH
from model.patch_model import get_compiled_model
import os

patch_directory = os.path.join(get_project_root(), "data", "train", "patch")
model_directory = os.path.join(get_project_root(), "model", "weight", "patch")
batch_size = 32
classes = ["fake", "real"]

if __name__ == "__main__":
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    sys_details = tf.sysconfig.get_build_info()
    cuda_version = sys_details["cuda_version"]
    print(cuda_version)

    # train_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
    # test_datagen = ImageDataGenerator(rescale=1. / 255)
    # train_generator = train_datagen.flow_from_directory(
    #     patch_directory,
    #     target_size=(PATCH_HEIGHT, PATCH_WIDTH),
    #     batch_size=batch_size,
    #     subset='training',
    #     seed=123,
    #     class_mode='binary')
    # validation_generator = train_datagen.flow_from_directory(
    #     patch_directory,
    #     target_size=(PATCH_HEIGHT, PATCH_WIDTH),
    #     batch_size=batch_size,
    #     subset='validation',
    #     seed=123,
    #     class_mode='binary')
    #
    # model = get_compiled_model(2)
    # model.fit_generator(
    #     train_generator,
    #     steps_per_epoch=2000,
    #     epochs=5,
    #     validation_data=validation_generator,
    #     validation_steps=800)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        patch_directory,
        validation_split=0.2,
        subset="training",
        seed=123,
        labels="inferred",
        class_names=classes,
        image_size=(PATCH_HEIGHT, PATCH_WIDTH),
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        patch_directory,
        validation_split=0.2,
        subset="validation",
        seed=123,
        labels="inferred",
        class_names=classes,
        image_size=(PATCH_HEIGHT, PATCH_WIDTH),
        batch_size=batch_size)

    # train_ds = train_ds.cache()
    # val_ds = val_ds.cache()
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)

    normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    normalized_validation_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    print(normalized_validation_ds)

    model = get_compiled_model(2)
    # model.predict()
    model.fit(normalized_train_ds, validation_data=normalized_validation_ds, epochs=50)
    model.save(model_directory)

    # predictions = np.array([])
    # labels = np.array([])
    # for x, y in val_ds:
    #     predictions = np.concatenate([predictions, np.argmax(model.predict(x), axis=-1)])
    #     labels = np.concatenate([labels, y.numpy()])
    #
    # print(labels)
    # tf.math.confusion_matrix(labels=labels, predictions=predictions).numpy()