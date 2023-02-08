# Author: Jacob Dawson
#
# This code used to be a part of tfClassifier.py, but I figured that it might
# be useful to repackage it so that other files could do the same thing that we
# do here.

###############################################################################
# imports
from constants import *
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf

autotune = tf.data.experimental.AUTOTUNE

###############################################################################
# load and preprocess:
# I found an example of some good preprocessing code here:
# https://www.kaggle.com/code/hironobukawaguchi/galaxy-zoo-xception
def preprocess_image(image, augment_flag=False):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_with_crop_or_pad(image, image_size, image_size)
    if augment_flag:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
    image /= 255  # to [0,1] range

    return image
def load_and_preprocess_image(path):
    img_path = dataDirectory + path + '.jpg'
    image = tf.io.read_file(img_path)
    return preprocess_image(image, augment_flag=augmentFlag)

def preprocess(csv, given_test_size=0.2):
    df = pd.read_csv(csv)

    (x_train, x_test, y_train, y_test) = train_test_split(
        df.values[:,0].astype(int).astype(str),
        df.values[:,1:],
        test_size=given_test_size,
        random_state=seed
    )
    # x_train is a list of ids, y_train is the list of target predictions

    path_ds = tf.data.Dataset.from_tensor_slices(x_train)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=autotune)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(y_train, tf.float32))
    ds = tf.data.Dataset.zip((image_ds, label_ds))
    ds = ds.apply(
        tf.data.experimental.shuffle_and_repeat(buffer_size=bufferSize)
    )
    ds = ds.batch(batch_size)
    path_ds_valid = tf.data.Dataset.from_tensor_slices(x_test)
    image_ds_valid = path_ds_valid.map(load_and_preprocess_image, num_parallel_calls=autotune)
    label_ds_valid = tf.data.Dataset.from_tensor_slices(tf.cast(y_test, tf.float32))
    ds_valid = tf.data.Dataset.zip((image_ds_valid, label_ds_valid))
    ds_valid = ds_valid.batch(batch_size)

    return ds, ds_valid, x_train.shape, x_test.shape
