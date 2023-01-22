# Author: Jacob Dawson
#
# Goal: make a nn to classify the images of the galaxy zoo dataset!

###############################################################################
# imports and constants
import pandas as pd
import numpy as np
from constants import *
from architecture import *
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

autotune = tf.data.experimental.AUTOTUNE
tf.random.set_seed(seed)

###############################################################################
# load and preprocess:

df = pd.read_csv(trainingCsv)

(x_train, x_test, y_train, y_test) = train_test_split(df.values[:,0].astype(int).astype(str), df.values[:,1:], test_size=0.2, random_state=0)
# x_train is a list of ids, y_train is the list of target predictions

# I found an example of some good preprocessing code here:
# https://www.kaggle.com/code/hironobukawaguchi/galaxy-zoo-xception
def preprocess_image(image, augment_flag=False):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_with_crop_or_pad(image, image_size, image_size)
    if augment_flag:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
    #image /= 255  # to [0,1] range

    return image
def load_and_preprocess_image(path):
    img_path = dataDirectory + path + '.jpg'
    image = tf.io.read_file(img_path)
    return preprocess_image(image, augment_flag=augmentFlag)

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

###############################################################################
# Model time!
network = convNet()
network.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learnRate),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=['acc'],
    #run_eagerly=True,
)

class EveryKCallback(keras.callbacks.Callback):
    def __init__(self,epoch_interval=5):
        self.epoch_interval = epoch_interval
    def on_epoch_begin(self,epoch,logs=None):
        if ((epoch % self.epoch_interval)==0):
            self.model.save_weights("ckpts/ckpt"+str(epoch), overwrite=True, save_format='h5')
            #self.model.save('network',overwrite=True)

history = network.fit(
    ds,
    epochs=epochs,
    verbose=1,
    callbacks=[EveryKCallback(epoch_interval=5)], # custom callbacks here!
    shuffle=False, # shuffling done via dataset api,
    steps_per_epoch=x_train.shape[0]//batch_size,
    #use_multiprocessing=True,
    #workers=8, 
    validation_steps=x_test.shape[0]//batch_size,
    validation_data=ds_valid
)

network.save_weights("ckpts/finished", overwrite=True, save_format='h5')
network.save('network',overwrite=True)

def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('trainingHistory.png', dpi=100)
    plt.show()

visualize_loss(history, "Training and Validation Loss")
