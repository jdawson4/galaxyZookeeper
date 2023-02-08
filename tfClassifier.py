# Author: Jacob Dawson
#
# Goal: make a nn to classify the images of the galaxy zoo dataset!

###############################################################################
# imports and constants
#import pandas as pd
#import numpy as np
from constants import *
from architecture import *
from preprocess import preprocess
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

autotune = tf.data.experimental.AUTOTUNE
tf.random.set_seed(seed)
keras.mixed_precision.set_global_policy('mixed_float16')

# note: preprocessing done in preprocess.py. We return only what we intend to
# use here.
ds, ds_valid, x_train_shape, x_test_shape = preprocess(trainingCsv)

###############################################################################
# Model time!
network = convNet()
network.summary()
network.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learnRate),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.RootMeanSquaredError()],
    #run_eagerly=True,
)

# use this code if you have a prepared checkpoint to use for output:
checkpoint = None
if checkpoint!=None:
    network.built=True
    network.load_weights(checkpoint)
    print("Checkpoint loaded, skipping training.")

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
    callbacks=[EveryKCallback(epoch_interval=2)], # custom callbacks here!
    shuffle=False,
    steps_per_epoch=x_train_shape[0]//batch_size,
    #use_multiprocessing=True,
    #workers=8,
    validation_steps=x_test_shape[0]//batch_size,
    validation_data=ds_valid,
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
    plt.savefig('trainingHistory.png', dpi=600)
    #plt.show()

visualize_loss(history, "Training and Validation Loss")
