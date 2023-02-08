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
#import matplotlib.pyplot as plt

autotune = tf.data.experimental.AUTOTUNE
tf.random.set_seed(seed)
keras.mixed_precision.set_global_policy('mixed_float16')

ds, _, x_train_shape, x_test_shape = preprocess(
    trainingCsv,
    given_test_size=0.0
)

network = convNet()
network.built=True
network.load_weights(checkpointFolder + 'ckpt40')

'''
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
'''

metrics = network.evaluate(
    ds,
    #batch_size=None, # data already batched
    return_dict=True,
)

print(metrics)
