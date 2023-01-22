# Author: Jacob Dawson
#
# In this file, we will specify the architecture of our neural network
# classifier. I'm thinking that, at least to begin with, we'll be using a
# pretty standard convolutional neural network
import tensorflow as tf
from tensorflow import keras
from constants import *

# conv, batchnorm, and activation
def layer(input, filters, size, stride, apply_batchnorm=True):
    out = keras.layers.Conv2D(filters, kernel_size=size, strides=stride, padding='same')(input)
    if apply_batchnorm:
        out = keras.layers.BatchNormalization()(out)
    return keras.layers.Activation('selu')(out)

# reduce the size of the image by half and only give a depth of filters
def downsample(input, filters, apply_batchnorm=True):
    out = keras.layers.Conv2D(filters, kernel_size=4, strides=2, padding='same')(input)
    if apply_batchnorm:
        out = keras.layers.BatchNormalization()(out)
    return keras.layers.Activation('selu')(out)

# link the above layers into a dense block!
def denseBlock(input, filters, size, apply_batchnorm=True):
    # how many layers do we want per dense block?
    l1 = layer(input, filters=filters, size=size, stride=1, apply_batchnorm=apply_batchnorm)
    l2 = layer(keras.layers.Concatenate()([input,l1]), filters=filters, size=size, stride=1, apply_batchnorm=apply_batchnorm)
    #l3 = layer(keras.layers.Concatenate()([input,l1,l2]), filters=filters, size=size, stride=1, apply_batchnorm=apply_batchnorm)
    #l4 = layer(keras.layers.Concatenate()([input,l1,l2,l3]), filters=filters, size=size, stride=1, apply_batchnorm=apply_batchnorm)
    #l5 = layer(keras.layers.Concatenate()([input,l1,l2,l3,l4]), filters=filters, size=size, stride=1, apply_batchnorm=apply_batchnorm)

    return downsample(l2, filters=filters, apply_batchnorm=apply_batchnorm)

def convNet():
    input = keras.layers.Input(shape=(image_size,image_size,num_channels), dtype=tf.float16)
    scale = keras.layers.Rescaling(1.0/255.0, offset=0)(input)
    out = denseBlock(scale, filters=4, size=3, apply_batchnorm=False)
    out = denseBlock(out, filters=8, size=3, apply_batchnorm=True)
    out = denseBlock(out, filters=16, size=3, apply_batchnorm=True)
    out = denseBlock(out, filters=32, size=3, apply_batchnorm=True)
    out = denseBlock(out, filters=64, size=3, apply_batchnorm=True)
    #out = keras.layers.Flatten()(out)
    out = keras.layers.GlobalAveragePooling2D()(out)
    out = keras.layers.Dense(output_options,activation=None)(out) # what activation should we use here?
    return keras.Model(inputs=input, outputs=out, name='classifier')

if __name__ == '__main__':
    network = convNet()
    network.summary()
    # this requires graphviz, the bane of my existence
    #keras.utils.plot_model(network, to_file='network_plot.png', show_shapes=True, show_layer_names=False, show_layer_activations=True, expand_nested=True)
