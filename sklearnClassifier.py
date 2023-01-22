# Author: Jacob Dawson
#
# Goal: play with the dataset using some simpler machine learning algorithms,
# to get a feel for the problem, before we make a neural net solution.
#
# Upshot: ...I'm beginning to see that this problem lends itself moreso to
# neural network solutions than to simpler ones like this.

"""
from constants import *

# might need these for file manipulation:
#import os
#import sys
#from pathlib import Path

# dataset manipulation:
#import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# math:
import pandas as pd
import numpy as np

# we need algorithms that can do mutlilabel classification.
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC # maybe set multi_class="crammer_singer"?

df = pd.read_csv(trainingCsv)
df_train, df_test = train_test_split(
    df,
    test_size=.9,
    random_state=seed,
    shuffle=True,
)

X_imgs = []
ids = df_train.values[:,0].astype(int).astype(str)
y_batch = df_train.values[:,1:]
for id in ids:
    img = cv2.imread(dataDirectory+id+'.jpg',cv2.IMREAD_COLOR)
    img = cv2.resize(img,(53,53)) # the largest factor of 424 is 53
    X_imgs.append(img)
X_imgs = np.array(X_imgs)

# this code will save our current first image. Let's see what we're working on
'''
figure = plt.figure(figsize=(8,8))
plt.xlabel(imgs[0].shape)
#plt.imshow(imgs[0])
plt.savefig('example.png', dpi=100)
'''

# reshape the X data into a format that sklearn is ok with, must be 2-dims:
X_set = []
for img in X_imgs:
    X_set.append(img.flatten()) # pains me to do this!
X_set = np.array(X_set)

# for simplicity, let's discretize our data:
y_batch = np.rint(y_batch) # round to nearest int
y_batch = LabelBinarizer().fit_transform(y_batch)
print(y_batch)

classifier = MultiOutputClassifier(
    estimator = LinearSVC(
        max_iter=100,
        class_weight='balanced',
        random_state=seed,
    ),
    n_jobs=1,
)
classifier.fit(X=X_set, Y=y_batch) # need to manipulate y_batch?
print(classifier.predict(X_set))
"""
