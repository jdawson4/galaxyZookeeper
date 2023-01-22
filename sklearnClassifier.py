# Author: Jacob Dawson
#
# Goal: play with the dataset using some simpler machine learning algorithms,
# to get a feel for the problem, before we make a neural net solution.

from constants import *

# might need these for file manipulation:
#import os
#import sys
#from pathlib import Path

# dataset manipulation:
#import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split

# math:
import pandas as pd
import numpy as np

# we need algorithms that can do mutlilabel classification.
from sklearn.svm import LinearSVC # maybe set multi_class="crammer_singer"?

df = pd.read_csv(trainingCsv)
df_train, df_test = train_test_split(df, test_size=.999) # silly test size for now

imgs = []
ids = df_train.values[:,0].astype(int).astype(str)
y_batch = df_train.values[:,1:]
for id in ids:
    img = cv2.imread(dataDirectory+id+'.jpg',cv2.IMREAD_COLOR)
    img = cv2.resize(img,(53,53)) # the largest factor of 424 is 53
    imgs.append(img)
imgs = np.array(imgs)

'''
figure = plt.figure(figsize=(8,8))
plt.xlabel(imgs[0].shape)
plt.imshow(imgs[0])
plt.savefig('myfilename.png', dpi=100)
'''

classifier = LinearSVC(
    max_iter=100,
    class_weight='balanced',
    random_state=seed
)
#classifier.fit(X=imgs, y=y_batch) # need to manipulate y_batch?
