# Author: Jacob Dawson
#
# Goal: play with the dataset using some simpler machine learning algorithms,
# to get a feel for the problem, before we make a neural net solution.

# might need these for file manipulation:
#import os
#import sys
from pathlib import Path

# image stuff:
#import matplotlib.pyplot as plt
import cv2

# math:
import pandas as pd
import numpy as np

# we need algorithms that can do mutlilabel classification.
from sklearn.svm import LinearSVC # maybe set multi_class="crammer_singer"?

dataDirectory = Path("galaxy-zoo-the-galaxy-challenge/images_training_rev1")
path = list(dataDirectory.glob(r"*.jpg"))
imgs = []
for file in path:
    img = cv2.imread(file.__str__(),0)
    img = cv2.resize(img,(53,53)) # the largest factor of 424 is 53
    imgs.append(img)
    if (len(imgs) >= 100):
        break

y_data = pd.read_csv("galaxy-zoo-the-galaxy-challenge/training_solutions_rev1/training_solutions_rev1.csv")
print(y_data.axes)
