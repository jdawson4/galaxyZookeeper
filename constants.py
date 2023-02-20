# Author: Jacob Dawson
#
# Consts we're using in places
seed = 7
dataDirectory = "galaxy-zoo-the-galaxy-challenge/images_training_rev1/"
trainingCsv = "galaxy-zoo-the-galaxy-challenge/training_solutions_rev1/training_solutions_rev1.csv"
checkpointFolder = "ckpts/"
batch_size = 64
image_size = 424 # natively 424
augmentFlag = False
bufferSize = 1000
num_channels = 3
output_options = 37
learnRate = 0.002 # adam's default is 0.001
epochs = 100
epoch_interval = 5
