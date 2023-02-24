# galaxyZookeeper
Found a "galaxy zoo" dataset on kaggle, thought I'd play around with it! The dataset can be found here: https://www.kaggle.com/competitions/galaxy-zoo-the-galaxy-challenge/, it's around 3 gigs of 424x424 jpgs of galaxies, ~60,000 images in total. The original intent is to allow citizen scientists to aid astronomers in classifying these--with the advent of deep learning, we can try and do this a bit faster than the mind-numbing task of by-hand classification.

I'd like to note that I have the following organization of files:  
|galaxyZookeeper--> README.md (this file)  
|-----------------> (all the other files of this repo)  
|-----------------> galaxy-zoo-the-galaxy-challenge---> all_ones_benchmark--------> all_ones_benchmark.csv  
|-----------------------------------------------------> all_zeros_benchmark-------> all_zeros_benchmark.csv  
|-----------------------------------------------------> central_pixel_benchmark---> central_pixel_benchmark.csv  
|-----------------------------------------------------> images_test_rev1----------> _tons of jpgs_  
|-----------------------------------------------------> images_training_rev1------> _tons of jpgs_  
|-----------------------------------------------------> training_solutions_rev1---> training_solutions_rev1.csv  

# Results
Currently, I'm still tinkering around with the model, but my "first shot" trained for 40 epochs and it looks like we're getting somewhere around 50% accuracy or 0.04 MSE loss on some validation data. Not terrible, considering the number of classes we're attempting to classify on!
