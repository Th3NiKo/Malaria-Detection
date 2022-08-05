# Malaria detection

Detecting malaria from images using neural networks (CNNs and transfer learning).

## Dataset

Based on: <https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria>

## Getting started

In order to reuse this code You need to have tensorflow installed (I used version 2.6.0).  
Raw dataset should be placed in same folder as data_loading.py file.  
Use train_test_split() function from data_loading.py to prepare dataset for tensorflow.  
Example: >>> train_test_split("cell_images", "dataset")  
Now You can run either cnn_training.py or xception_transfer_learning.py to create model.
