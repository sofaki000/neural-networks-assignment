# import torch
#
# print(f"Is CUDA supported by this system?, {torch.cuda.is_available()} ")
# print(f"CUDA version: {torch.version.cuda}")
#
# # Storing ID of current CUDA device
# cuda_id = torch.cuda.current_device()
# print(f"ID of current CUDA device:{torch.cuda.current_device()}")
#
# print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)} ")


# Importing libraries
import keras
from keras.layers import Flatten, Dense, Dropout
import keras_tuner
import keras
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from tensorflow import keras
import keras_tuner as kt

import numpy as np
import math

# Using Numpy to create an array X
from utilities.plot_utilities import save_model_train_and_test_loss_plot

# X = np.arange(0, math.pi * 2, 0.05)
#
# # Assign variables to the y axis part of the curve
# y = np.sin(X)
# z = np.cos(X)
#
# save_model_train_and_test_loss_plot(y, z, title='test', file_name='test_file')
import numpy as np
import keras_tuner
from keras.layers import Flatten, Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow import keras
import tensorflow as tf
from Keras.hyperparameters_tuning.hyperparameters_tuning_utilities import perform_random_search_on_model, \
    perform_hyperband_tuning_on_model, \
    get_model_with_best_hyperparameters, train_and_save_results, get_model_with_default_config
from data_utilities.sre_dataset_utilities import get_transformed_data
from torch.utils.tensorboard import SummaryWriter

import os

experiment_folder= 'aaa\\what'
os.makedirs(experiment_folder, exist_ok=True)