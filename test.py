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
import matplotlib.pyplot as plt
import numpy as np
import math

# Using Numpy to create an array X
from plot_utilities import save_model_train_and_test_loss_plot

X = np.arange(0, math.pi * 2, 0.05)

# Assign variables to the y axis part of the curve
y = np.sin(X)
z = np.cos(X)

save_model_train_and_test_loss_plot(y, z, title='test', file_name='test_file')