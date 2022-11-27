
cifar_filepath = 'C:\\Users\\Lenovo\\Desktop\\νευρωνικά δίκτυα\\neural-networks-assignment\\Keras\\data'
load_development_dataset = False

import os
experiment_folder = 'experiments/newest'
os.makedirs(experiment_folder, exist_ok=True)

experiments_file_name= f"{experiment_folder}/nearest_centroid_experiments_impl.txt"
best_model_name ='nearest_centoird_model'
file_name_loss = f"{experiment_folder}/nearest_centroid_loss"
file_name_acc = f"{experiment_folder}/nearest_centroid_acc"

experiment_1_title="Experiment 1- Raw train and test data\n"
experiment_2_title="Experiment 2- Normalizing train and test data\n"
experiment_3_title="Experiment 3- Standardization of train and test data\n"
experiment_4_title="Experiment 4- Rescaled train and test data in the range [0,1]\n"
experiment_5_title="Experiment 5- Scaling features independently\n"