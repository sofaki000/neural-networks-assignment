
batch_size = 128
# 4 is for whole dataset
load_dataset_number = 4
output_classes = 7

# 1 is for 5 samples, to see if program runs
# load_dataset_number = 1
# output_classes = 5

num_classes = output_classes
epochs = 100
lr = 0.01
weight_decay = 0.01
dropout =0.2
use_pca = False

import os
experiment_folder= 'results\\v2\\whole_dataset'
os.makedirs(experiment_folder, exist_ok=True)

#### experiments filenames and plot titles

################### EARLY STOPPING ############################
file_name_loss_plain_early_stopping = f'{experiment_folder}/early_stopping_loss'
file_name_acc_plain_early_stopping = f'{experiment_folder}/early_stopping_acc'
title_loss_plain_early_stopping='Early stopping loss'
title_acc_plain_early_stopping='Early stopping acc'

##################### BATCH NORM AND WEIGHT DECAY #########################################
file_name_loss_batch_norm_weight_dec = f'{experiment_folder}/early_stopping_with_batch_norm_loss_v1'
file_name_acc_batch_norm_weight_dec = f'{experiment_folder}/early_stopping_with_batch_norm_acc_v1'
title_loss_batch_norm_weight_dec= f'Batch norm and weight decay={weight_decay}, loss'
title_acc_batch_norm_weight_dec= f'Batch norm and weight decay={weight_decay}, acc'


###################### DROPOUT #####################################
file_name_loss_dropout = f'{experiment_folder}/early_stopping_with_dropout_in_hidden_lrs_loss_v1'
file_name_acc_dropout  = f'{experiment_folder}/early_stopping_with_dropout_in_hidden_lrs_acc_v1'
title_loss_dropout =f'Dropout={dropout}, loss'
title_acc_dropout =f'Dropout={dropout}, acc'

##################### WEIGHT DECAY ###########################
file_name_loss_weight_decay = f'{experiment_folder}/early_stopping_with_weight_decay_loss'
file_name_acc_weight_decay = f'{experiment_folder}/early_stopping_with_weight_decay_acc'
title_loss_weight_decay= f'Weight decay={weight_decay}, loss'
title_acc_weight_decay= f'Weight decay={weight_decay}, acc'

################## WEIGHT DECAY AND INITIALIZED WEIGHTS ######################
file_name_loss_weight_decay_and_initialized_weights = f'{experiment_folder}/early_stopping_with_weight_decay_and_initialized_weights_loss'
file_name_acc_weight_decay_and_initialized_weights =f'{experiment_folder}/early_stopping_with_weight_decay_and_initialized_weights_acc'
title_loss_weight_decay_and_initialized_weights='Initialized weights loss'
title_acc_weight_decay_and_initialized_weights='Initialized weights acc'

##################### L1 ############
file_name_loss_l1 = f'{experiment_folder}/early_stopping_with_l1_regularizer_loss'
file_name_acc_l1 = f'{experiment_folder}/early_stopping_with_l1_regularizer_acc'
title_loss_l1='L1 regularizer loss'
title_acc_l1='L1 regularizer acc'