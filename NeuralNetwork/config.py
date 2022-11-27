import os

model_name = 'models/cifar_net_v2.pth'

n_epochs = 100
train_repeats = 30
lr = 0.001
batch_size = 256 # 128
experiments_folder = f'results/cnn_with_early_stopping/with_validation_set'
os.makedirs(experiments_folder, exist_ok=True)


# experiment results
# For train loss
train_loss_plot_title = f"Train loss,Ep:{n_epochs},lr{lr}, batch:{batch_size}"
train_loss_file_name = f'{experiments_folder}/train_loss'

# For train accuracy
train_acc_plot_title = f"Train accuracy,Ep:{n_epochs},lr{lr}, batch:{batch_size}"
train_acc_file_name = f'{experiments_folder}/train_accuracy'

# For validation loss and accuracy
validation_loss_plot_title = f"Validation loss,Ep:{n_epochs},lr{lr}, batch:{batch_size}"
validation_loss_file_name = f'{experiments_folder}/validation_loss'
validation_acc_plot_title = f"Validation accuracy,Ep:{n_epochs},lr{lr}, batch:{batch_size}"
validation_acc_file_name = f'{experiments_folder}/validation_accuracy'

# For test loss
test_loss_plot_title = f"Test loss,Ep:{n_epochs},lr{lr}, batch:{batch_size}"
test_loss_file_name = f'{experiments_folder}/test_loss'

# For test accuracy
test_acc_plot_title = f"Test accuracy,Ep:{n_epochs},lr{lr}, batch:{batch_size}"
test_acc_file_name = f'{experiments_folder}/test_accuracy'