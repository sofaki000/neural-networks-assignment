
model_name = 'models/cifar_net.pth'

n_epochs = 50
import os
lr = 0.001
batch_size = 128
experiment_title = "results/Adam/Adam_results"

experiments_folder = f'results/cnn'
os.makedirs(experiments_folder, exist_ok=True)


# experiment results

# For train loss
train_loss_plot_title = f"Train loss,Ep:{n_epochs},lr{lr}, batch:{batch_size}"
train_loss_file_name = f'{experiments_folder}/train_loss_v2'

# For train accuracy
train_acc_plot_title = f"Train accuracy,Ep:{n_epochs},lr{lr}, batch:{batch_size}"
train_acc_file_name = f'{experiments_folder}/train_accuracy_v2'

# For test loss
test_loss_plot_title = f"Test loss,Ep:{n_epochs},lr{lr}, batch:{batch_size}"
test_loss_file_name = f'{experiments_folder}/test_loss'

# For test accuracy
test_acc_plot_title = f"Test accuracy,Ep:{n_epochs},lr{lr}, batch:{batch_size}"
test_acc_file_name = f'{experiments_folder}/test_accuracy'