from time import strftime, gmtime

from matplotlib import pyplot as plt


def save_multiple_plots_for_different_experiments(experiments_mean_losses, titles, file_name):
    file_name = strftime(f"{file_name}Date%Y_%m_%d_Time%H_%M_%S", gmtime())
    figure, axis = plt.subplots(2, 2,figsize=(7,7), gridspec_kw={
                           'width_ratios': [3, 3],
                           'height_ratios': [3, 3], 'wspace': 0.4,  'hspace': 0.4})
    k = len(experiments_mean_losses) - 1
    figure.tight_layout()
    for i in range(2):
        experiment_mean_losses = experiments_mean_losses[i]
        title = titles[i]
        axis[i, 0].plot(experiment_mean_losses)
        axis[i, 0].set_title(title)
        experiment_mean_losses = experiments_mean_losses[k - i]
        title = titles[k - i]
        axis[i, 1].plot(experiment_mean_losses)
        axis[i, 1].set_title(title)

    figure.savefig(f'{file_name}.png')



def save_model_train_and_test_metric(metric, train_metric , test_metric, title, file_name):
    file_name = strftime(f"{file_name}Date%Y_%m_%d_Time%H_%M_%S", gmtime())
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.suptitle(title, fontsize=10)
    plt.xlabel('Epochs', fontsize=10)
    if metric=='loss':
        plt.ylabel('Losses', fontsize=10)
        ax.plot(train_metric, color='r', label='Train loss')
        ax.plot(test_metric, color='g', label='Test loss')
    elif metric=='acc':
        plt.ylabel('Accuracy', fontsize=10)
        ax.plot(train_metric, color='r', label='Train accuracy')
        ax.plot(test_metric, color='g', label='Test accuracy')
    plt.legend()
    fig.savefig(file_name)

def save_model_train_and_test_loss_plot(train_losses, test_losses, title, file_name):
    save_model_train_and_test_metric('loss', train_losses , test_losses, title, file_name)
def save_model_train_and_test_accuracy_plot(train_losses, test_losses, title, file_name):
    save_model_train_and_test_metric('acc',train_losses , test_losses, title, file_name)

def save_multiple_plots_for_two_experiments(experiments_mean_losses, titles, file_name):
    file_name = strftime(f"{file_name}Date%Y_%m_%d_Time%H_%M_%S", gmtime())
    figure, axis = plt.subplots(2, sharex=True ,figsize=(7,7), gridspec_kw={
                           'width_ratios': [3], 'height_ratios': [3,2], 'wspace': 0.4,  'hspace': 0.4})
    k = len(experiments_mean_losses) - 1
    # figure.tight_layout()
    figure.suptitle('Losses over epochs')
    for i in range(2):
        experiment_mean_losses = experiments_mean_losses[i]
        title = titles[i]
        axis[i].plot(experiment_mean_losses)
        axis[i].set_title(title)
    figure.savefig(f'{file_name}.png')
