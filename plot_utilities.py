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
