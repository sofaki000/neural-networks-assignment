from sklearn.datasets import make_blobs
from numpy import mean
from numpy import std
from Keras.ensemble_methods import config
from Keras.ensemble_methods.ensemble_methods_utilities import get_one_bootstraped_sample, evaluate_multiple_members, train_and_evaluate_new_models_for_ensemble_method, \
    split_data_randomly, plot_single_member_scores_vs_ensemble_members, \
    train_and_evaluate_new_models_for_kfold_cross_validation
import os

os.makedirs(config.experiment_folder, exist_ok=True)

# generate 2d classification dataset
dataX, datay = make_blobs(n_samples=55000, centers=3, n_features=2, cluster_std=2, random_state=2)
X, newX = dataX[:5000, :], dataX[5000:, :]
y, newy = datay[:5000], datay[5000:]


# K-fold cross validation
scores_from_cross_validation, cross_val_models = train_and_evaluate_new_models_for_kfold_cross_validation(X, y,
                                                                                                          best_model_name=config.best_model_name_kfold_cross_val,
                                                                                                          n_folds=config.n_splits)
# summarize expected performance
print('Estimated Accuracy using Cross-Validation Ensemble %.3f (%.3f)' % (mean(scores_from_cross_validation),
                                                                          std(scores_from_cross_validation)))
# evaluate different numbers of ensembles on hold out set
single_scores_cross_validation, ensemble_scores_cross_validation = evaluate_multiple_members(config.n_splits, cross_val_models, newX, newy)

# plotting the results
filename = f'{config.experiment_folder}/{config.n_splits}_fold_cross_validation'
plot_single_member_scores_vs_ensemble_members(filename,
                                              config.title_kfold_cross_validation,
                                              config.n_splits,
                                              single_scores_cross_validation,
                                              ensemble_scores_cross_validation)

# Bagging ensemble
# multiple train-test splits
scores_from_bootstraped_models, bootstrapped_models = train_and_evaluate_new_models_for_ensemble_method(X,
                                                                                                        y,
                                                                                                        n_splits =config.n_splits,
                                                                                                        best_model_name=config.best_model_name_bootstrap,
                                                                                                        method=get_one_bootstraped_sample)
# summarize expected performance
print('Estimated Accuracy using Bagging Ensemble %.3f (%.3f)' % (mean(scores_from_bootstraped_models),
                                                                 std(scores_from_bootstraped_models)))
# evaluate different numbers of ensembles on hold out set
single_scores_bootstraped, ensemble_scores_bootstraped =  evaluate_multiple_members(config.n_splits,
                                                                                    bootstrapped_models,
                                                                                    newX,
                                                                                    newy)
# plotting the results
filename= f'{config.experiment_folder}/Bagging_ensemble_scores'
plot_single_member_scores_vs_ensemble_members(filename,config.fig_title_bagging_ensemble,
                                              config.n_splits,single_scores_bootstraped,
                                              ensemble_scores_bootstraped)

# Random Splits Ensemble
scores_from_random_splits, models_from_random_splits = train_and_evaluate_new_models_for_ensemble_method(X, y,
                                                                                                         n_splits =config.n_splits,
                                                                                                         best_model_name=config.best_model_name_random_split,
                                                                                                         method=split_data_randomly)
# summarize expected performance
print('Estimated Accuracy using Random Splits Ensemble %.3f (%.3f)' % (mean(scores_from_random_splits), std(scores_from_random_splits)))
# evaluate different numbers of ensembles on hold out set
single_scores_rs, ensemble_scores_rs = evaluate_multiple_members(config.n_splits, models_from_random_splits, newX, newy)

# plotting the results
filename = f'{config.experiment_folder}/Random_splits_scores'
plot_single_member_scores_vs_ensemble_members(filename, config.title_random_splits, config.n_splits, single_scores_rs, ensemble_scores_rs)


