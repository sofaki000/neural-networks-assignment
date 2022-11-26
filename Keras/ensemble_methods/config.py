# Define scope of experiment
n_splits= 20
epochs = 100
patience = 40 # for early stopping technique
save_best_model = True
use_pretrained_model = False
test_percentage_of_data = 0.20
model_type= 2

# Saving experiment results
experiment_folder= 'results\\ensemble_methods_with_more_epochs_and_patience'
# saving best models names
models_dir="models/"
best_model_name_random_split= f"{models_dir}best_random_splits_model"
best_model_name_bootstrap= f"{models_dir}best_bootstrap_model"
best_model_name_kfold_cross_val= f"{models_dir}best_kfold_cross_val_model"


# Plot titles
fig_title_bagging_ensemble= f'Bagging Ensemble, Epochs:{epochs}, splits:{n_splits}'
title_kfold_cross_validation = f'{n_splits}-Fold Cross Validation Ensemble, Epochs:{epochs}, Pretrained-models VGG16'
title_random_splits = f'Random Splits Ensemble, Epochs:{epochs}, Test size:{test_percentage_of_data * 100}%'