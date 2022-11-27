# Define scope of experiment
n_splits= 20
epochs = 100
patience = 40 # for early stopping technique
save_best_model = True
use_pretrained_model = False
test_percentage_of_data = 0.20
model_type= 4

# Saving experiment results
experiment_folder= f'results\\ensemble_methods_model_type_{model_type}_v2'
# saving best models names
models_dir="models/"
best_model_name_random_split= f"{models_dir}best_random_splits_model"
best_model_name_bootstrap= f"{models_dir}best_bootstrap_model"
best_model_name_kfold_cross_val= f"{models_dir}best_kfold_cross_val_model"


# Plot titles
fig_title_bagging_ensemble= f'Bagging Ensemble, Epochs:{epochs}, splits:{n_splits}'
title_kfold_cross_validation = f'{n_splits}-Fold Cross Validation Ensemble, Epochs:{epochs}'
title_random_splits = f'Random Splits Ensemble, Epochs:{epochs}, Test size:{test_percentage_of_data * 100}%'