import numpy as np
import keras_tuner
from keras.layers import Flatten, Dense, Dropout, Conv1D, MaxPooling1D
from sklearn.decomposition import PCA
from tensorflow import keras
import tensorflow as tf
from Keras.hyperparameters_tuning.hyperparameters_tuning_utilities import perform_random_search_on_model, \
    perform_hyperband_tuning_on_model, \
    get_model_with_best_hyperparameters, train_and_save_results, get_model_with_default_config
from data_utilities.sre_dataset_utilities import get_transformed_data

experiments_folder = 'models_comparison'

x_train, y_train, x_test, y_test = get_transformed_data(4)
output_classes = 7
use_pca = True

if use_pca:
    pca = PCA(0.9)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)
    input_features = 16
else:
    input_features = 40

def build_model(hp):
    inputs = keras.Input(shape=(input_features,))
    # Model type can be MLP or CNN.
    model_type = hp.Choice("model_type", ["mlp", "cnn"])
    x = inputs
    if model_type == "mlp":
        x = Flatten()(x)
        # Number of layers of the MLP is a hyperparameter.
        for i in range(hp.Int("mlp_layers", 1, 3)):
            x = Dense( units=hp.Int(f"units_{i}", 32, 128, step=32), activation="relu")(x)
    else:
        x = tf.expand_dims(x, axis=-1)
        for i in range(hp.Int("cnn_layers", 1, 3)):
            x = Conv1D(hp.Int(f"filters_{i}", 32, 128, step=32), kernel_size=1,  activation="relu")(x)
            x = MaxPooling1D(pool_size=2)(x)
        x = Flatten()(x)

    # A hyperparamter for whether to use dropout layer.
    if hp.Boolean("dropout"):
        x = Dropout(0.5)(x)

    outputs = Dense(units=output_classes, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=keras.optimizers.Adam(learning_rate=1e-3))
    return model


# Initialize the `HyperParameters` and set the values.
hp = keras_tuner.HyperParameters()
hp.values["model_type"] = "cnn"
# Build the model using the `HyperParameters`.
model = build_model(hp)
# Print a summary of the model.
model.summary()

# Do the same for MLP model.
hp.values["model_type"] = "mlp"
model = build_model(hp)
model.summary()

train_size = (x_train.shape[0])
test_size = (x_test.shape[0])

##################### Random search algorithm #####################
random_search_tuner = perform_random_search_on_model(build_model,x_train, y_train,x_test, y_test)
model_from_random_search, best_hp_rs = get_model_with_best_hyperparameters(random_search_tuner)
# getting hyperparameters for printing
best_lr  = 1e-3# best_hp_rs.get('learning_rate')
model_type  = best_hp_rs.get('model_type')

best_model_name_rs = "models/best_model_from_random_search"
file_name_loss = f'{experiments_folder}/losses_random_search_model'
file_name_acc = f'{experiments_folder}/acc_random_search_model'
title_for_loss_plot =f'Best model from random search:lr:{best_lr}, model type:{model_type}, train size:{train_size}, test size:{test_size}'
title_for_acc_plot =f'Best model from random search:lr:{best_lr}, model type:{model_type}, train size:{train_size}, test size:{test_size}'
train_and_save_results(model_from_random_search, best_model_name_rs,file_name_loss, file_name_acc, title_for_loss_plot, title_for_acc_plot,x_train, y_train, x_test, y_test )

print('-- done from random search --')
##################### Hyperband algorithm #####################
hyperband_tuner = perform_hyperband_tuning_on_model(build_model,x_train, y_train,x_test, y_test)
model_from_hyperband ,best_hp_hb= get_model_with_best_hyperparameters(hyperband_tuner)


# getting hyperparameters for printing
best_lr_hb  = 1e-3#best_hp_hb.get('learning_rate')
model_type_hb  = best_hp_hb.get('model_type')
title_for_loss_plot_hb =f'Best model from Hyperband:lr:{best_lr_hb}, model type:{model_type_hb}, train size:{train_size}, test size:{test_size}'
title_for_acc_plot_hb =f'Best model from Hyperband:lr:{best_lr_hb}, model type:{model_type_hb}, train size:{train_size}, test size:{test_size}'

best_model_name_hb = "models/best_model_from_hyperband"
file_name_loss_hb = f'{experiments_folder}/losses_hyperband_model'
file_name_acc_hb = f'{experiments_folder}/acc_hyperband_model'
train_and_save_results(model_from_hyperband, best_model_name_hb,file_name_loss_hb, file_name_acc_hb, title_for_loss_plot_hb, title_for_acc_plot_hb,x_train, y_train, x_test, y_test )


print('-- done from hyperband --')
########## default config ##############

default_model = get_model_with_default_config(input_size=input_features, output_classes=output_classes)
lr = 1e-3
title_for_loss_default_model =f'lr:{lr}, train size:{train_size}, test size:{test_size}'
title_for_acc_plot_default_model =f'lr:{lr}, train size:{train_size}, test size:{test_size}'

default_model_name= "models/default_model"
file_name_loss_default_model= f'{experiments_folder}/losses_default_model'
file_name_acc_default_model= f'{experiments_folder}/acc_default_model'
train_and_save_results(default_model, default_model_name,file_name_loss_default_model, file_name_acc_default_model, title_for_loss_default_model, title_for_acc_plot_default_model,x_train, y_train, x_test, y_test )

