import  keras.optimizers as optim
import keras
from keras import Sequential
from keras.layers import Dense, BatchNormalization
from data_utilities.sre_dataset_utilities import get_transformed_data
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

x_train, y_train, x_test, y_test = get_transformed_data(0)

batch_size = 128
epochs_to_train = 300
epochs_to_tune = 10

def model_builder(hp):
  model = Sequential()
  model.add(Dense(15, input_dim=40, activation='relu'))
  model.add(BatchNormalization())
  hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
  model.add(Dense(units=hp_units, activation='relu'))

  model.add(Dense(5, activation='softmax'))
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
  model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), metrics=['accuracy'])
  return model


# Perform hypertuning
tuner = kt.Hyperband(model_builder, objective='val_accuracy', max_epochs=10,  factor=3, directory='my_dir',  project_name='intro_to_kt')
print(tuner.search_space_summary( ))
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
tuner.search(x_train, y_train, epochs=epochs_to_tune, validation_split=0.2, callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f'best hyper parameters: {best_hps.values}')
print(f""" The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

# # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
h_model = tuner.hypermodel.build(best_hps)
h_model.summary()
best_model_name = "hypertuned_model"

file_name_loss = 'results/hyperparameter_tuned_model_loss'
file_name_acc = 'results/hyperparameter_tuned_model_acc'
best_lr  = best_hps.get('learning_rate')
hidden_layers  = best_hps.get('units')
# title_for_loss_plot = f'Train acc:{train_acc:.3f},Test acc:{test_acc:.3f},lr:{best_lr}, hidden layer neurons:{hidden_layers}, Epochs:{training_callbacks[0].stopped_epoch}'
# title_for_acc_plot = f'lr:{best_lr}, hidden layer neurons:{hidden_layers}, Epochs:{training_callbacks[0].stopped_epoch}'

