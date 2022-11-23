import  keras.optimizers as optim
import keras
from keras import Sequential
from keras.layers import Dense, BatchNormalization
from data_utilities.sre_dataset_utilities import get_transformed_data
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

from utilities.plot_utilities import save_model_train_and_test_loss_plot
from utilities.train_utilities import get_callbacks_for_training

x_train, y_train, x_test, y_test = get_transformed_data(3)
batch_size = 128
epochs = 300

def model_builder(hp):
  model = Sequential()
  model.add(Dense(15, input_dim=40, activation='relu'))
  model.add(BatchNormalization())
  hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
  model.add(Dense(units=hp_units, activation='relu'))

  model.add(Dense(7, activation='softmax'))
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
  model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), metrics=['accuracy'])

  return model

# Instantiate the tuner
tuner = kt.Hyperband(model_builder, objective='val_accuracy', max_epochs=10, factor=3, directory='dir', project_name='intro_to_kt')

print(tuner.search_space_summary( ))
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Perform hypertuning
epochs_to_tune = 10
tuner.search(x_train, y_train, epochs=epochs_to_tune, validation_split=0.2, callbacks=[stop_early])
best_hp=tuner.get_best_hyperparameters()[0]
print(f'best hyper parameters: {best_hp.values}')

tuner = kt.Hyperband(model_builder, objective='val_accuracy', max_epochs=10,  factor=3, directory='my_dir',  project_name='intro_to_kt')
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
tuner.search(x_train, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
#
print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

# # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
h_model = tuner.hypermodel.build(best_hps)
h_model.summary()

# train model
history = h_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test), callbacks=get_callbacks_for_training("hypertuned_model"))

# evaluate the model
_, train_acc = h_model.evaluate(x_train, y_train, verbose=0)
_, test_acc = h_model.evaluate(x_test , y_test.astype("float32"))

print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
best_lr  = best_hps.get('learning_rate')
hidden_layers  = best_hps.get('units')
title = f'Train acc:{train_acc:.3f},Test acc:{test_acc:.3f},lr:{best_lr}, hidden layer neurons:{hidden_layers}'
file_name = 'results/hyperparameter_tuned_model'
save_model_train_and_test_loss_plot(history.history['loss'], history.history['val_loss'], title, file_name)

# model = tuner.hypermodel.build(best_hps)
# history = model.fit(x_train, y_train, epochs=50, validation_split=0.2)
# #
# val_acc_per_epoch = history.history['val_accuracy']
# best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
# print('Best epoch: %d' % (best_epoch,))
# hypermodel = tuner.hypermodel.build(best_hps)
# # Retrain the model
# hypermodel.fit(x_train, y_train, epochs=best_epoch, validation_split=0.2)