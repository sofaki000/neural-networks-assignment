from keras.layers import Flatten, Dense, Dropout
import keras_tuner
import keras
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

from utilities.plot_utilities import save_model_train_and_test_loss_plot, save_model_train_and_test_accuracy_plot
from utilities.train_utilities_keras import get_callbacks_for_training


def perform_random_search_on_model(build_model,x_train, y_train,x_test, y_test):
    epochs_to_tune = 100
    max_trials = 100
    tuner = keras_tuner.RandomSearch(build_model, max_trials=max_trials, overwrite=True, objective="val_accuracy", directory="/tmp/random_rearch_logs")
    tuner.search(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs_to_tune,  callbacks=[keras.callbacks.TensorBoard(log_dir="/tmp/random_rearch_logs", profile_batch=0)])
    return tuner

def perform_hyperband_tuning_on_model(build_model,x_train, y_train,x_test, y_test):
    # Perform hypertuning
    epochs_to_tune= 100
    max_epochs = 100
    tuner = kt.Hyperband(build_model, objective='val_accuracy', overwrite=True, max_epochs=max_epochs, factor=3, directory='/tmp/hyperband_tuning' )
    print(tuner.search_space_summary())
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
    tensorboard_visualization = keras.callbacks.TensorBoard(log_dir="/tmp/hyperband_tuning", profile_batch=0)
    tuner.search(x_train, y_train, epochs=epochs_to_tune, validation_data=(x_test, y_test), callbacks=[stop_early,tensorboard_visualization])

    return tuner


def get_model_with_best_hyperparameters(tuner):
    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
    h_model = tuner.hypermodel.build(best_hps)
    h_model.summary()
    return h_model,best_hps



def train_and_save_results(model_to_train, best_model_name,file_name_loss, file_name_acc, title_for_loss_plot, title_for_acc_plot,x_train, y_train,x_test , y_test):
  # train model
  epochs_to_train = 50
  batch_size = 128
  training_callbacks = get_callbacks_for_training(best_model_name)

  stopped_at_epoch = training_callbacks[0].stopped_epoch
  if stopped_at_epoch==0:
      stopped_at_epoch= epochs_to_train

  title_for_loss_plot =  f'{title_for_loss_plot},ep:{stopped_at_epoch}'
  title_for_acc_plot = f'{title_for_acc_plot},ep:{stopped_at_epoch}'

  history = model_to_train.fit(x_train, y_train, batch_size=batch_size, epochs=epochs_to_train, verbose=1, validation_data=(x_test, y_test), callbacks=training_callbacks)
  # evaluate the model
  _, train_acc = model_to_train.evaluate(x_train, y_train, verbose=0)
  _, test_acc = model_to_train.evaluate(x_test , y_test.astype("float32"))


  val_accuracy = history.history['val_accuracy']
  accuracy = history.history['accuracy']
  val_loss = history.history['val_loss']
  loss = history.history['loss']

  # epochs_index =   [i for i in range(epochs_to_train)]
  save_model_train_and_test_loss_plot(loss, val_loss, title_for_loss_plot, file_name_loss )
  save_model_train_and_test_accuracy_plot(accuracy, val_accuracy,title_for_acc_plot, file_name_acc)




def get_model_with_default_config(input_size, output_classes):
    inputs = keras.Input(shape=(input_size,))
    x = inputs
    # x = Flatten()(x)
    x = Dense(units=32, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(units=128, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(units=output_classes, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=keras.optimizers.Adam(learning_rate=1e-3))
    print(model.summary())
    return model
