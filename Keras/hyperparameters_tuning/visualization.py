import numpy as np
import keras_tuner
from keras.layers import Flatten, Conv2D, Dense, MaxPooling2D, Dropout
from tensorflow import keras
import tensorflow as tf
from data_utilities.sre_dataset_utilities import get_transformed_data
from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/sre_experiment1')

x_train, y_train, x_test, y_test = get_transformed_data(3)
output_classes = 7
# Print the shapes of the data.
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
def build_model(hp):
    inputs = keras.Input(shape=(40,))
    # Model type can be MLP or CNN.
    model_type = hp.Choice("model_type", ["mlp", "cnn"])
    x = inputs
    if model_type ==  True: #"mlp":
        x = Flatten()(x)
        # Number of layers of the MLP is a hyperparameter.
        for i in range(hp.Int("mlp_layers", 1, 3)):
            # Number of units of each layer are
            # different hyperparameters with different names.
            x = Dense( units=hp.Int(f"units_{i}", 32, 128, step=32), activation="relu")(x)

            # def model_builder(hp):
            #     model = Sequential()
            #     model.add(Dense(15, input_dim=40, activation='relu'))
            #     model.add(BatchNormalization())
            #     hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
            #     model.add(Dense(units=hp_units, activation='relu'))
            #
            #     model.add(Dense(5, activation='softmax'))
            #     hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
            #     model.compile(loss='categorical_crossentropy',  optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), metrics=['accuracy'])
            #     return model

    # else:
    #     # Number of layers of the CNN is also a hyperparameter.
    #     x = tf.expand_dims(x, axis=-1)
    #     x = tf.expand_dims(x, axis=-1)
    #     x = tf.expand_dims(x, axis=-1)
    #     x = x[0]
    #     for i in range(hp.Int("cnn_layers", 1, 3)):
    #         x = Conv2D( hp.Int(f"filters_{i}", 32, 128, step=32),
    #             kernel_size=(3, 3),
    #             activation="relu",
    #         )(x)
    #         x = MaxPooling2D(pool_size=(2, 2))(x)
    #     x = Flatten()(x)

    # A hyperparamter for whether to use dropout layer.
    if hp.Boolean("dropout"):
        x = Dropout(0.5)(x)

    outputs = Dense(units=output_classes, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    # Compile the model.
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam")
    return model


# Initialize the `HyperParameters` and set the values.
hp = keras_tuner.HyperParameters()
hp.values["model_type"] = "cnn"
# Build the model using the `HyperParameters`.
model = build_model(hp)
# Test if the model runs with our data.
model(x_train[:100])
# Print a summary of the model.
model.summary()

# # Do the same for MLP model.
# hp.values["model_type"] = "mlp"
# model = build_model(hp)
# model(x_train[:100])
# model.summary()

tuner = keras_tuner.RandomSearch(  build_model,  max_trials=10, overwrite=True,   objective="val_accuracy",  directory="/tmp/tb")

tuner.search( x_train, y_train, validation_split=0.2, epochs=2 , callbacks=[keras.callbacks.TensorBoard(log_dir="/tmp/tb", profile_batch=0)])