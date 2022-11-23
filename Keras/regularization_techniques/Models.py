from keras import regularizers
from keras.optimizers import RMSprop
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
import tensorflow as tf

def get_model(num_classes):
    model = Sequential()
    model.add(Dense(1024, activation='relu', input_shape=(99,)))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(),  metrics=['accuracy'])

    return model



def get_model_with_batch_normalization_and_weight_decay(num_classes, optimizer = RMSprop, lr =0.001):
    model = Sequential()
    model.add(Dense(1024,   kernel_regularizer=l2(0.01), activation='relu', input_shape=(99,)))
    model.add(BatchNormalization())
    model.add(Dense(1024,  kernel_regularizer=l2(0.01), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(512, kernel_regularizer=l2(0.01),  activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(256, kernel_regularizer=l2(0.01),  activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer= optimizer(learning_rate=lr), metrics=['accuracy'])
    return model

def get_model_with_weight_decay(num_classes):
    model = Sequential()
    model.add(Dense(1024,   kernel_regularizer=l2(0.01), activation='relu', input_shape=(99,)))
    model.add(Dense(1024,  kernel_regularizer=l2(0.01), activation='relu'))
    model.add(Dense(512, kernel_regularizer=l2(0.01),  activation='relu'))
    model.add(Dense(256, kernel_regularizer=l2(0.01),  activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    return model

def get_model_with_dropout_in_hidden_layers(num_classes):
    model = Sequential()
    model.add(Dense(1024,   kernel_regularizer=l2(0.01), activation='relu', input_shape=(99,)))
    model.add(Dropout(.2))
    model.add(Dense(1024,  kernel_regularizer=l2(0.01), activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(512, kernel_regularizer=l2(0.01),  activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(256, kernel_regularizer=l2(0.01),  activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(num_classes, activation='softmax'))

    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

    return model

def get_model_with_l1_regularizer(num_classes):
    model = Sequential()
    regularizer= regularizers.L1(0.01)
    model.add(Dense(1024,   kernel_regularizer=regularizer, activation='relu', input_shape=(99,)))
    model.add(Dense(1024,  kernel_regularizer=regularizer, activation='relu'))
    model.add(Dense(512, kernel_regularizer=regularizer,  activation='relu'))
    model.add(Dense(256, kernel_regularizer=regularizer,  activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

    return model

def get_model_with_initialized_weights(num_classes):
    initializer = tf.keras.initializers.GlorotNormal(seed=None)
    model = Sequential()
    model.add(Dense(1024, kernel_regularizer=l2(0.01), kernel_initializer=initializer, activation='relu', input_shape=(99,)))
    model.add(Dense(1024, kernel_regularizer=l2(0.01), kernel_initializer=initializer,activation='relu'))
    model.add(Dense(512, kernel_regularizer=l2(0.01),kernel_initializer=initializer, activation='relu'))
    model.add(Dense(256, kernel_regularizer=l2(0.01),kernel_initializer=initializer, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

    return model