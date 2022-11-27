from keras import regularizers
from keras.optimizers import RMSprop
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
import tensorflow as tf

def get_model(input_features, num_classes):
    model = Sequential()
    model.add(Dense(1024, activation='relu', input_shape=(input_features,)))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(),  metrics=['accuracy'])

    return model



def get_model_with_batch_normalization_and_weight_decay(input_features, num_classes, optimizer = RMSprop, lr =0.001, weight_decay=0.01):
    model = Sequential()
    model.add(Dense(1024,   kernel_regularizer=l2(weight_decay), activation='relu', input_shape=(input_features,)))
    model.add(BatchNormalization())
    model.add(Dense(1024,  kernel_regularizer=l2(weight_decay), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(512, kernel_regularizer=l2(weight_decay),  activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(256, kernel_regularizer=l2(weight_decay),  activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer= optimizer(learning_rate=lr), metrics=['accuracy'])
    return model

def get_model_with_weight_decay(input_features,num_classes):
    model = Sequential()
    model.add(Dense(1024,   kernel_regularizer=l2(0.01), activation='relu', input_shape=(input_features,)))
    model.add(Dense(1024,  kernel_regularizer=l2(0.01), activation='relu'))
    model.add(Dense(512, kernel_regularizer=l2(0.01),  activation='relu'))
    model.add(Dense(256, kernel_regularizer=l2(0.01),  activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    return model

def get_model_with_dropout_in_hidden_layers(input_features,num_classes, dropout):
    model = Sequential()
    model.add(Dense(1024,   kernel_regularizer=l2(0.01), activation='relu', input_shape=(input_features,)))
    model.add(Dropout(dropout))
    model.add(Dense(1024,  kernel_regularizer=l2(0.01), activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(512, kernel_regularizer=l2(0.01),  activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(256, kernel_regularizer=l2(0.01),  activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation='softmax'))

    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    return model


def get_model_with_l1_regularizer(input_features,num_classes):
    model = Sequential()
    regularizer= regularizers.L1(0.01)
    model.add(Dense(1024,   kernel_regularizer=regularizer, activation='relu', input_shape=(input_features,)))
    model.add(Dense(1024,  kernel_regularizer=regularizer, activation='relu'))
    model.add(Dense(512, kernel_regularizer=regularizer,  activation='relu'))
    model.add(Dense(256, kernel_regularizer=regularizer,  activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

    return model

def get_model_with_initialized_weights(input_features,num_classes):

    # he weight initialization
    initializer = tf.keras.initializers.HeNormal()
    # initializer = tf.keras.initializers.GlorotNormal(seed=None)
    use_weight_decay = False

    if use_weight_decay:
        model = Sequential()
        model.add(Dense(1024, kernel_regularizer=l2(0.01), kernel_initializer=initializer, activation='relu', input_shape=(input_features,)))
        model.add(Dense(1024, kernel_regularizer=l2(0.01), kernel_initializer=initializer,activation='relu'))
        model.add(Dense(512, kernel_regularizer=l2(0.01),kernel_initializer=initializer, activation='relu'))
        model.add(Dense(256, kernel_regularizer=l2(0.01),kernel_initializer=initializer, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
    else:
        model = Sequential()
        model.add(Dense(1024 , kernel_initializer=initializer, activation='relu',  input_shape=(input_features,)))
        model.add(Dense(1024,  kernel_initializer=initializer, activation='relu'))
        model.add(Dense(512,  kernel_initializer=initializer, activation='relu'))
        model.add(Dense(256,   kernel_initializer=initializer, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    return model