
# evaluate a specific number of members in an ensemble
import numpy as np
from keras import Input
from keras.applications import VGG16
from keras.saving.save import load_model
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from matplotlib import pyplot, pyplot as plt
from numpy import mean
from numpy import std
import numpy
from numpy import array
from numpy import argmax

from Keras.ensemble_methods import config
from utilities.train_utilities_keras import get_callbacks_for_training


def split_data_randomly(X, y, test_size=config.test_percentage_of_data):
    return train_test_split(X, y, test_size=test_size)

def get_one_bootstraped_sample(X, y):
    ix = [i for i in range(len(X))]
    train_ix = resample(ix, replace=True, n_samples=4500)
    test_ix = [x for x in ix if x not in train_ix]
    # select data
    trainX, trainy = X[train_ix], y[train_ix]
    testX, testy = X[test_ix], y[test_ix]
    return trainX,testX, trainy, testy

def evaluate_multiple_members(n_splits, members, newX, newy):
    # evaluate different numbers of ensembles on hold out set
    single_scores, ensemble_scores = list(), list()

    for i in range(1, n_splits + 1):
        ensemble_score = evaluate_n_members(members, i, newX, newy)
        newy_enc = to_categorical(newy)
        _, single_score = members[i - 1].evaluate(newX, newy_enc, verbose=0)
        print('> %d: single=%.3f, ensemble=%.3f' % (i, single_score, ensemble_score))
        ensemble_scores.append(ensemble_score)
        single_scores.append(single_score)

    return single_scores, ensemble_scores

def plot_single_member_scores_vs_ensemble_members(filename,title,n_splits,single_scores,ensemble_scores):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.suptitle(title, fontsize=7)
    # plot score vs number of ensemble members
    print('Accuracy %.3f (%.3f)' % (mean(single_scores), std(single_scores)))
    x_axis = [i for i in range(1, n_splits + 1)]
    ax.plot(x_axis, single_scores, marker='o', linestyle='None', label='Single model scores')
    ax.plot(x_axis, ensemble_scores, marker='o', label='Ensemble model scores')
    plt.legend()
    fig.savefig(filename)


def train_and_evaluate_new_models_for_kfold_cross_validation(X, y,best_model_name, n_folds=10):
    scores, members = list(), list()

    # prepare the k-fold cross-validation configuration
    kfold = KFold(n_splits=n_folds, shuffle=True,random_state= 1)
    for train_ix, test_ix in kfold.split(X):
        # select samples
        trainX, trainy = X[train_ix], y[train_ix]
        testX, testy = X[test_ix], y[test_ix]
        # evaluate model
        model = get_model(input_dim=2, output_dim=3)
        test_acc = evaluate_model(model, trainX, trainy, testX, testy,best_model_name)
        print('>%.3f' % test_acc)
        scores.append(test_acc)
        members.append(model)
    return scores, members

def train_and_evaluate_new_models_for_ensemble_method(X, y,best_model_name,n_splits = 10, method=get_one_bootstraped_sample):
    # multiple train-test splits
    scores, members = list(), list()

    for _ in range(n_splits):
        # select data
        trainX,testX, trainy, testy = method(X, y)
        # create a new model and evaluate model
        model = get_model(input_dim=2, output_dim=3)
        test_acc = evaluate_model(model, trainX, trainy, testX, testy,best_model_name)
        print('>%.3f' % test_acc)
        scores.append(test_acc)
        members.append(model)
    return scores, members

def get_model(input_dim, output_dim):
    if config.model_type==1:
        model = Sequential()
        model.add(Dense(50, input_dim=input_dim, activation='relu'))
        model.add(Dense(output_dim, activation='softmax'))
    elif config.model_type==2:
        model = Sequential()
        model.add(Dense(64, input_dim=input_dim, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(output_dim, activation='softmax'))
        # model.add(Conv1D(32, 2, activation='relu', input_shape=(input_dim,1)))
        # model.add(MaxPooling1D(1))
        # model.add(Conv1D(64,2, activation='relu'))
        # model.add(MaxPooling1D(1))
        # model.add(Conv1D(64, 2, activation='relu'))
        # model.add(Flatten())
        # model.add(Dense(64, activation='relu'))
        # model.add(Dense(output_dim))
    elif config.model_type==3:
        model = Sequential()
        model.add(Dense(64, input_dim=input_dim, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(output_dim, activation='softmax'))
    elif config.model_type==4:
        model = Sequential()
        model.add(Dense(64, input_dim=input_dim, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(output_dim, activation='softmax'))

    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
#
# print(get_model(2, 3).summary())

def evaluate_n_members(members, n_members, testX, testy):
    # select a subset of members
    subset = members[:n_members]
    # make prediction
    yhat = ensemble_predictions(subset, testX)
    # calculate accuracy
    return accuracy_score(testy, yhat)


# make an ensemble prediction for multi-class classification
def ensemble_predictions(members, testX):
    # make predictions
    yhats = [model.predict(testX) for model in members]
    yhats = array(yhats)
    # sum across ensemble members
    summed = np.sum(yhats, axis=0)
    # argmax across classes
    result = argmax(summed, axis=1)
    return result


# evaluate a single mlp model
def evaluate_model(model, trainX, trainy, testX, testy,best_model_name, epochs=config.epochs):
    # encode targets
    trainy_enc = to_categorical(trainy)
    testy_enc = to_categorical(testy)

    if config.use_pretrained_model:
        model = load_model(f'{best_model_name}.h5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # new_input = Input(shape=(2,))
        # model = VGG16(include_top=False, input_tensor=new_input)

    # fit model
    model.fit(trainX, trainy_enc, epochs=epochs, verbose=0, callbacks=
                                                        get_callbacks_for_training(best_model_name,
                                                                                   metric="accuracy",
                                                                                   patience=config.patience,
                                                                                   save_best_model=config.save_best_model))

    # val_accuracy = history.history['val_accuracy']
    # accuracy = history.history['accuracy']
    # val_loss = history.history['val_loss']
    # loss = history.history['loss']
   # results = dict({accuracy:accuracy, val_accuracy:val_accuracy, val_loss:val_loss, loss:loss})
    # evaluate the model
    _, test_acc = model.evaluate(testX, testy_enc, verbose=0)
    return test_acc
