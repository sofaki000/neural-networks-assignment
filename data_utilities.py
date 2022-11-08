
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from utilities import extract_mfcc
import warnings
import os
warnings.filterwarnings('ignore')

def load_feeling(feelings):
    paths = []
    labels = []
    path = '../TESS Toronto emotional speech set data'
    if os.path.exists(path) is False:
        raise Exception("Can't find data")
    counter =0
    for dirname, _, filenames in os.walk(path):
        counter+=1
        for filename in filenames:
            label = filename.split('_')[-1]
            label = label.split('.')[0]
            if label in feelings:
                labels.append(label.lower())
                paths.append(os.path.join(dirname, filename))
        # if len(paths) == 2800:
        #     break
        if len(paths) == 2:
            break
        if counter==3:
            return paths, labels
    print('Dataset is Loaded')
    return paths, labels


# paths, labels = load_feeling(["angry", "Sad"])
# print(paths)
def loadTestSet(get_chunks=10):
    paths = []
    labels = []
    path = '../test_data3'
    if os.path.exists(path) is False:
        raise Exception("Can't find data")
    counter =0
    for dirname, _, filenames in os.walk(path):
        counter+=1
        for filename in filenames:
            paths.append(os.path.join(dirname, filename))
            # print(filename)
            label = filename.split('_')[-1]
            label = label.split('.')[0]
            labels.append(label.lower())
        # if len(paths) == 2800:
        #     break
        if len(paths) == 2:
            break
        # if counter==get_chunks:
        #     return paths, labels
    print('Dataset is Loaded')
    return paths, labels



def loadDataFromPathAndLabels(paths, labels, encoder=OneHotEncoder):
    df = pd.DataFrame()
    df['speech'] = paths
    df['label'] = labels
    samples_size = len(labels)
    # for each speech sample apply function extract_mfcc
    X_mfcc = df['speech'].apply(lambda x: extract_mfcc(x))

    input_features = [x for x in X_mfcc]
    input_features = np.array(input_features)  # samples x n_features
    enc = encoder()
    actual_labels = enc.fit_transform(df[['label']])
    if hasattr(actual_labels, "__len__") is False:
        actual_labels = actual_labels.toarray()
    data_split = (int)(samples_size * 0.7)
    X_train = input_features[:data_split]
    y_train = actual_labels[:data_split]
    X_test = input_features[data_split:]
    y_test = actual_labels[data_split:]
    return X_train, y_train, X_test, y_test



def load_test_data(encoder):
    paths, labels = loadTestSet(encoder)
    return loadDataFromPathAndLabels(paths, labels,encoder=encoder)

def loadPathsAndLabels(get_chunks=10):
    paths = []
    labels = []
    path = '../TESS Toronto emotional speech set data'
    if os.path.exists(path) is False:
        raise Exception("Can't find data")
    counter =0
    for dirname, _, filenames in os.walk(path):
        counter+=1
        for filename in filenames:
            paths.append(os.path.join(dirname, filename))
            # print(filename)
            label = filename.split('_')[-1]
            label = label.split('.')[0]
            labels.append(label.lower())
        # if len(paths) == 2800:
        #     break
        if len(paths) == 2:
            break
        if get_chunks is not None and counter==get_chunks:
           return paths, labels
    print('Dataset is Loaded')
    return paths, labels


def load_train_and_test_data_for_some_feelings(feelings):
    paths, labels = load_feeling(feelings)
    return loadDataFromPathAndLabels(paths, labels)

def load_train_and_test_data(encoder = OneHotEncoder, get_chunks=10):
    paths, labels = loadPathsAndLabels(get_chunks)
    return loadDataFromPathAndLabels(paths, labels)
