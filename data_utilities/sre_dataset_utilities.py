import librosa
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import warnings
import os
warnings.filterwarnings('ignore')

# path = '../TESS Toronto emotional speech set data'
data_path = 'C:/Users/Lenovo/Desktop/νευρωνικά δίκτυα/neural-networks-assignment/data/'

def load_feeling(feelings):
    paths = []
    labels = []
    path = f'{data_path}test'
    if os.path.exists(path) is False:
        raise Exception("Can't find dataa")
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
def loadTestSet(load_dataset=0):
    paths = []
    labels = []
    if load_dataset==0:
        path =  f'{data_path}test'
    elif load_dataset==1:
        path =  f'{data_path}test_data'
    elif load_dataset==2:
        path =  f'{data_path}test_data2'
    elif load_dataset==3:
        path =  f'{data_path}test_data3'

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
        if len(paths) == 2:
            break
    print('Dataset is Loaded')
    return paths, labels

def extract_mfcc(filename):
    data, sampling_rate = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
    return mfcc

def loadDataFromPathAndLabels(paths, labels, encoder=OneHotEncoder ):
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

    X_test = input_features[:data_split]
    y_test = actual_labels[:data_split]
    X_train = input_features[data_split:]
    y_train = actual_labels[data_split:]
    return X_train, y_train, X_test, y_test


def load_test_data(load_dataset):
    print("loading test data is called")
    paths, labels = loadTestSet(load_dataset)
    return loadDataFromPathAndLabels(paths, labels)


def load_train_and_test_data_for_some_feelings(feelings):
    paths, labels = load_feeling(feelings)
    return loadDataFromPathAndLabels(paths, labels)


def load_feel_test():
    return load_train_and_test_data_for_some_feelings(['angry' , 'happy', 'fear'])

def get_transformed_data(load_dataset):
    X_train, y_train, X_test, y_test = load_test_data(load_dataset) # load_feel_test()

    # preprocessing
    scaler = preprocessing.StandardScaler().fit(X_train)
    scaler_test = preprocessing.StandardScaler().fit(X_test)
    y_train =np.float32(y_train.toarray())
    y_test =np.float32(y_test.toarray())
    x_train = scaler.transform(X_train)
    x_test = scaler_test.transform(X_test)
    return x_train, y_train, x_test, y_test