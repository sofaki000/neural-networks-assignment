from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler

from data_utilities import load_train_and_test_data, load_test_data

X_train = []
y_train = []
X_test =[]
y_test = []

# Loads some data for developing
X_train, y_train, X_test, y_test = load_test_data(encoder=LabelEncoder)
#Loads hole dataset:
#X_train, y_train, X_test, y_test = load_train_and_test_data(encoder=LabelEncoder, get_chunks=None)

def get_raw_data():
    return X_train, y_train, X_test, y_test

def get_normalized_data():
    normalized_train_data = preprocessing.normalize(X_train)
    normalized_test_data = preprocessing.normalize(X_test)
    return normalized_train_data, y_train, normalized_test_data, y_test


def get_standarized_data():
    scaler_train = preprocessing.StandardScaler().fit(X_train)
    x_scaled_train = scaler_train.transform(X_train)
    scaler_test = preprocessing.StandardScaler().fit(X_test)
    x_scaled_test = scaler_test.transform(X_test)
    return x_scaled_train, y_train, x_scaled_test, y_test

def get_rescaled_data():
    x_scaled_train = MinMaxScaler().fit_transform(X_train)
    x_scaled_test = MinMaxScaler().fit_transform(X_test)
    return x_scaled_train, y_train, x_scaled_test, y_test

def get_robustly_scaled_data():
    transformer_train = RobustScaler().fit(X_train)
    transformer_test = RobustScaler().fit(X_test)
    robustly_scaled_train_data = transformer_train.transform(X_train)
    robustly_scaled_test_data = transformer_test.transform(X_test)
    return robustly_scaled_train_data, y_train, robustly_scaled_test_data, y_test
