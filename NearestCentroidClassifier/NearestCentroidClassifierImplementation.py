import numpy as np
from NearestCentroidClassifier import config
from data_utilities.preprocessing_data_utilities import get_raw_data, get_normalized_data, get_standarized_data, get_rescaled_data, \
    get_robustly_scaled_data


"""A nearest centroid classifier. Similar class to NearestCentroid from module sklearn.neighbors.
  Methods:
      fit(np.array(X_train), np.array(y_train)) - model training method
      predict(np.array(X_test)) - method returning a numpy array of predicted class labels for input test data.
      score(np.array(X_test, y_test)) - method checking accuracy of the model by. Returns float number.
  """
class Nearest_centroid_Impl(object):
    def __int__(self):
        self.X = None
        self.Y = None
        self.means = None

    def fit(self, X_train, y_train):
        if X_train.shape[0] != y_train.shape[0]:
            print(X_train.shape[0], y_train.shape[0])
            raise ValueError("Training and testing sets are not same size")

        self.X = X_train
        self.Y = y_train
        self.means = self._find_means()

        return self.means
    def _find_means(self):
        """Auxillary method used for computing feature classes means."""
        # gia kathe klash, briskei ton meso oro tou kathe feature
        #klash 1: [mean_feature1, mean_feature2, mean_feature3....]
        #klash 2: [mean_feature1, mean_feature2, mean_feature3....]
        return np.array([np.mean(self.X[self.Y == i], axis=0) for i in np.unique(self.Y)])
    def _find_distance(self, x):
        """Auxillary method for computing distances between one of the testing vector and classes feature means."""
        return np.sqrt(np.sum(np.power(self.means - x, 2), axis=1))
    def predict(self, A):
        if type(A) is not np.ndarray:
            raise ValueError("Both sets must numpy.ndarray type.")

        result = np.array([]).astype('int8')
        i = 0
        n = A.shape[0]
        while i < n:
            #for each sample, we calculate the distances from the class feature means
            distances = self._find_distance(A[i])
            # we find the min distance
            min_dist= np.argmin(distances)
            result = np.append(result,min_dist)
            i += 1
        return result

    def score(self, A, B):
        results = self.predict(A)
        acc = np.mean(results == B)
        return acc, results

from timeit import default_timer as timer
from time import gmtime, strftime

experiments_date = strftime("%Y-%m-%d %H:%M:%S", gmtime())
f = open(config.experiments_file_name, "a")

f.write(f"---------------------------{experiments_date}------------------------------------\n")
X_train, y_train, X_test, y_test = get_raw_data()
data_used = f'Training data shape:{X_train.shape}\nTraining labels shape:{y_train.shape}\nTest data shape:{X_test.shape}\nTest labels' \
            f' shape: {y_test.shape}\n'
f.write(data_used)

def train_and_test_data_for_different_preprocessing(experiment_title, X_train, y_train, X_test, y_test):
    start = timer()
    # train
    classifier = Nearest_centroid_Impl()
    means = classifier.fit(X_train, y_train)
    number_of_classes = means.shape[0]
    # test
    accuracy, results = classifier.score(X_test, y_test)
    end = timer()
    accuracy_result = f"Accuracy: {accuracy}\n"
    time_passed = f"Training time:{end - start}\n"
    print(experiment_title)
    print(accuracy_result)
    print(time_passed)
    f.write(experiment_title)
    f.write(accuracy_result)
    f.write(time_passed)


###################### Experiment 1 ######################
train_and_test_data_for_different_preprocessing(config.experiment_1_title,
                                                X_train=X_train,
                                                y_train=y_train,
                                                X_test=X_test,
                                                y_test=y_test)

###################### Experiment 2 ######################
# normalized_train_data, y_train, normalized_test_data, y_test = get_normalized_data()
# train_and_test_data_for_different_preprocessing(config.experiment_2_title,
#                                                 X_train=normalized_train_data,
#                                                 y_train=y_train,
#                                                 X_test=normalized_test_data,
#                                                 y_test=y_test)
#
#
# ###################### Experiment 3 ######################
# # StandardScaler therefore cannot guarantee balanced feature scales in the presence of outliers.
# x_scaled_train, y_train, x_scaled_test, y_test = get_standarized_data()
# train_and_test_data_for_different_preprocessing(config.experiment_3_title,
#                                                 X_train=x_scaled_train, y_train=y_train,
#                                                 X_test=x_scaled_test, y_test=y_test)
#
#
# ###################### Experiment 4 ######################
# # MinMaxScaler rescales the data set such that all feature values are in the range [0, 1] as shown in the right panel below.
# x_scaled_train, y_train, x_scaled_test, y_test =get_rescaled_data()
# train_and_test_data_for_different_preprocessing(config.experiment_4_title,
#                                                 X_train=x_scaled_train,
#                                                 y_train=y_train,
#                                                 X_test=x_scaled_test,
#                                                 y_test=y_test)
#
# ###################### Experiment 5 ######################
# # Robust scaler: Scales features using statistics that are robust to outliers.
# # This Scaler removes the median and scales the data according to the quantile range (defaults to IQR:
# # Interquartile Range). The IQR is the range between the 1st quartile (25th quantile) and the 3rd quartile (75th quantile).
# # Centering and scaling happen independently on each feature by computing the relevant statistics
# # on the samples in the training set. Median and interquartile range are then stored to be used on later data using the transform method.
#
# x_scaled_train, y_train, x_scaled_test, y_test = get_robustly_scaled_data()
# train_and_test_data_for_different_preprocessing(config.experiment_5_title, X_train=x_scaled_train, y_train=y_train, X_test=x_scaled_test, y_test=y_test)
# f.close()