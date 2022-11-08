import numpy as np
from sklearn.preprocessing import LabelEncoder

experiments_file_name="nearest_centroid_experiments_impl.txt"
f = open(experiments_file_name, "a")

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
from data_utilities import load_train_and_test_data, load_test_data
import matplotlib.pyplot as plt
from sklearn import preprocessing
from timeit import default_timer as timer


f.write("-------------------------------------------------------------------\n")
# X_train, y_train, X_test, y_test = load_test_data(encoder=LabelEncoder)
X_train, y_train, X_test, y_test = load_train_and_test_data(encoder=LabelEncoder, get_chunks=None)
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
f.write("\nExperiment 1\n")
experiment_1_title="Raw train and test data\n"
train_and_test_data_for_different_preprocessing(experiment_1_title,
                                                X_train=X_train, y_train=y_train,
                                                X_test=X_test, y_test=y_test)

###################### Experiment 2 ######################
f.write("\nExperiment 2\n")
experiment_2_title="Normalizing train and test data\n"
normalized_train_data = preprocessing.normalize(X_train)
normalized_test_data = preprocessing.normalize(X_test)
train_and_test_data_for_different_preprocessing(experiment_2_title,
                                                X_train=normalized_train_data, y_train=y_train,
                                                X_test=normalized_test_data, y_test=y_test)


###################### Experiment 3 ######################
f.write("\nExperiment 3\n")
experiment_3_title="Standardization of train and test data\n"
scaler_train = preprocessing.StandardScaler().fit(X_train)
x_scaled_train = scaler_train.transform(X_train)
scaler_test = preprocessing.StandardScaler().fit(X_test)
x_scaled_test = scaler_test.transform(X_test)
train_and_test_data_for_different_preprocessing(experiment_3_title,
                                                X_train=x_scaled_train, y_train=y_train,
                                                X_test=x_scaled_test, y_test=y_test)


f.close()