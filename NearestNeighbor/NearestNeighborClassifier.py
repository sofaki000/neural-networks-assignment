from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import warnings

from NearestNeighbor.NearestNeighborClassifierImpl2 import load_datasets
warnings.filterwarnings('ignore')
from enum import Enum
from timeit import default_timer as timer

# start = timer()
#
# class minkowski_metric(Enum):
#      EUCLIDIEAN_DISTANCE = 1
#      MANHATTAN_DISTANCE = 2
#
#
# f = open("nearest_neighbor_experiments.txt", "a")
# f.write("\n\n")
#
#
# # X_train, y_train, X_test, y_test = load_train_and_test_data(get_chunks=40)
# X_train, y_train, X_test, y_test = load_datasets()
#
# data_used = f'Training data shape:{X_train.shape}\nTraining labels shape:{y_train.shape}\nTest data shape:{X_test.shape}\nTest labels'\
#      f' shape: {y_test.shape}\n'
# print(data_used)
# f.write(data_used)
#
# def train_with_k(n_neighbor, algorithm="auto", p=minkowski_metric.EUCLIDIEAN_DISTANCE):
#      #classifier = KNeighborsClassifier(n_neighbors=n_neighbor, algorithm=algorithm, p=p.value)
#      classifier = KNeighborsClassifier(n_neighbors=n_neighbor)
#      classifier.fit(X_train, y_train)
#      y_test_pred = classifier.predict(X_test)
#      acc = (np.mean(y_test_pred == y_test))
#      result = f'K={n_neighbor}\naccuracy: {acc:.5f}\n'
#      print(result)
#      return result
#
#
# title = f"Algorithm: auto, minkowski metric: Euclidean distance\n"
# f.write(title)
# print(title)
# for i in range(1,4):
#      result = train_with_k(i)
#      f.write(result)
#
# f.write("\n")
# title =f"Algorithm: auto, minkowski metric: Manhattan distance\n"
# f.write(title)
# print(title)
# for i in range(1,4):
#      print(f'k:{i}')
#      result = train_with_k(i,algorithm="auto", p=minkowski_metric.MANHATTAN_DISTANCE)
#      f.write(result)
#
# end = timer()
# time_passed = f"Time passed:{end-start}\n"
# f.write(time_passed)
# f.close()

X_train, y_train, X_test, y_test = load_datasets()

# flatten out all images to be one-dimensional
# Xtr_rows = X_train.reshape(X_train.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
# Xte_rows = X_test.reshape(X_test.shape[0], 32 * 32 * 3) # Xte_rows becomes 10000 x 3072
Xtr_rows = X_train.reshape(X_train.shape[0], 32 * 32  ) # Xtr_rows becomes 50000 x 3072
Xte_rows = X_test.reshape(X_test.shape[0], 32 * 32 ) # Xte_rows becomes 10000 x 3072


class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y

  def predict(self, X):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0]
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

    # loop over all test rows
    for i in range(num_test):
      # find the nearest training image to the i'th test image using the L1 distance (sum of absolute value differences)
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
      min_index = np.argmin(distances) # get the index with smallest distance
      Ypred[i] = self.ytr[min_index] # predict the label of the nearest example
      print("\rClassifiying {} ...".format(i),end="")
    return Ypred

nn = NearestNeighbor() # create a Nearest Neighbor classifier class
nn.train(Xtr_rows, y_train) # train the classifier on the training images and labels
Yte_predict = nn.predict(Xte_rows) # predict labels on the test images
# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)
print ('accuracy: %f' % ( np.mean(Yte_predict == y_test) ))