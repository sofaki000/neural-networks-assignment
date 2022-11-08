import numpy as np

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