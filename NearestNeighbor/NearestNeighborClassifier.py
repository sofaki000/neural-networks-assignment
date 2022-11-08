from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import warnings
from data_utilities import load_train_and_test_data
warnings.filterwarnings('ignore')
from enum import Enum
from timeit import default_timer as timer

start = timer()

class minkowski_metric(Enum):
     EUCLIDIEAN_DISTANCE = 1
     MANHATTAN_DISTANCE = 2


f = open("nearest_neighbor_experiments.txt", "a")
f.write("\n\n")
X_train, y_train, X_test, y_test = load_train_and_test_data(get_chunks=40)


data_used = f'Training data shape:{X_train.shape}\nTraining labels shape:{y_train.shape}\nTest data shape:{X_test.shape}\nTest labels'\
     f' shape: {y_test.shape}\n'
print(data_used)
f.write(data_used)

def train_with_k(n_neighbor, algorithm="auto", p=minkowski_metric.EUCLIDIEAN_DISTANCE):
     classifier = KNeighborsClassifier(n_neighbors=n_neighbor, algorithm=algorithm, p=p.value)
     classifier.fit(X_train, y_train)
     y_test_pred = classifier.predict(X_test)
     acc = (np.mean(y_test_pred == y_test))
     result = f'K={n_neighbor}\naccuracy: {acc:.5f}\n'
     print(result)
     return result


title = f"Algorithm: auto, minkowski metric: Euclidean distance\n"
f.write(title)
print(title)
for i in range(1,7):
     print(f'k:{i}')
     result = train_with_k(i)
     f.write(result)


f.write("\n")
title =f"Algorithm: auto, minkowski metric: Manhattan distance\n"
f.write(title)
print(title)
for i in range(1,7):
     print(f'k:{i}')
     result = train_with_k(i,algorithm="auto", p=minkowski_metric.MANHATTAN_DISTANCE)
     f.write(result)


end = timer()
time_passed = f"Time passed:{end-start}\n"
f.write(time_passed)


f.close()
