
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from NearestNeighbor.NearestNeighborClassifierImpl2 import load_datasets
from NearestNeighbor.PrincipalComponentAnalysis import get_pca_data
import time

from NearestNeighbor.feature_extraction import perform_feature_selection

wine = datasets.load_wine()

def train_with_k(n_neighbor, X_train, y_train, X_test, y_test, metric="minkowski", algorithm="auto", n_jobs=None ):
     classifier = KNeighborsClassifier(n_neighbors=n_neighbor, metric= metric, algorithm=algorithm, n_jobs = n_jobs)
     classifier.fit(X_train, y_train)
     y_test_pred = classifier.predict(X_test)
     acc_score = metrics.accuracy_score(y_test, y_test_pred)
     result = f'K={n_neighbor} accuracy: {acc_score:.5f}'
     return result

def perform_knn_and_record_results(title_exp, X_train , y_train , X_test , y_test, metric="minkowski", algorithm="auto", n_jobs=None ):
     f = open("nearest_neighbor_experiments.txt", "a")
     f.write("\n")
     # experiment name
     f.write(title_exp)
     print(title_exp)

     # shape of data used in the experiment
     data_used = f'Training data shape:{X_train.shape}\nTraining labels shape:{y_train.shape}\nTest data shape:{X_test.shape}\nTest labels' \
                 f' shape: {y_test.shape}\n'
     print(data_used)
     f.write(data_used)

     # performing KNN
     for k in range(1, 4):
          start_time = time.time()
          result = train_with_k(n_neighbor=k, X_train=X_train,  y_train=y_train,  X_test=X_test, y_test=y_test, metric=metric, algorithm=algorithm, n_jobs=n_jobs)
          end_time = time.time()
          time_took_for_training = f'For k={k} training time was {(end_time-start_time):.3f}s'
          f.write(result)
          f.write(time_took_for_training)
          print(time_took_for_training)
          print(result)



# X_train, y_train, X_test, y_test = load_train_and_test_data(get_chunks=40)
#X_train, y_train, X_test, y_test = load_datasets()

title_exp = "Wine dataset\n"
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3)
perform_knn_and_record_results(title_exp, X_train , y_train , X_test , y_test)

title_exp = "Wine dataset with feature extraction\n"
reducted_train_features, reducted_test_features = perform_feature_selection(X_train, y_train, X_test, y_test,4)
perform_knn_and_record_results(title_exp, reducted_train_features , y_train , reducted_test_features , y_test)



# title_exp = "CIFAR with normalized data (pixels from 0 to 1 inclusive)\n"
X_train, y_train, X_test, y_test = load_datasets()
# # # we flatten the dataset
X_train = X_train.reshape(-1,3072)
X_test = X_test.reshape(-1,3072)
# perform_knn_and_record_results(title_exp, X_train , y_train , X_test , y_test)


title_exp = "FILTER METHOD: CIFAR with feature extraction using Chi-Squared statistical test for non-negative features, keeping 99 features"
reducted_train_features, reducted_test_features = perform_feature_selection(X_train, y_train, X_test, y_test,99)
perform_knn_and_record_results(title_exp, reducted_train_features , y_train , reducted_test_features , y_test)


title_exp = "FILTER METHOD: CIFAR with feature extraction using Chi-Squared statistical test for non-negative features, keeping 2000 features"
reducted_train_features, reducted_test_features = perform_feature_selection(X_train, y_train, X_test, y_test,2000)
perform_knn_and_record_results(title_exp, reducted_train_features , y_train , reducted_test_features , y_test)


title_exp = "FILTER METHOD: CIFAR with feature extraction using Chi-Squared statistical test for non-negative features, keeping 2500 features"
reducted_train_features, reducted_test_features = perform_feature_selection(X_train, y_train, X_test, y_test,2500)
perform_knn_and_record_results(title_exp, reducted_train_features , y_train , reducted_test_features , y_test)



title_exp = "WRAPPER METHOD (recursive feature elimination): CIFAR with feature extraction"
# reducted_train_features = perform_feature_selection_with_recursive_feature_elimination(X_train, y_train, X_test, y_test,10)
# perform_knn_and_record_results(title_exp, reducted_train_features , y_train , X_test , y_test)
#


# with pca
title_exp = "CIFAR with PCA\n"
x_train_with_pca, x_test_with_pca = get_pca_data(X_train, X_test)
perform_knn_and_record_results(title_exp, x_train_with_pca , y_train , x_test_with_pca , y_test)


# ### with cosine similarity instead of euclidean distance
title_exp = "CIFAR without PCA, with cosine similarity to calculate nearest neighbors\n"
perform_knn_and_record_results(title_exp, X_train , y_train , X_test , y_test, metric='cosine', algorithm='brute',  n_jobs=-1 )

title_exp = "CIFAR with PCA, with cosine similarity to calculate nearest neighbors\n"
perform_knn_and_record_results(title_exp, x_train_with_pca , y_train , x_test_with_pca , y_test, metric='cosine', algorithm='brute',  n_jobs=-1 )




