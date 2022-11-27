import time

from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold

from data_utilities.cifar10_utilities import load_datasets


def train_with_nearest_centroid(X_train, y_train, X_test, y_test):
    model = NearestCentroid()
    model.fit(X_train, y_train)

    kf = KFold(n_splits=6, shuffle=True, random_state=42)
    cross_val_score_train = cross_val_score(model, X_train, y_train, cv=kf)
    cross_val_score_test = cross_val_score(model, X_test, y_test, cv=kf)

    # Printing Accuracy on Training and Test sets
    result = f'Training Set Score : {model.score(X_train, y_train) * 100:.3f} %\nTest Set Score : {model.score(X_test, y_test) * 100:.3f}%\nCross validation average score on train data: {cross_val_score_train.mean()* 100:.3f}%\nCross validation average score on test data: {cross_val_score_test.mean()* 100:.3f}%'
    return result


def perform_knc_and_record_results(title_exp, X_train, y_train, X_test, y_test):
    f = open("NearestCentroidExperiments.txt", "a")
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
    start_time = time.time()
    result = train_with_nearest_centroid(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    end_time = time.time()
    time_took_for_training = f'Training time was {(end_time - start_time):.2f}s'
    f.write(result)
    f.write(time_took_for_training)
    print(time_took_for_training)
    print(result)


X_train, y_train, X_test, y_test = load_datasets()
# # # we flatten the dataset
X_train = X_train.reshape(-1,3072)
X_test = X_test.reshape(-1,3072)
Xtrain, X_test, ytrain, ytest = train_test_split(X_train, y_train, test_size=0.15)


experiment_title = "Nearest centroid classifier"
xtrain, xtest, ytrain, ytest=train_test_split(Xtrain, ytrain, test_size=0.15)
# perform_knc_and_record_results(experiment_title, xtrain, ytrain, xtest, ytest)
#
# experiment_title = "Nearest centroid classifier with PCA"
# x_train_with_pca, x_test_with_pca = get_pca_data(xtrain, xtest)
# perform_knc_and_record_results(experiment_title, x_train_with_pca, ytrain, x_test_with_pca, ytest)


title_exp = "FILTER METHOD: CIFAR with feature extraction using Chi-Squared statistical test for non-negative features, keeping 99 features"
reducted_train_features, reducted_test_features = perform_feature_selection(xtrain, ytrain, xtest, ytest, 99)
perform_knc_and_record_results(title_exp, reducted_train_features, ytrain, reducted_test_features, ytest)

title_exp = "FILTER METHOD: CIFAR with feature extraction using Chi-Squared statistical test for non-negative features, keeping 2000 features"
reducted_train_features, reducted_test_features = perform_feature_selection(xtrain, ytrain, xtest, ytest,2000)
perform_knc_and_record_results(title_exp, reducted_train_features, ytrain, reducted_test_features, ytest)


