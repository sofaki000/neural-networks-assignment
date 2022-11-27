from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA

# we will pass how much variance we want PCA to capture. passing 0.9-> PCA will hold 90% of the variance
# and the number of components required to capture 90% variance
def get_pca_data(x_train, x_test):
    pca = PCA(0.9)
    pca.fit(x_train)
    train_img_pca = pca.transform(x_train)
    test_img_pca = pca.transform(x_test)
    return train_img_pca, test_img_pca

def perform_feature_selection_with_recursive_feature_elimination(model, X_train, y_train, X_test, y_test,num_features):
     # model = KNeighborsClassifier() !!doesnt work for knn
     rfe = RFE(model, n_features_to_select=num_features, step = 3)
     fit = rfe.fit(X_train, y_train)

     # sfm = SelectFromModel(estimator=model,  threshold=0.25)
     #sfm.fit(X_train, y_train)
     print("Num Features: %s" % (fit.n_features_))
     print("Selected Features: %s" % (fit.support_))
     print("Feature Ranking: %s" % (fit.ranking_))
     return fit


def perform_feature_selection(X_train, y_train, X_test, y_test, num_features_to_keep=4):
    # with feature selection
    test = SelectKBest(score_func=chi2, k=num_features_to_keep)
    fit_train = test.fit(X_train, y_train)
    fit_test = test.fit(X_test, y_test)

    # Summarize scores
    train_features = fit_train.transform(X_train)
    test_features = fit_test.transform(X_test)
    return train_features, test_features