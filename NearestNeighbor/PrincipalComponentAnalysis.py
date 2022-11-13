from sklearn.decomposition import PCA

# we will pass how much variance we want PCA to capture. passing 0.9-> PCA will hold 90% of the variance
# and the number of components required to capture 90% variance
def get_pca_data(x_train, x_test):
    pca = PCA(0.9)
    pca.fit(x_train)
    train_img_pca = pca.transform(x_train)
    test_img_pca = pca.transform(x_test)
    return train_img_pca, test_img_pca
