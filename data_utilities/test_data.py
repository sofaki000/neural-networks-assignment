from sklearn.datasets import make_blobs


def get_test_data():
    dataX, datay = make_blobs(n_samples=55000, centers=3, n_features=2, cluster_std=2, random_state=2)
    X, newX = dataX[:5000, :], dataX[5000:, :]
    y, newy = datay[:5000], datay[5000:]
    return X, newX, y,newy