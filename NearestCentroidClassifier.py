import torch
from data_utilities import load_train_and_test_data
from sklearn.neighbors import NearestCentroid
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd

# Loading the dataset
dataset = load_iris()

# Separating data and target labels
X = pd.DataFrame(dataset.data)
y = pd.DataFrame(dataset.target)

# Splitting training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)


X_train, y_train, X_test, y_test = load_train_and_test_data()

# Creating the Nearest Centroid Classifier
model = NearestCentroid()

# Training the classifier
model.fit(X_train, torch.tensor(y_train))

# Printing Accuracy on Training and Test sets
print(f"Training Set Score : {model.score(X_train, y_train) * 100} %")
print(f"Test Set Score : {model.score(X_test, y_test) * 100} %")

# Printing classification report of classifier on the test set set data
print(f"Model Classification Report : \n{classification_report(y_test, model.predict(X_test))}")