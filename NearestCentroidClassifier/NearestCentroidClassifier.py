import torch
import torchvision
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder

from NearestNeighbor.data_handler import load_datasets
from data_utilities import  load_test_data
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import classification_report
from numpy import mean, std, arange
from timeit import default_timer as timer

experiments_file_name = "nearest_centroid_experiments_sklearn.txt"

start = timer()
# X_train, y_train, X_test, y_test =  load_train_and_test_data(encoder = LabelEncoder, get_chunks=40)
# X_train, y_train, X_test, y_test = load_test_data(encoder=LabelEncoder)

X_train, y_train, X_test, y_test = load_datasets()
# # # we flatten the dataset
X_train = X_train.reshape(-1,3072)
X_test = X_test.reshape(-1,3072)

def print_and_save_results(training_test_score, test_set_score,report ):
     data_used = f'Training data shape:{X_train.shape}\nTraining labels shape:{y_train.shape}\nTest data shape:{X_test.shape}\nTest labels' \
                 f' shape: {y_test.shape}\n'
     print(data_used)
     f = open(experiments_file_name, "a")
     f.write("\n\n")
     f.write(data_used)
     f.write(f'{training_test_score}\n{test_set_score}\n')
     print(training_test_score)
     print(test_set_score)
     print(report)
     f.write(report)
     end = timer()
     time_taken = f"Time passed:{end - start}s\n"
     f.write(time_taken)
     print(time_taken)

def run_nearest_centroid_sklearn():
     # Creating the Nearest Centroid Classifier
     model = NearestCentroid()
     # Training the classifier
     model.fit(X_train, torch.tensor(y_train))
     # Printing Accuracy on Training and Test sets
     training_test_score = f"Training Set Score : {model.score(X_train, y_train) * 100} %"
     test_set_score = f"Test Set Score : {model.score(X_test, y_test) * 100} %"
     # Printing classification report of classifier on the test set set data
     report = f"Model Classification Report : \n{classification_report(y_test, model.predict(X_test))}\n"
     return (training_test_score, test_set_score,report)



# training_test_score, test_set_score,report = run_nearest_centroid_sklearn()
# print_and_save_results(training_test_score, test_set_score,report )

# define model
model = NearestCentroid()
# define model evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
# summarize result
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

# define model
model = NearestCentroid()
# define model evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid
grid = dict()
grid['shrink_threshold'] = arange(0, 1.01, 0.01)
# define search
search = GridSearchCV(model, grid, scoring='accuracy', cv=cv, n_jobs=-1)
# perform the search
results = search.fit(X_train ,y_train)
# summarize
print('Mean Accuracy: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)