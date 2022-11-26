from keras.callbacks import  EarlyStopping, ModelCheckpoint
from keras.optimizers import RMSprop
from sklearn.decomposition import PCA
from Models import get_model, get_model_with_weight_decay, get_model_with_initialized_weights, \
    get_model_with_l1_regularizer, get_model_with_batch_normalization_and_weight_decay, \
    get_model_with_dropout_in_hidden_layers
from keras.utils import np_utils
from NearestNeighbor.data_handler import load_datasets
from data_utilities.sre_dataset_utilities import get_transformed_data
from utilities.plot_utilities import save_model_train_and_test_loss_plot, save_model_train_and_test_accuracy_plot
import os

experiment_folder= 'results\\without_pca_wd_and_dropout'
os.makedirs(experiment_folder, exist_ok=True)
# X_train, y_train, X_test, y_test = load_datasets('../data')
# X_train = X_train.reshape(-1,3072)
# X_test = X_test.reshape(-1,3072)
# # pca = PCA(0.9)
# # train_img_pca = pca.fit_transform(X_train)
# # test_img_pca = pca.transform(X_test)
# # y_train = np_utils.to_categorical(y_train)
# # y_test = np_utils.to_categorical(y_test)
from utilities.train_utilities import get_callbacks_for_training

x_train, y_train, x_test, y_test = get_transformed_data(4)
output_classes = 7
use_pca = False

if use_pca:
    pca = PCA(0.9)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)
    input_features = 16
else:
    input_features = 40


batch_size = 128
num_classes = output_classes
epochs = 300


def train_and_evaluate_model(model, file_name_loss, file_name_acc,title_acc, title_loss,   best_model_name,lr=0.001):
    # patience: how many epochs we wait with no improvement before we stop training

    training_callbacks = get_callbacks_for_training(best_model_name)
    history = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1, validation_data=(x_test, y_test),  callbacks=training_callbacks)

    # evaluate the model
    _, train_acc = model.evaluate(x_train, y_train, verbose=0)
    _, test_acc = model.evaluate(x_test.astype("float32"),y_test.astype("float32"))

    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    epochs_trained = training_callbacks[0].stopped_epoch
    if epochs_trained==0:
        epochs_trained= epochs

    val_accuracy = history.history['val_accuracy']
    accuracy = history.history['accuracy']
    val_loss = history.history['val_loss']
    loss = history.history['loss']

    train_size = (x_train.shape[0])
    test_size = (x_test.shape[0])

    title_loss = f'{title_loss},epoch {epochs_trained},lr:{lr},Train size:{train_size},Test size:{test_size}'
    title_acc = f'{title_acc},epoch {epochs_trained},lr:{lr},Train size:{train_size},Test size:{test_size}'

    save_model_train_and_test_loss_plot(loss, val_loss, title_loss, file_name_loss)
    save_model_train_and_test_accuracy_plot(accuracy, val_accuracy, title_acc, file_name_acc)

# # early stopping
# model = get_model(input_features=input_features,num_classes=num_classes)
# file_name_loss = f'{experiment_folder}/early_stopping_loss'
# file_name_acc = f'{experiment_folder}/early_stopping_acc'
# title_loss='Early stopping loss'
# title_acc='Early stopping acc'
# train_and_evaluate_model(model, file_name_loss=file_name_loss , file_name_acc=file_name_acc,title_acc=title_acc, title_loss=title_loss, best_model_name="best_model_early_stop")

# early stopping with batch norm and weight decay
model = get_model_with_batch_normalization_and_weight_decay(input_features=input_features,num_classes=num_classes,optimizer = RMSprop, lr =0.01, weight_decay=0.01)
file_name_loss = f'{experiment_folder}/early_stopping_with_batch_norm_smaller_lr_loss_v1'
file_name_acc = f'{experiment_folder}/early_stopping_with_batch_norm_smaller_lr_acc_v1'
title_loss='Batch norm and weight decay=0.01 loss'
title_acc='Batch norm and weight decay=0.01 acc'
train_and_evaluate_model(model, file_name_loss=file_name_loss , file_name_acc=file_name_acc,title_acc=title_acc, title_loss=title_loss, lr=0.01, best_model_name="best_model_batch_norm_small_lr")

# different lr,early stopping with batch norm and weight decay
model = get_model_with_batch_normalization_and_weight_decay(input_features=input_features ,num_classes=num_classes,optimizer = RMSprop, lr =0.01, weight_decay=0.01)
file_name_loss = f'{experiment_folder}/early_stopping_with_batch_norm_loss_v2'
file_name_acc = f'{experiment_folder}/early_stopping_with_batch_norm_loss_acc_v2'
title_loss='Batch norm and weight decay=0.1 loss'
title_acc='Batch norm and weight decay=0.1 cc'
train_and_evaluate_model(model, file_name_loss=file_name_loss , file_name_acc=file_name_acc,title_acc=title_acc, title_loss=title_loss, best_model_name="best_model_batch_norm")

# early stopping with l1 regularizer
# model = get_model_with_l1_regularizer(input_features=input_features,num_classes=num_classes)
# file_name_loss = f'{experiment_folder}/early_stopping_with_l1_regularizer_loss'
# file_name_acc = f'{experiment_folder}/early_stopping_with_l1_regularizer_acc'
# title_loss='L1 regularizer loss'
# title_acc='L1 regularizer acc'
# train_and_evaluate_model(model, file_name_loss=file_name_loss , file_name_acc=file_name_acc,title_acc=title_acc, title_loss=title_loss, best_model_name="best_model_l1_regul")

# early stopping with DROPOUT layer in hidden layers
model = get_model_with_dropout_in_hidden_layers(input_features=input_features,num_classes=num_classes,dropout=.2)
file_name_loss = f'{experiment_folder}/early_stopping_with_dropout_in_hidden_lrs_loss_v1'
file_name_acc = f'{experiment_folder}/early_stopping_with_dropout_in_hidden_lrs_acc_v1'
title_loss='Dropout=0.2 loss'
title_acc='Dropout=0.2 acc'
train_and_evaluate_model(model,  file_name_loss=file_name_loss , file_name_acc=file_name_acc,title_acc=title_acc, title_loss=title_loss, best_model_name="best_model_dropout")

# early stopping with DROPOUT layer in hidden layers
model = get_model_with_dropout_in_hidden_layers(input_features=input_features,num_classes=num_classes,dropout=.5)
file_name_loss = f'{experiment_folder}/early_stopping_with_dropout_in_hidden_lrs_loss_v2'
file_name_acc = f'{experiment_folder}/early_stopping_with_dropout_in_hidden_lrs_acc_v2'
title_loss='Dropout=0.5 loss'
title_acc='Dropout=0.5 acc'
train_and_evaluate_model(model,  file_name_loss=file_name_loss , file_name_acc=file_name_acc,title_acc=title_acc, title_loss=title_loss, best_model_name="best_model_dropout")


# # early stopping with weight decay
# model = get_model_with_weight_decay(input_features=input_features,num_classes=num_classes)
# file_name_loss = f'{experiment_folder}/early_stopping_with_weight_decay_loss'
# file_name_acc = f'{experiment_folder}/early_stopping_with_weight_decay_acc'
# title_loss='Weight decay loss'
# title_acc='Weight decay acc'
# train_and_evaluate_model(model, file_name_loss=file_name_loss , file_name_acc=file_name_acc,title_acc=title_acc, title_loss=title_loss,best_model_name="best_model_weight_decay")
#
# # early stopping with weight decay and initialized weights
# model = get_model_with_initialized_weights(input_features=input_features,num_classes=num_classes)
# file_name_loss = f'{experiment_folder}/early_stopping_with_weight_decay_and_initialized_weights_loss'
# file_name_acc =f'{experiment_folder}/early_stopping_with_weight_decay_and_initialized_weights_acc'
# title_loss='Initialized weights loss'
# title_acc='Initialized weights acc'
# train_and_evaluate_model(model,  file_name_loss=file_name_loss , file_name_acc=file_name_acc,title_acc=title_acc, title_loss=title_loss, best_model_name="best_model_weight_decay_initialzied_weights")