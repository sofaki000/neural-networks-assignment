from keras.callbacks import  EarlyStopping, ModelCheckpoint
from keras.optimizers import RMSprop
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from Models import get_model, get_model_with_weight_decay, get_model_with_initialized_weights, \
    get_model_with_l1_regularizer, get_model_with_batch_normalization_and_weight_decay, \
    get_model_with_dropout_in_hidden_layers
from keras.utils import np_utils
from NearestNeighbor.data_handler import load_datasets
from plot_utilities import save_model_train_and_test_loss_plot

X_train, y_train, X_test, y_test = load_datasets('../data')
X_train = X_train.reshape(-1,3072)
X_test = X_test.reshape(-1,3072)

# patience: how many epochs we wait with no improvement before we stop training
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
mc = ModelCheckpoint('../best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1) # callback to save best model

cb_list = [es,mc]

pca = PCA(0.9)
pca.fit(X_train)

train_img_pca = pca.transform(X_train)
test_img_pca = pca.transform(X_test)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

batch_size = 128
num_classes = 10
epochs = 300


def train_and_evaluate_model(model, file_name, lr=0.001):
    history = model.fit(train_img_pca, y_train,batch_size=batch_size,epochs=epochs,verbose=1, validation_data=(test_img_pca, y_test),  callbacks=cb_list)

    # evaluate the model
    _, train_acc = model.evaluate(train_img_pca, y_train, verbose=0)
    _, test_acc = model.evaluate(test_img_pca.astype("float32"),y_test.astype("float32"))

    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    # plot training history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

    title = f'Stopped at epoch {es.stopped_epoch},Train acc:{train_acc:.3f},Test acc:{test_acc:.3f},lr:{lr}'
    # file_name = 'results/early_stopping'
    save_model_train_and_test_loss_plot(history.history['loss'], history.history['val_loss'], title, file_name)


# early stopping
model = get_model(num_classes=num_classes)
file_name = 'results/early_stopping'
train_and_evaluate_model(model, file_name)

# early stopping with batch norm and weight decay
model = get_model_with_batch_normalization_and_weight_decay(num_classes=num_classes,optimizer = RMSprop, lr =0.01)
file_name = 'results/early_stopping_with_batch_norm_smaller_lr'
train_and_evaluate_model(model, file_name,lr=0.01)

# different lr,early stopping with batch norm and weight decay
model = get_model_with_batch_normalization_and_weight_decay(num_classes=num_classes,optimizer = RMSprop, lr =0.001)
file_name = 'results/early_stopping_with_batch_norm'
train_and_evaluate_model(model, file_name)

# early stopping with l1 regularizer
model = get_model_with_l1_regularizer(num_classes=num_classes)
file_name = 'results/early_stopping_with_l1_regularizer'
train_and_evaluate_model(model, file_name)

# early stopping with dropout layer in hidden layers
model = get_model_with_dropout_in_hidden_layers(num_classes=num_classes)
file_name = 'results/early_stopping_with_dropout_in_hidden_lrs'
train_and_evaluate_model(model, file_name)

# early stopping with weight decay
model = get_model_with_weight_decay(num_classes=num_classes)
file_name = 'results/early_stopping_with_weight_decay'
train_and_evaluate_model(model, file_name)

# early stopping with weight decay and initialized weights
model = get_model_with_initialized_weights(num_classes)
file_name = 'results/early_stopping_with_weight_decay_and_initialized_weights'
train_and_evaluate_model(model, file_name)