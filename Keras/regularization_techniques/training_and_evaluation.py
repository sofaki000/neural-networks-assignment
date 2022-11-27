from Keras.regularization_techniques import config
from utilities.plot_utilities import save_model_train_and_test_loss_plot, save_model_train_and_test_accuracy_plot
from utilities.train_utilities_keras import get_callbacks_for_training
import time
def train_and_evaluate_model(x_train, y_train, x_test, y_test, model,
                             file_name_loss, file_name_acc,title_acc, title_loss,
                             best_model_name,lr=0.001):

    start_training_time =time.time()

    training_callbacks = get_callbacks_for_training(best_model_name)
    history = model.fit(x_train, y_train,
                        batch_size=config.batch_size,
                        epochs=config.epochs,
                        verbose=1, validation_data=(x_test, y_test),
                        callbacks=training_callbacks)

    # evaluate the model
    _, train_acc = model.evaluate(x_train, y_train, verbose=0)
    _, test_acc = model.evaluate(x_test.astype("float32"),y_test.astype("float32"))

    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

    epochs_trained = training_callbacks[0].stopped_epoch
    if epochs_trained==0:
        epochs_trained= config.epochs

    val_accuracy = history.history['val_accuracy']
    accuracy = history.history['accuracy']
    val_loss = history.history['val_loss']
    loss = history.history['loss']
    train_size = (x_train.shape[0])
    test_size = (x_test.shape[0])

    training_duration = time.time() - start_training_time
    title_loss = f'{title_loss},epoch {epochs_trained},lr:{lr},Train size:{train_size},Test size:{test_size},Training duration:{training_duration:.2f}s'
    title_acc = f'{title_acc},epoch {epochs_trained},lr:{lr},Train size:{train_size},Test size:{test_size}'

    save_model_train_and_test_loss_plot(loss, val_loss, title_loss, file_name_loss)
    save_model_train_and_test_accuracy_plot(accuracy, val_accuracy, title_acc, file_name_acc)