from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

def get_callbacks_for_training(best_model_name, metric="val_loss", patience=10, save_best_model=True):
    es = EarlyStopping(monitor=metric, mode='min', verbose=1, patience=patience)
    mc = ModelCheckpoint(f'./{best_model_name}.h5', monitor=metric, mode='min', save_best_only=True,  verbose=1)  # callback to save best model

    if save_best_model:
        cb_list = [es, mc]
    else:
        cb_list = [es]
    return cb_list



def get_callbacks_for_training_with_visualization(best_model_name,log_dir):
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    mc = ModelCheckpoint(f'./{best_model_name}.h5', monitor='val_loss', mode='min', save_best_only=True,  verbose=1)  # callback to save best model

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    cb_list = [es, mc,tensorboard_callback]
    return cb_list

