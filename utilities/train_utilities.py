from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

def get_callbacks_for_training(best_model_name):
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    mc = ModelCheckpoint(f'./{best_model_name}.h5', monitor='val_loss', mode='min', save_best_only=True,  verbose=1)  # callback to save best model

    cb_list = [es, mc]
    return cb_list



def get_callbacks_for_training_with_visualization(best_model_name,log_dir):
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    mc = ModelCheckpoint(f'./{best_model_name}.h5', monitor='val_loss', mode='min', save_best_only=True,  verbose=1)  # callback to save best model

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    cb_list = [es, mc,tensorboard_callback]
    return cb_list