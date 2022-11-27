from keras.optimizers import RMSprop
from sklearn.decomposition import PCA
from Keras.regularization_techniques import config
from Keras.regularization_techniques.training_and_evaluation import train_and_evaluate_model
from Models import get_model, get_model_with_weight_decay, get_model_with_initialized_weights, \
    get_model_with_l1_regularizer, get_model_with_batch_normalization_and_weight_decay, \
    get_model_with_dropout_in_hidden_layers
from data_utilities.sre_dataset_utilities import get_transformed_data


x_train, y_train, x_test, y_test = get_transformed_data(config.load_dataset_number)
if config.use_pca:
    pca = PCA(0.9)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)
    input_features = 16
else:
    input_features = 40


# plain early stopping
# model = get_model(input_features=input_features,num_classes=config.num_classes)
#
# train_and_evaluate_model(x_train, y_train, x_test, y_test,
#                          model,
#                          file_name_loss=config.file_name_loss_plain_early_stopping ,
#                          file_name_acc=config.file_name_acc_plain_early_stopping,
#                          title_acc=config.title_acc_plain_early_stopping,
#                          title_loss=config.title_loss_plain_early_stopping,
#                          best_model_name="best_model_early_stop")
#
# # early stopping with batch norm and weight decay
# model = get_model_with_batch_normalization_and_weight_decay(input_features=input_features,
#                                                             num_classes=config.num_classes,
#                                                             optimizer = RMSprop,
#                                                             lr =config.lr,
#                                                             weight_decay=config.weight_decay)
#
# train_and_evaluate_model(x_train, y_train, x_test, y_test,
#                          model,
#                          file_name_loss=config.file_name_loss_batch_norm_weight_dec ,
#                          file_name_acc=config.file_name_acc_batch_norm_weight_dec,
#                          title_acc=config.title_acc_batch_norm_weight_dec,
#                          title_loss=config.title_loss_batch_norm_weight_dec,
#                          lr=0.01,
#                          best_model_name="best_model_batch_norm_and_weight_decay")
#
#
# # early stopping with l1 regularizer
# model = get_model_with_l1_regularizer(input_features=input_features,num_classes=config.num_classes)
# train_and_evaluate_model(x_train, y_train, x_test, y_test,
#                          model,
#                          file_name_loss=config.file_name_loss_l1 ,
#                          file_name_acc=config.file_name_acc_l1,
#                          title_acc=config.title_acc_l1,
#                          title_loss=config.title_loss_l1,
#                          best_model_name="best_model_l1_regul")
#
# # early stopping with DROPOUT layer in hidden layers
# model = get_model_with_dropout_in_hidden_layers(input_features=input_features,
#                                                 num_classes=config.num_classes,
#                                                 dropout=config.dropout)
#
# train_and_evaluate_model(x_train, y_train, x_test, y_test,
#                          model,
#                          file_name_loss=config.file_name_loss_dropout  ,
#                          file_name_acc=config.file_name_acc_dropout ,
#                          title_acc=config.title_acc_dropout ,
#                          title_loss=config.title_loss_dropout ,
#                          best_model_name="best_model_dropout")
#
# # early stopping with weight decay
# model = get_model_with_weight_decay(input_features=input_features,num_classes=config.num_classes)
# train_and_evaluate_model(x_train, y_train, x_test, y_test,
#                          model,
#                          file_name_loss=config.file_name_loss_weight_decay ,
#                          file_name_acc=config.file_name_acc_weight_decay,
#                          title_acc=config.title_acc_weight_decay,
#                          title_loss=config.title_loss_weight_decay,
#                          best_model_name="best_model_weight_decay")

# early stopping with weight decay and initialized weights
model = get_model_with_initialized_weights(input_features=input_features,num_classes=config.num_classes)
train_and_evaluate_model(x_train, y_train, x_test, y_test,
                         model,
                         file_name_loss= config.file_name_loss_weight_decay_and_initialized_weights ,
                         file_name_acc= config.file_name_acc_weight_decay_and_initialized_weights,
                         title_acc= config.title_acc_weight_decay_and_initialized_weights,
                         title_loss= config.title_loss_weight_decay_and_initialized_weights,
                         best_model_name="best_model_weight_decay_initialzied_weights")