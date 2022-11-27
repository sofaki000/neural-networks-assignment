import torch
import torch.optim as optim
import torch.nn as nn
from NeuralNetwork import config
from NeuralNetwork.model import Conv2dModel
from data_utilities.cifar10_utilities import get_train_cifar_data, get_train_cifar_data_quick
from torch.utils.data import Subset
import time
from utilities.model_utilities import save_model_on_path, test, validate_one_epoch
from utilities.plot_utilities import save_model_train_metric
from utilities.train_utilities_pytorch import EarlyStopping

if __name__ == '__main__':
    # get_train_cifar_data_quick()  #
    train_dataset, test_dataset =  get_train_cifar_data()
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    samples_length = len(trainloader.dataset)

    model = Conv2dModel()
    criterion = nn.CrossEntropyLoss()

    titles = []
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    train_loss_over_epochs = []
    train_accuracy_over_epochs = []

    validation_loss_over_epochs = []
    validation_accuracy_over_epochs = []

    title = f'Ep:{config.n_epochs} lr:{config.lr} batch:{config.batch_size}'
    print(title)
    titles.append(title)
    model.train()
    start_training_time = time.time()

    early_stopping = EarlyStopping(tolerance=5, min_delta=10)
    epoch_stopped = 0

    for epoch in range(config.n_epochs):
        running_loss = 0.0
        correct_train_guess_per_epoch = 0
        total=0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            # calculating train accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_train_guess_per_epoch += (predicted == labels).sum().item()


        # validation when done at current epoch
        with torch.no_grad():
            validation_loss_current_epoch, validation_accuracy_current_epoch =\
                validate_one_epoch(model, testloader, criterion)

        # we sum up the results from the epoch we just finished
        train_loss_current_epoch = running_loss / len(trainloader)
        accuracy_current_epoch = 100. * correct_train_guess_per_epoch / total

        # for all epochs we save results
        train_accuracy_over_epochs.append(accuracy_current_epoch)
        train_loss_over_epochs.append(train_loss_current_epoch)
        validation_loss_over_epochs.append(validation_loss_current_epoch)
        validation_accuracy_over_epochs.append(validation_accuracy_current_epoch)

        # early stopping
        early_stopping(train_loss_current_epoch, validation_loss_current_epoch)

        if early_stopping.early_stop:
            epoch_stopped = epoch
            print("We are at epoch:", epoch)
            break

        print(f'Epoch {epoch+1}: Train loss={train_loss_current_epoch:.3f} Accuracy:{accuracy_current_epoch:.3f}%')

    training_time_in_secs = time.time()- start_training_time

    if epoch_stopped==0:
        # we didnt early stopped nowhere
        epoch_stopped = config.n_epochs

    print(f'Finished Training in {training_time_in_secs:.3f}s at epoch {epoch_stopped}...\nSaving model...')
    save_model_on_path(model, config.model_name)


    # plot train loss and accuracy
    save_model_train_metric(train_loss_over_epochs,
                            f'{config.train_loss_plot_title}, time:{training_time_in_secs:.2f}s, {samples_length} samples, Stopped epoch:{epoch_stopped}',
                            config.train_loss_file_name,
                            "Train Loss")
    save_model_train_metric(train_accuracy_over_epochs, config.train_acc_plot_title,
                            config.train_acc_file_name,
                            "Train accuracy")

    # plot validation loss and accuracy
    save_model_train_metric(validation_accuracy_over_epochs,
                            config.validation_acc_plot_title,
                            config.validation_acc_file_name,
                            "Validation accuracy")
    save_model_train_metric(validation_loss_over_epochs,
                            config.validation_loss_plot_title,
                            config.validation_loss_file_name,
                            "Validation loss")

    # reloading model
    model = Conv2dModel()
    model.load_state_dict(torch.load(config.model_name))

    eval_losses, eval_accu = test(model, testloader, criterion)

    save_model_train_metric(eval_losses, config.test_loss_plot_title, config.test_loss_file_name, "Test Loss")
    save_model_train_metric(eval_accu, config.test_acc_plot_title, config.test_acc_file_name, "Test accuracy")