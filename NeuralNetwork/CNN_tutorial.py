import torch
import torch.optim as optim
import torch.nn as nn
from NeuralNetwork import config
from NeuralNetwork.model import Conv2dModel
from data_utilities.cifar10_utilities import get_train_cifar_data
from torch.utils.data import Subset
import time
from utilities.model_utilities import save_model_on_path, test
from utilities.plot_utilities import save_model_train_metric

if __name__ == '__main__':

    train_dataset, test_dataset = get_train_cifar_data()
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    data_size = len(trainloader)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model = Conv2dModel()
    criterion = nn.CrossEntropyLoss()

    accuracy_over_epochs = []
    titles = []
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    loss_per_epoch = []
    title = f'Ep:{config.n_epochs} lr:{config.lr} batch:{config.batch_size}'
    print(title)
    titles.append(title)
    model.train()

    start_training_time = time.time()
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

        # we sum up the results from the epoch we just finished
        train_loss_per_epoch = running_loss / len(trainloader)
        accuracy_per_epoch=100.*correct_train_guess_per_epoch/total

        accuracy_over_epochs.append(accuracy_per_epoch)
        loss_per_epoch.append(train_loss_per_epoch)

        print(f'Epoch {epoch+1}: Train loss={train_loss_per_epoch:.3f} Accuracy:{accuracy_per_epoch:.3f}%')

    training_time_in_secs = time.time()- start_training_time
    print(f'Finished Training in {training_time_in_secs}s...\nSaving model...')
    save_model_on_path(model, config.model_name)

    save_model_train_metric(loss_per_epoch,
                            f'{config.train_loss_plot_title}, time:{training_time_in_secs}, {data_size} samples',
                            config.train_loss_file_name,
                            "Train Loss")
    save_model_train_metric(accuracy_over_epochs, config.train_acc_plot_title, config.train_acc_file_name, "Train accuracy")

    # reloading model
    model = Conv2dModel()
    model.load_state_dict(torch.load(config.model_name))

    eval_losses = []
    eval_accu = []

    epochs = 10
    for epoch in  range(config.n_epochs):
        test_loss, accu = test(model, testloader,criterion)
        eval_losses.append(test_loss)
        eval_accu.append(accu)

    save_model_train_metric(loss_per_epoch, config.test_loss_plot_title, config.test_loss_file_name, "Test Loss")
    save_model_train_metric(accuracy_over_epochs, config.test_acc_plot_title, config.test_acc_file_name, "Test accuracy")