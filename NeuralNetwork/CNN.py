from time import strftime, gmtime

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder

from dataHandler import get_raw_data
from plot_utilities import save_multiple_plots_for_different_experiments


class CNN(nn.Module):
    def __init__(self,batch_size,number_of_output_classes):
        super().__init__( )
        self.conv1 = nn.Conv2d(1, 6, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 1)
        # self.fc1 = nn.Linear(16 * batch_size * 40, 120)
        self.fc1 = nn.Linear(16 * batch_size * 10, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, number_of_output_classes)


    def forward(self, x):
        x = F.relu(self.conv1(x)) #  self.pool(
        x =  F.relu(self.conv2(x))#self.pool(
        x = torch.flatten(x, 1).flatten() # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x)


transform = transforms.Compose([transforms.ToTensor(),  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
#
# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

X_train, y_train, X_test, y_test = get_raw_data()

#TODO: try different encoding
enc = OneHotEncoder()
encoded_labels = enc.fit_transform(y_train.reshape(len(y_train), 1))

number_of_output_classes = encoded_labels.shape[1]

criterion = nn.CrossEntropyLoss()

learning_rates = [0.1,0.01,0.001,0.0001]
n_epochs = 10
cnn = CNN(batch_size=2, number_of_output_classes=number_of_output_classes)

losses_over_experiments = []
titles = []

optimizers = [ optim.SGD, optim.Adam]

for optimizer_idx in range(len(optimizers)):
    for k in range(len(learning_rates)):
        # optimizer = optim.SGD(cnn.parameters(), lr=learning_rates[k], momentum=0.9)
        optimizer = optimizers[optimizer_idx](cnn.parameters(), lr=learning_rates[k])
        losses_over_epochs = []
        title = f'Ep:{n_epochs} lr:{learning_rates[k]}'
        print(title)
        titles.append(title)

        for epoch in range(n_epochs):  # loop over the dataset multiple times
            epoch_loss = 0.0
            for i in range(X_train.shape[0]):
                # batch_data = X_train[(i+1):(i+batch_size)]
                # batch_labels = y_train[(i+1):(i+batch_size)]
                batch_data = X_train[i]
                batch_labels = encoded_labels[i]
                # get the inputs; data is a list of [inputs, labels]
                inputs= torch.tensor(batch_data )
                inputs = inputs[None,None, :]
                labels = torch.tensor(batch_labels.toarray(),dtype=torch.float32)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = cnn(inputs)
                loss = criterion(outputs, labels.squeeze())
                loss.backward()
                optimizer.step()

                # print statistics
                epoch_loss += loss.item()
            if epoch % 2 == 0:    # print every 2 mini-batches
                loss_per_batch = (epoch_loss / 2)
                print(f'Epoch: {epoch} loss: {loss_per_batch:.3f}')
                losses_over_epochs.append(loss_per_batch)
                epoch_loss = 0.0
        losses_over_experiments.append(losses_over_epochs)


    if optimizer_idx==0: # prwta to SGD
        experiment_title = "SGD_results"
    else:
        experiment_title="Adam_results"
    save_multiple_plots_for_different_experiments(losses_over_experiments, titles, experiment_title)


# we will plot the experiment results



print('Finished Training')
# PATH = './cifar_net.pth'
# torch.save(net.state_dict(), PATH)

# correct = 0
# total = 0
# # since we're not training, we don't need to calculate the gradients for our outputs
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         # calculate outputs by running images through the network
#         outputs = net(images)
#         # the class with the highest energy is what we choose as prediction
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
# print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')