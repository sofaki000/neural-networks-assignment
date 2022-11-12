import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from plot_utilities import save_multiple_plots_for_different_experiments


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    from torchvision.datasets import CIFAR10
    from torch.utils.data import Subset

    # ds = CIFAR10('~/.torch/data/', train=True, download=True)
    ds = CIFAR10('./data/', train=True, download=True)
    dog_indices, deer_indices, other_indices = [], [], []
    dog_idx, deer_idx = ds.class_to_idx['dog'], ds.class_to_idx['deer']

    for i in range(len(ds)):
        current_class = ds[i][1]
        if current_class == dog_idx:
            dog_indices.append(i)
        elif current_class == deer_idx:
            deer_indices.append(i)
        else:
            other_indices.append(i)
    dog_indices = dog_indices[:int(0.6 * len(dog_indices))]
    deer_indices = deer_indices[:int(0.6 * len(deer_indices))]
    new_dataset = Subset(ds, dog_indices + deer_indices + other_indices)

    transform = transforms.Compose([transforms.ToTensor(),  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    print(f'Number of set: {len(dog_indices)+len(deer_indices)}')
    batch_size = 4
    #trainset = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(new_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    #
    # testset = torchvision.datasets.CIFAR10(root='./data', train=False,  download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    # get some random training images
    # dataiter = iter(trainloader)
    # images, labels = next(dataiter)

    # show images
    #imshow(torchvision.utils.make_grid(images))
    # print labels
    #print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
    net = Net()
    learning_rates = [0.1,0.01,0.001,0.0001]
    criterion = nn.CrossEntropyLoss()
    n_epochs = 2
    losses_over_experiments = []
    titles = []
    for lr_idx in range(len(learning_rates)):
        optimizer = optim.SGD(net.parameters(), lr=learning_rates[lr_idx])
        losses_over_epochs = []
        title = f'Ep:{n_epochs} lr:{learning_rates[lr_idx]} batch:{batch_size}'
        print(title)
        titles.append(title)
        for epoch in range(n_epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                #if i % 2000 == 1999:    # print every 2000 mini-batches
                if i % 2 == 0:  # print every 2 mini-batches
                    losses_over_epochs.append((running_loss / 2000))
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0
        losses_over_experiments.append(losses_over_epochs)
    print('Finished Training')

    experiment_title = "results/Adam/Adam_results"
    save_multiple_plots_for_different_experiments(losses_over_experiments, titles, experiment_title)

    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    # dataiter = iter(testloader)
    # images, labels = next(dataiter)

    # print images
    # imshow(torchvision.utils.make_grid(images))
    # print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
    #
    # net = Net()
    # net.load_state_dict(torch.load(PATH))

    # outputs = net(images)
    #
    # _, predicted = torch.max(outputs, 1)
    #
    # print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))

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