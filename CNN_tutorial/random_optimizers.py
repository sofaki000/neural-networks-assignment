
import torch
import torchvision
from torch.utils.data import Dataset
import torch.nn as nn
import torchvision.transforms as transforms

from plot_utilities import save_multiple_plots_for_different_experiments

class ConvNeuralNet(nn.Module):
    #  Determine what layers and their order in CNN object
    def __init__(self, num_classes):
        super(ConvNeuralNet, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(1600, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    # Progresses data across layers
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.max_pool1(out)

        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool2(out)

        out = out.reshape(out.size(0), -1)  # flatten everything except batch size

        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out  # batch_size * output_classes


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Use transforms.compose method to reformat images for modeling,
# and save to variable all_transforms for later use
all_transforms = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
# # Create Training dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=all_transforms, download=True)
# Create Testing dataset
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=all_transforms, download=False)

# Instantiate loader objects to facilitate processing
batch_size = 64
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,  batch_size = batch_size, shuffle = True)

num_classes = 10
learning_rate = 0.001
n_epochs = 1

# Set Loss function with criterion
criterion = nn.CrossEntropyLoss()

# Set optimizer with optimizer
plain_sgd = lambda model: torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.000, momentum=0.000)
optimizer_sgd =  lambda model:torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9)
optimizer_sgd_without_momentum = lambda model: torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.0)
optimizer_adam =  lambda model: torch.optim.Adam(model.parameters(), lr=learning_rate)

optimizers = [plain_sgd, optimizer_sgd, optimizer_sgd_without_momentum, optimizer_adam]


losses_per_experiment = []
experiment_titles = []

models = []

for optimizer_idx in range(len(optimizers)):  # trying different optimizers
    model = ConvNeuralNet(num_classes)
    model.to(device)
    losses_per_epoch = []
    optimizer = optimizers[optimizer_idx](model)
    experiment_title = f'E:{n_epochs} Opt:{optimizer.__class__.__name__} B:{batch_size} lr:{learning_rate} Cl:{num_classes}'
    print(experiment_title)
    experiment_titles.append(experiment_title)
    for epoch in range(n_epochs):
        # Load in the data in batches using the train_loader object
        for i, (images, labels) in enumerate(train_loader):
            # Forward pass
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses_per_epoch.append(loss.item())
        #if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}')
    losses_per_experiment.append(losses_per_epoch)
    # training over epochs finished. We save models_optimizers
    PATH = f'models_optimizers/optimizer_model_{optimizer_idx}.pth'
    torch.save(model.state_dict(), PATH)


save_multiple_plots_for_different_experiments(losses_per_experiment, experiment_titles, "results")
def test(model, testloader):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = (f'Accuracy of the network on the {total} test images: {100 * correct // total} %')
    print(accuracy)
    return accuracy

f = open("test_results_optimizer_experiments.txt", "a")
titles = []
titles.append("SGD, weight decay=0.0,  momentum=0.0")
titles.append("SGD, weight decay=0.005, no momentum=0.9")
titles.append("SGD, no weight decay=0.005, no momentum=0.0")
titles.append("Adam")
# we test the models_optimizers we trained
for optimizer_idx in range(len(optimizers)):
    PATH = f'models_optimizers/optimizer_model_{optimizer_idx}.pth'
    model = ConvNeuralNet(num_classes)
    model.load_state_dict(torch.load(PATH))
    acc = test(model, test_loader)
    f.write(f'{titles[optimizer_idx]}: {acc}\n')
f.close()




