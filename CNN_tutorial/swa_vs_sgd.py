
import torch
import torchvision
from torch.utils.data import Dataset
import torch.nn as nn
import torchvision.transforms as transforms
import time
from plot_utilities import save_model_train_and_test_loss_plot


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

start_time = time.time()
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=all_transforms, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=all_transforms, download=False)

time_tool_to_load_data = f'Time took to load data:{time.time()-start_time:.3f}s'
print(time_tool_to_load_data)
# Instantiate loader objects to facilitate processing
batch_size = 64
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,  batch_size = batch_size, shuffle = True)

num_classes = 10
learning_rate = 0.001
n_epochs = 10

def train_with_model_and_scheduler(model,scheduler, optimizer, experiment_title, model_path):
    model.to(device)
    losses_per_epoch = []
    print(experiment_title)
    for epoch in range(n_epochs):
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
            scheduler.step()
        losses_per_epoch.append(loss.item())
        # if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}')
    # training over epochs finished. We save models_optimizers
    print("Training finished. Saving model...")
    torch.save(model.state_dict(), model_path)
    return losses_per_epoch

def train_model(model, optimizer, experiment_title, model_path):
    model.to(device)
    losses_per_epoch = []
    print(experiment_title)
    for epoch in range(n_epochs):
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
        # if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}')
    # training over epochs finished. We save models_optimizers
    print("Training finished. Saving model...")
    torch.save(model.state_dict(), model_path)
    return losses_per_epoch


def test(model_to_test, testloader):
    correct = 0
    total = 0
    test_losses = []
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model_to_test(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(predicted.float(), labels.float())
            test_losses.append(loss)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = (f'Accuracy of the network on the {total} test images: {100 * correct // total} %')
    print(accuracy)
    return test_losses, accuracy

f = open("swa_experiments.txt", "a")
f.write(f'Epochs:{n_epochs}')
model_no_scheduler_path = f'SWA_models/model_no_scheduler.pth'
model_with_scheduler_path = f'SWA_models/model_with_scheduler.pth'

criterion = nn.CrossEntropyLoss()
model1 = ConvNeuralNet(num_classes)
model1.to(device)
optimizer = torch.optim.SGD(model1.parameters(), lr=0.001, momentum=0.9)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)


# we train a model with only optimizer

start_time = time.time()
losses_over_epochs_const_lr = train_model(model1, optimizer, "SGD with const lr", model_path=model_no_scheduler_path)
train_time1 = f'Train time for sgd with const lr {(time.time() - start_time):.3f}s\n'
f.write(train_time1)

# we train a model with scheduler
start_time = time.time()
model2 = ConvNeuralNet(num_classes)
losses_with_cyclic_lr =train_with_model_and_scheduler(model2, scheduler, optimizer, experiment_title="SGD with scheduler", model_path=model_with_scheduler_path)
train_time2 = f'Train time for sgd with cyclic lr {(time.time() - start_time):.3f}s\n'
f.write(train_time2)

# test both models

# Const lr
model = ConvNeuralNet(num_classes)
model.load_state_dict(torch.load(model_no_scheduler_path))
test_losses_without_scheduler, accuracy = test(model,test_loader)
save_model_train_and_test_loss_plot(train_losses=losses_over_epochs_const_lr, test_losses=test_losses_without_scheduler, title="SGD with const lr", file_name="sgd_no_scheduler")
f.write(f'SGD with const lr: {accuracy}\n')

# cyclic lr
model = ConvNeuralNet(num_classes)
model.load_state_dict(torch.load(model_with_scheduler_path))
test_losses_with_scheduler, accuracy  = test(model,test_loader)
f.write(f'SGD with cyclic lr: {accuracy}\n')
save_model_train_and_test_loss_plot(train_losses=losses_with_cyclic_lr, test_losses=test_losses_with_scheduler, title="SGD with cyclic lr", file_name="sgd_with_scheduler")

