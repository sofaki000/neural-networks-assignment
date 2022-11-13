import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import *
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import torchvision
import numpy as np
import torchvision.models as models
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset


def get_dataset_for_developing(transform):
    ds = CIFAR10('./data/', train=True, download=False,transform=transform)
    dog_indices, deer_indices, airplane_indices,automobile_indices,ship_indices,truck_indices,bird_indices,frog_indices,horse_indices,cat_indices = [], [], [],[], [], [],[], [], [],[]
    dog_idx, deer_idx = ds.class_to_idx['dog'], ds.class_to_idx['deer']
    airplane_idx, automobile_idx = ds.class_to_idx['airplane'], ds.class_to_idx['automobile']
    ship_idx, truck_idx = ds.class_to_idx['ship'], ds.class_to_idx['truck']
    bird_idx, frog_idx = ds.class_to_idx['bird'], ds.class_to_idx['frog']
    horse_idx, cat_idx = ds.class_to_idx['horse'], ds.class_to_idx['cat']

    for i in range(len(ds)):
        current_class = ds[i][1]
        if current_class == dog_idx:
            dog_indices.append(i)
        elif current_class == deer_idx:
            deer_indices.append(i)
        elif current_class==airplane_idx:
            airplane_indices.append(i)
        elif current_class==automobile_idx:
            automobile_indices.append(i)
        elif current_class == ship_idx:
            ship_indices.append(i)
        elif current_class == truck_idx:
            truck_indices.append(i)
        elif current_class == bird_idx:
            bird_indices.append(i)
        elif current_class == frog_idx:
            frog_indices.append(i)
        elif current_class == horse_idx:
            horse_indices.append(i)
        elif current_class == cat_idx:
            cat_indices.append(i)

    dog_indices = dog_indices[:int(0.5 * len(dog_indices))]
    deer_indices = deer_indices[:int(0.5 * len(deer_indices))]
    airplane_indices= airplane_indices[:int(0.5 * len(airplane_indices))]
    automobile_indices = automobile_indices[:int(0.5 * len(automobile_indices))]
    ship_indices = ship_indices[:int(0.5 * len(ship_indices))]
    truck_indices= truck_indices[:int(0.5 * len(truck_indices))]
    bird_indices = bird_indices[:int(0.5 * len(bird_indices))]
    frog_indices = frog_indices[:int(0.5 * len(frog_indices))]
    horse_indices = horse_indices[:int(0.5 * len(horse_indices))]
    cat_indices = cat_indices[:int(0.5 * len(cat_indices))]
    new_dataset = Subset(ds, dog_indices + deer_indices + airplane_indices+ automobile_indices+ship_indices+truck_indices+bird_indices+frog_indices+horse_indices+cat_indices)
    return new_dataset

def load_datasets():
    # Transforms for the image.
    # transform = transforms.Compose([
    #     transforms.Grayscale(), transforms.Resize((32, 32)),  transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), nn.Flatten()
    # ])
    transform = transforms.Compose([ transforms.Resize((32, 32)),  transforms.ToTensor(), transforms.Normalize((0 ,), (1,)), nn.Flatten()])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=False)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=False)

    # trainset= get_dataset_for_developing(transform)
    # testset = get_dataset_for_developing(transform)
    # Define indexes and get the subset random sample of each.
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=True)

    # Convert data to tensors. This could be made faster.
    x_test = []
    y_test = []
    for idx, (data, tar) in enumerate(test_dataloader):
        x_test = data.squeeze()
        y_test = tar.squeeze()

    x_train = []
    y_train = []
    for idx, (data, tar) in enumerate(train_dataloader):
        x_train = data.squeeze()
        y_train = tar.squeeze()

    x_test = x_test.clone().detach()#.requires_grad_(True)
    y_test = y_test.clone().detach()#.requires_grad_(True)
    x_train = x_train.clone().detach()#.requires_grad_(True)
    y_train = y_train.clone().detach()#.requires_grad_(True)
    return x_train, y_train, x_test, y_test

def knn(x_train, y_train, x_test, k,  log_interval=100, log=True):

    # Get the amount of images, training images, and image size.
    num_images = x_test.shape[0]
    num_train = y_train.shape[0]
    img_size = x_test.shape[1]

    y_test = torch.zeros((num_images),  dtype=torch.float)

    # For each of the images in the test set
    for test_index in range(0, num_images):

        # Get the image and calculate the distance to every item in the trainset
        test_image = x_test[test_index]
        distances = torch.norm(x_train - test_image, dim=1)

        # Get the top k indexes and get the most used index between them all
        indexes = torch.topk(distances, k, largest=False)[1]
        classes = torch.gather(y_train, 0, indexes)
        mode = int(torch.mode(classes)[0])

        # Save the test value in the index.
        y_test[test_index] = mode

        # Logging since with large sets it may be helpful
        if log:
            if test_index % log_interval == 0:
                print("Currently predicting at test_index = %d" % test_index)

    return y_test

#
# print("Loading data from folders.")
# x_train, y_train, x_test, y_test = load_datasets()
# print("Loaded train and test with sizes: %s, %s" % (str(x_train.shape), str(x_test.shape)))
# pred = knn(x_train, y_train, x_test, k=1)
# correct = pred.eq(y_test.view_as(pred)).sum()
# print("Correct predictions: %d/%d, Accuracy: %f" % (correct, y_test.shape[0], 100. * correct / y_test.shape[0]))