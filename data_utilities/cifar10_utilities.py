

import torchvision.transforms as transforms
from keras.utils import np_utils
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset

cifar_data_fila_path = 'C:/Users/Lenovo/Desktop/νευρωνικά δίκτυα/neural-networks-assignment/CNN/data/'
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def get_cifar():
    X_train, y_train, X_test, y_test = load_datasets('../data')
    X_train = X_train.reshape(-1,3072)
    X_test = X_test.reshape(-1,3072)
    pca = PCA(0.9)
    train_img_pca = pca.fit_transform(X_train)
    test_img_pca = pca.transform(X_test)
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    return X_train, X_test,y_train,y_test


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

def load_datasets(filepath='./data', load_for_development=True):
    transform = transforms.Compose([ transforms.Resize((32, 32)),  transforms.ToTensor(),
                                     transforms.Normalize((0 ,), (1,)),
                                     nn.Flatten()])

    trainset = torchvision.datasets.CIFAR10(root=filepath, train=True, transform=transform, download=False)
    testset = torchvision.datasets.CIFAR10(root=filepath, train=False, transform=transform, download=False)

    # Define indexes and get the subset random sample of each.
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=True)

    # Convert data to tensors.
    x_test = []
    y_test = []
    x_train = []
    y_train = []

    # if load_for_development: # we dont load the whole cifar
    #     counter =0
    #     for idx, (data, tar) in enumerate(test_dataloader):
    #         x_test = data.squeeze()
    #         y_test = tar.squeeze()
    #         counter = counter+1
    #
    #         if counter==2000:
    #             break
    #     counter = 0
    #     for idx, (data, tar) in enumerate(train_dataloader):
    #         x_train = data.squeeze()
    #         y_train = tar.squeeze()
    #         if counter==1000:
    #             break
    # else:
    if True:
        for idx, (data, tar) in enumerate(test_dataloader):
            x_test = data.squeeze()
            y_test = tar.squeeze()
        for idx, (data, tar) in enumerate(train_dataloader):
            x_train = data.squeeze()
            y_train = tar.squeeze()

    x_test = x_test.clone().detach()#.requires_grad_(True)
    y_test = y_test.clone().detach()#.requires_grad_(True)
    x_train = x_train.clone().detach()#.requires_grad_(True)
    y_train = y_train.clone().detach()#.requires_grad_(True)
    return x_train, y_train, x_test, y_test



def get_train_cifar_data_quick(train_data_percentage = 0.7):
    from torchvision.datasets import CIFAR10
    from torch.utils.data import Subset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    #trainset = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform)

    # ds = CIFAR10('~/.torch/data/', train=True, download=True)
    ds = CIFAR10(cifar_data_fila_path, train=True, download=False,transform=transform)
    dog_indices, deer_indices, other_indices = [], [], []
    plane_indices, car_indices  = [], []
    frog_indices, ship_indices  = [], []
    dog_idx, deer_idx = ds.class_to_idx['dog'], ds.class_to_idx['deer']
    plane_idx, car_idx = ds.class_to_idx['airplane'], ds.class_to_idx['automobile']
    frog_idx, ship_idx = ds.class_to_idx['frog'], ds.class_to_idx['ship']

    for i in range(len(ds)):
        current_class = ds[i][1]
        if current_class == dog_idx:
            dog_indices.append(i)
        elif current_class == deer_idx:
            deer_indices.append(i)
        elif current_class == plane_idx:
            plane_indices.append(i)
        elif current_class == car_idx:
            car_indices.append(i)
        elif current_class == frog_idx:
            frog_indices.append(i)
        elif current_class == ship_idx:
            ship_indices.append(i)
        else:
            other_indices.append(i)

    dog_split_indices = (int)(len(dog_indices) * train_data_percentage)
    # dog_indices_train = dog_indices[:int(0.5* len(dog_indices))]
    # dog_indices_test = dog_indices[int(0.5 * len(dog_indices)):int(0.8 * len(dog_indices))]
    dog_indices_train = dog_indices[:dog_split_indices]
    dog_indices_test = dog_indices[dog_split_indices:]

    plane_split_indices = (int)(len(plane_indices) * train_data_percentage)
    plane_indices_train = deer_indices[:plane_split_indices]
    plane_indices_test = deer_indices[plane_split_indices:]


    deer_split_indices = (int)(len(deer_indices) * train_data_percentage)
    deer_indices_train = deer_indices[:deer_split_indices]
    deer_indices_test = deer_indices[deer_split_indices:]

    ship_split_indices = (int)(len(ship_indices) * train_data_percentage)
    ship_indices_train = ship_indices[:ship_split_indices]
    ship_indices_test = ship_indices[ship_split_indices:]


    frog_split_indices = (int)(len(frog_indices) * train_data_percentage)
    frog_indices_train = frog_indices[:frog_split_indices]
    frog_indices_test = frog_indices[frog_split_indices:]


    car_split_indices = (int)(len(car_indices) * train_data_percentage)
    car_indices_train = car_indices[:car_split_indices]
    car_indices_test = car_indices[car_split_indices:]

    train_dataset = Subset(ds, dog_indices_train + deer_indices_train )# + other_indices
    # test dataset
    test_dataset = Subset(ds,dog_indices_test + deer_indices_test )  # + other_indices

    print(f'Number of set: {len(dog_indices) + len(deer_indices)}')

    return train_dataset,test_dataset

def get_train_cifar_data(train_data_percentage = 0.7):
    from torchvision.datasets import CIFAR10
    from torch.utils.data import Subset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    #trainset = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform)

    # ds = CIFAR10('~/.torch/data/', train=True, download=True)
    ds = CIFAR10(cifar_data_fila_path, train=True, download=False,transform=transform)
    dog_indices, deer_indices, other_indices = [], [], []
    plane_indices, car_indices  = [], []
    frog_indices, ship_indices  = [], []
    dog_idx, deer_idx = ds.class_to_idx['dog'], ds.class_to_idx['deer']
    plane_idx, car_idx = ds.class_to_idx['airplane'], ds.class_to_idx['automobile']
    frog_idx, ship_idx = ds.class_to_idx['frog'], ds.class_to_idx['ship']

    for i in range(len(ds)):
        current_class = ds[i][1]
        if current_class == dog_idx:
            dog_indices.append(i)
        elif current_class == deer_idx:
            deer_indices.append(i)
        elif current_class == plane_idx:
            plane_indices.append(i)
        elif current_class == car_idx:
            car_indices.append(i)
        elif current_class == frog_idx:
            frog_indices.append(i)
        elif current_class == ship_idx:
            ship_indices.append(i)
        else:
            other_indices.append(i)

    dog_split_indices = (int)(len(dog_indices) * train_data_percentage)
    # dog_indices_train = dog_indices[:int(0.5* len(dog_indices))]
    # dog_indices_test = dog_indices[int(0.5 * len(dog_indices)):int(0.8 * len(dog_indices))]
    dog_indices_train = dog_indices[:dog_split_indices]
    dog_indices_test = dog_indices[dog_split_indices:]

    plane_split_indices = (int)(len(plane_indices) * train_data_percentage)
    plane_indices_train = deer_indices[:plane_split_indices]
    plane_indices_test = deer_indices[plane_split_indices:]


    deer_split_indices = (int)(len(deer_indices) * train_data_percentage)
    deer_indices_train = deer_indices[:deer_split_indices]
    deer_indices_test = deer_indices[deer_split_indices:]

    ship_split_indices = (int)(len(ship_indices) * train_data_percentage)
    ship_indices_train = ship_indices[:ship_split_indices]
    ship_indices_test = ship_indices[ship_split_indices:]


    frog_split_indices = (int)(len(frog_indices) * train_data_percentage)
    frog_indices_train = frog_indices[:frog_split_indices]
    frog_indices_test = frog_indices[frog_split_indices:]


    car_split_indices = (int)(len(car_indices) * train_data_percentage)
    car_indices_train = car_indices[:car_split_indices]
    car_indices_test = car_indices[car_split_indices:]

    train_dataset = Subset(ds, dog_indices_train + deer_indices_train + car_indices_train+frog_indices_train+ship_indices_train +plane_indices_train)# + other_indices
    # test dataset
    test_dataset = Subset(ds,dog_indices_test + deer_indices_test + car_indices_test + frog_indices_test +ship_indices_test +plane_indices_test)  # + other_indices

    print(f'Number of set: {len(dog_indices) + len(deer_indices)+len(frog_indices_train)+len(car_indices_train)+len(ship_indices_train)+len(plane_indices_train)}')

    return train_dataset,test_dataset