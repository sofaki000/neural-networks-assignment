

import torchvision.transforms as transforms
from keras.utils import np_utils
from sklearn.decomposition import PCA

from NearestNeighbor.data_handler import load_datasets

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