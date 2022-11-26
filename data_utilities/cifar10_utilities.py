

import torchvision.transforms as transforms

def get_train_cifar_data():
    from torchvision.datasets import CIFAR10
    from torch.utils.data import Subset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    #trainset = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform)

    # ds = CIFAR10('~/.torch/data/', train=True, download=True)
    ds = CIFAR10('C:/Users/Lenovo/Desktop/νευρωνικά δίκτυα/neural-networks-assignment/CNN/data/', train=True, download=False,transform=transform)
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

    dog_indices_train = dog_indices[:int(0.2* len(dog_indices))]
    deer_indices_train = deer_indices[:int(0.2 * len(deer_indices))]
    train_dataset = Subset(ds, dog_indices_train + deer_indices_train )# + other_indices

    # test dataset
    dog_indices_test = dog_indices[int(0.2 * len(dog_indices)):int(0.3 * len(dog_indices))]
    deer_indices_test = deer_indices[int(0.2 * len(deer_indices)):int(0.3 * len(dog_indices))]
    test_dataset = Subset(ds, dog_indices_test + deer_indices_test)  # + other_indices

    print(f'Number of set: {len(dog_indices) + len(deer_indices)}')
    return train_dataset,test_dataset