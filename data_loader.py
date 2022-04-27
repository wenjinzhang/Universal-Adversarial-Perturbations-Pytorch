import torch
import torchvision
import torchvision.transforms as transforms
import os
import pickle

import numpy as np

device = torch.device('cpu')
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),  # Transform to tensor
        transforms.Normalize((0.5,), (0.5,))  # Min-max scaling to [-1, 1]
    ])

    data_dir = os.path.join("./", 'fashion_mnist')
    print('Data stored in %s' % data_dir)
    trainset = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=False)
    return trainloader,testloader


def load_CSI_data():
    data_dir = os.path.join("./", 'CSI_dataset', 'processed_dataset.pk')
    print('Data stored in %s' % data_dir)

    file = open(data_dir, 'rb')
    (X_train, Y_train, X_valid, Y_valid) = pickle.load(file)

    Y_train = np.argmax(Y_train, axis=1)
    Y_valid = np.argmax(Y_valid, axis=1)

    Y_train = torch.Tensor(Y_train).type(torch.LongTensor)
    Y_valid = torch.Tensor(Y_valid).type(torch.LongTensor)
    
    trainset = torch.utils.data.TensorDataset(torch.Tensor(X_train), Y_train)
    testset = torch.utils.data.TensorDataset(torch.Tensor(X_valid), Y_valid)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
    return trainloader, testloader

if __name__ == '__main__':
    trainloader, testloader = load_CSI_data()
    (x,y) = trainloader.dataset[0]
    print(x.size())
    print(y)