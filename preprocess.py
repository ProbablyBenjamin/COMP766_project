"""Preprocesses folders with 16-class Imagenet folders and separate them into training, validation and test. """


import torch
import torchvision

def get_train_test_datasets(train_percent, validation_percent, batch_size):
        
    weights=torchvision.models.ResNet50_Weights.DEFAULT

    dataset = torchvision.datasets.ImageFolder('/Users/benlo/repo/COMP766_project/ImageNetDiffused', transform=weights.transforms())


    train_size = int(train_percent*len(dataset))
    validation_size = int(validation_percent*len(dataset))
    test_size = len(dataset) - train_size - validation_size
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


    return train_loader, validation_loader, test_loader