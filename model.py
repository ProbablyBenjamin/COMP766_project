"""File to define and load a pre-trained resNet50 model"""

import torchvision
import torch
import matplotlib.pyplot as plt
from preprocess import get_train_test_datasets
from load_pretrained_models import load_model

def get_model(lr):
    resnet50 = load_model('resnet50')

    for param in resnet50.parameters():
        param.requires_grad = False

    print('base layers now NON-Trainable')


    num_classes = 16 #training on 16-class ImageNet like Geirhos paper.


    resnet50.fc = torch.nn.Linear(resnet50.fc.in_features, num_classes)

    for param in resnet50.fc.parameters():
        param.requires_grad = True


    #train only the fully-connected layers

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(resnet50.fc.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4) #change after 30 epochs 

    return resnet50, criterion, optimizer

def save_model(epoch, model, path='model.pth'):
    """Save the model state dictionary at the given path."""
    torch.save(model.state_dict(), path)
    print(f'Model saved at epoch {epoch} to {path}')

def load_model_weights(model, path):
    """Load weights into the model."""
    model.load_state_dict(torch.load(path))
    model.eval()  # Set the model to evaluation mode
    print(f'Model loaded from {path}')


