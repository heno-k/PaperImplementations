# This isn't based off a paper, but it is imperative to understand the fundamentals 
# of convolutional neural networks

# I tried to implement AlexNet but because ImageNet data set wasn't available (Only providing max of 3x64x64 sized images)
# I wouldn't be able to use the architecture unless I upscale the images. I also don't have a second GPU to split the training
# like they did in the paper... Possibly revisit in the future

#Revisit data augmentation techniques?

# https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module
# Base class for all nn models

import os.path
from os import path
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

# Path to save model
Save_Trained_Model = True
Model_Path = "C:/Users/Henok/source/repos/PaperImplementations/SavedModels/ConvNet/ConvNet.pth"

# Train new model or load saved trained model
Load_Saved_Model = False

# Define the hyperparameters
NumHiddenLayers = 6
ActivationFunction = "ReLu"
MiniBatchSize = 4
NumEpochs = 2
DropOutUsed = False
LearningRate = 0.001    # Learning rate of .01 results in gradient not converging to local minima
Momentum = .9


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # No soft max layer because it is called in CrossEntropyLoss

def PrintHyperParameters():
    #TODO: Understand Learning Rate, Momentum and Dropout

    print("Convolutional Neural Network:\nHyperparameters:\n")
    print("   Number of Hidden layers is: %d" % NumHiddenLayers)
    print("   Learning Rate is: %f" % LearningRate)
    print("   Momentum is: %f" % Momentum)
    print("   Activation Function is: " + ActivationFunction)
    print("   Minibatch size is: %d" % MiniBatchSize)
    print("   Number of Epochs is: %d" % NumEpochs)
    print(f"   Dropout used is: {DropOutUsed!s}\n")

def LoadImages():
    print("Obtaining the training and testing data using Pytorch dataloader functions\n")

    transform = transforms.Compose(
    [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Extract the CIFAR10 training dataset from torchvision dataset
    trainset  = torchvision.datasets.CIFAR10(root='D:\Pictures\CIFAR10_Dataset' , train=True,
                                            download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=MiniBatchSize,
                                            shuffle=True, num_workers=0)
    
    # Extract the CIFAR10 testing dataset from torchvision dataset
    testset = torchvision.datasets.CIFAR10(root='D:\Pictures\CIFAR10_Dataset', train=False, 
                                        download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=MiniBatchSize,
                                            shuffle=True, num_workers=0)

    #classes = ('plane', 'car', 'bird', 'cat',
    #        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader

def TrainNet(net, trainloader, criterion, optimizer, device):
    #training the network
    print("\nBeginning training using the neural net\n")
    for epoch in range(NumEpochs): # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            #get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            #zero the parameter gradients
            optimizer.zero_grad()

            #forward + backward
            outputs = net(inputs)
            # Compute the Cross-Entropy Loss of the outputs of NN
            loss = criterion(outputs, labels)
            # Computes the gradient of the loss and stores in the tensors
            loss.backward()
            # Uses computed gradient from previous step to update the parameters of the NN
            # Essentially w = w - lr * gradient 
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i+1, running_loss /2000))
                running_loss = 0.0

    # Save the trained model so that it is not necessary to train again
    if(Save_Trained_Model):
        print("\nSaving the Trained Model")
        torch.save(net.state_dict(), Model_Path)

    print('Finished Training\n')
    return net

def TestNet(net, testloader, device):
    print("Beginning testing using the neural net")
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            #get the test images; data is a list of [inputs, labels]
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy of the network on the 10000 test images: %d %%' % (
    100*correct / total))
    print("Testing complete\n")

def main():
    #Print the hyper parameters for the convolutional neural network
    PrintHyperParameters()

    # Switch to GPU to train and test quicker
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    net = Model()
    net = net.to(device)

    # Gather the training and testing data using pytorch
    trainloader, testloader = LoadImages()

    if(Load_Saved_Model):
        print("\nAttempting to find existing model")
        Saved_Model_Found = path.exists(Model_Path)

    if(Load_Saved_Model == False or not Saved_Model_Found):
        if(Load_Saved_Model == True):
            print("Model was not found... Need to retrain")
        #TODO: Understand criterion and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=LearningRate, momentum=Momentum)

        # Train the Neural Net
        net = TrainNet(net, trainloader, criterion, optimizer, device)
    else:
        print("Trained Model found... loading for testing\n")
        net.load_state_dict(torch.load(Model_Path))
    
    # Test the Neural Net
    TestNet(net, testloader, device)