import os
from model import Net
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from dataset import CustomImageDataset
import PIL
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
import torchsummary

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

training_data = CustomImageDataset("./labels/labels.csv", "./images", transform=transforms.Compose([transforms.Resize(128, antialias=True), transforms.functional.rgb_to_grayscale]))
train_dataloader = DataLoader(training_data, batch_size=16, shuffle=True)

print('Beginning training...')

net = Net()
net.to(device)

torchsummary.summary(net, input_size=(1, 128, 128))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(25):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2:.3f}')
        running_loss = 0.0

print('Finished training...')

dirCheck = os.path.exists("./models")
if not dirCheck:
    os.makedirs("./models")

PATH = './models/test.pth'
torch.save(net.state_dict(), PATH)

print('Saved model!')
print('Evaluating accuracy on dataset...')

correct = 0
total = 0
classes = pd.read_csv("./labels/classes.csv")

with torch.no_grad():
    for data in train_dataloader:
        images, labels = data[0].to(device), data[1].to(device)
        # calculate outputs by running images through the network
        outputs = net(images)
        print('actual: ', ' '.join(classes.iloc[labels[j].item(), 1] for j in range(len(labels))))

        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        print('prediction: ', ' '.join(classes.iloc[predicted[j].item(), 1] for j in range(len(labels))))

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network: {100 * correct // total} %')
