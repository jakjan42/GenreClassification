import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
# from tqdm import tqdm
import pandas as pd
import os
from torchvision.io import read_image



def get_classes(audio_dir=None, feature_file=None):
    if audio_dir is not None:
        if not os.path.isdir(audio_dir):
            raise Exception("audio directory does not exit")
        return os.listdir(audio_dir)
    if feature_file is not None:
        labels = pd.read_csv(feature_file, usecols=["label"])
        label_names = sorted(labels["label"].unique())
        return label_names
    
    raise Exception("Argument expected in function.")



class SpectDataset(Dataset):
    def __init__(self, annotations_file, img_dir, 
                 transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path) 
        # image = image.convert("RGB")
        # image = image.to(torch.float)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class NumericalFeatureDataset(Dataset):
    def __init__(self, file,  
                 transform=None, target_transform=None):
        labels = pd.read_csv(file, usecols=["label"])
        label_names = sorted(labels["label"].unique())
        label_dict = {}
        i = 0
        for label in label_names:
            label_dict[label] = i
            i += 1

        labels = [label_dict[label] for label in labels["label"].to_list()]

        self.labels = torch.tensor(labels).long()
        
        data = pd.read_csv(file, nrows=1)
        cols = list(data.columns)
        cols.remove("filename")
        cols.remove("label")
        data = pd.read_csv(file, usecols=cols)
        data_norm = (data - data.min()) / (data.max() - data.min())
        self.data = torch.tensor(data_norm.to_numpy()).float()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class FeatureNetwork(nn.Module):
    def __init__(self, num_classes=10, num_features=58):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(num_features, 128)
        self.fc2 = nn.Linear(128, 192)
        self.fc3 = nn.Linear(192, 128)
        self.fc4 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
    def train_model(self, trainloader, epochs=10, device='cpu',
                optimizer=None,
                loss=None,
                testloader=None,
                loss_threshold=1000.0):

        if optimizer is None:
            optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        if loss is None:
            loss = nn.CrossEntropyLoss()

        history = []
        self.train()
        for epoch in range(epochs):
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs)
                loss_val = loss(outputs, labels)
                loss_val.backward()
                optimizer.step()

            if testloader is None:
                val_accuracy = ''
            else:
                val_accuracy = f'validation accuracy \
                    {self.validate(testloader, device=device)}'

            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss_val.item():.4f} \
                  {val_accuracy}')
            history.append(loss_val.item())

            if loss_val.item() < loss_threshold:
                break

        return history
    
    def validate(self, testloader, device='cpu'):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = self(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 100.0 * correct / total


class SpectCnn(nn.Module):
    def __init__(self, num_channels=1, num_classes=10,
                img_w=10, img_h=10):
        super(SpectCnn, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(23040, 256)
        self.fc3 = nn.Linear(256, num_classes)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x

    def train_model(self, trainloader, epochs=10, device='cpu',
                optimizer=None,
                loss=None,
                testloader=None,
                loss_threshold=100.0):

        if optimizer is None:
            optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        if loss is None:
            loss = nn.CrossEntropyLoss()

        history = []
        self.train()
        for epoch in range(epochs):
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs)
                loss_val = loss(outputs, labels)
                loss_val.backward()
                optimizer.step()

            if testloader is None:
                val_accuracy = ''
            else:
                val_accuracy = f'validation accuracy \
                    {self.validate(testloader, device=device)}'

            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss_val.item():.4f} \
                  {val_accuracy}')
            history.append(loss_val.item())

            if loss_val.item() < loss_threshold:
                break

        return history
    

    def validate(self, testloader, device='cpu'):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = self(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 100.0 * correct / total