import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.datasets.mnist
import torchvision.datasets.mnist
import genre_classifier as gc
import audio_processor as ap
import matplotlib.pyplot as plt
import torch
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
import torch.optim as optim
import torchvision.datasets as datasets
from torch import nn
from tqdm import tqdm
from torchmetrics import *
import pandas as pd
import csv
import cv2


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def get_data_loaders(trainset, testset, batch_size=16,
                     shuffle_trainset=True, shuffle_testset=False):
    if trainset is not None:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=shuffle_trainset, num_workers=2)
    else:
        trainloader = None

    if testset is not None:
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                  shuffle=shuffle_testset, num_workers=2)
    else:
        testloader = None

    return trainloader, testloader

def get_CIFAR10_set(root="./data", transform=transforms.ToTensor()):
    testset = torchvision.datasets.CIFAR10(root=root, train=False,
                                           download=True, transform=transform)
    trainset = torchvision.datasets.CIFAR10(root=root, train=True,
                                            download=True, transform=transform)
    
    return trainset, testset

def get_spectrogram_set(root="spect_datasets", transform=transforms.ToTensor()):
    trainset = gc.SpectDataset(
        annotations_file=os.path.join(root, "train", "labels.csv"),
        img_dir=os.path.join(root, "train"),
        transform=transform
    )

    testset = gc.SpectDataset(
        annotations_file=os.path.join(root, "test", "labels.csv"),
        img_dir=os.path.join(root, "test"),
        transform=transform
    )

    return trainset, testset


def main():
    # signal, sr = librosa.load(librosa.util.example('brahms'))
    # ap.audio_to_spectorgram_dataset(data_dir='data',
    #                                 audio_dir='audio')

    transform = transforms.Compose([
        transforms.CenterCrop((218, 336)),
        transforms.ToTensor(),
        transforms.RandomCrop((218, 200)),
        transforms.Normalize((0.5,), (0.5,))
    ])



    batch_size = 64
    trainset, testset = get_spectrogram_set(transform=transform, root='data')
    # trainset, testset = get_CIFAR10_set(transform=transform)
    trainloader, testloader = get_data_loaders(trainset, testset, batch_size=batch_size)
    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # data = next(iter(trainloader))
    # images, labels = data[0].to(device), data[1].to(device)
    
    # imshow(torchvision.utils.make_grid(images.cpu()))

    # cnn_path = 'spect_cnn3xconv3x3.pth'
    net = gc.SpectCnn(num_channels=3).to(device)
    # net.load_state_dict(torch.load(cnn_path, weights_only=True))

    transform = transforms.Normalize((0.5), (0.5))
    # # transform = None
    ds = gc.NumericalFeatureDataset('features/features_30_sec.csv',
                                    transform=transform)
    v, l = ds[0]
    print(v)
    print(l)
    trainloader = torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    net = gc.FeatureNetwork(num_features=58).to(device=device)
    # net = gc.SpectCnn(img_w=32, img_h=32, num_channels=3).to(device)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(net.parameters())
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)

    epochs = 50
    net.train_model(trainloader, epochs=epochs,
                    loss_threshold=0.1,
                    optimizer=optimizer,
                    loss=criterion,
                    device=device)

    print('Finished Training')
    cnn_path = "feature_FFNN.pth"
    torch.save(net.state_dict(), cnn_path)
    print(f'Train accuracy: {net.validate(trainloader, device=device)}')
    print(f'Test accuracy: {net.validate(testloader, device=device)}')

    # data = next(iter(testloader))
    # images, labels = data[0].to(device), data[1].to(device)
    # outputs = net(images)
    # _, predicted = torch.max(outputs, 1)
    # classes = ('plane', 'car', 'bird', 'cat',
    #        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # print([classes[p] for p in predicted])
    # imshow(torchvision.utils.make_grid(images.cpu()))
    
    # data_dir="spect_datasets"
    # spect_dir="spectrograms"
    # # os.mkdir(data_dir)
    # train_path = os.path.join(data_dir, "train")
    # test_path = os.path.join(data_dir, "test")
    # # os.mkdir(train_path)
    # # os.mkdir(test_path)
    # genre_dict = {}

    # i = 0
    # for dir in os.listdir(spect_dir):
    #     genre_dict[dir] = i
    #     i += 1

    # test_labels = []
    # train_labels = []
    # for genre in os.listdir(spect_dir):
    #     samples_count = len(os.listdir(spect_dir))
    #     i = 0
    #     for sample in os.listdir(os.path.join(spect_dir, genre)):
    #         is_train_set = i / samples_count > 0.8
    #         data_file = sample
    #         dir_path = train_path if is_train_set else test_path
    #         sample_path = os.path.join(spect_dir, genre, sample)
    #         data_path = os.path.join(dir_path, data_file)

    #         os.rename(sample_path, data_path)

    #         if is_train_set:
    #             train_labels.append([data_file, genre_dict[genre]])
    #         else:
    #             test_labels.append([data_file, genre_dict[genre]])

    #         i += 1

    # with open(os.path.join(test_path, "labels.csv"), 'w') as labels_file:
    #     labels = csv.writer(labels_file)
    #     labels.writerows(test_labels)
    # with open(os.path.join(train_path, "labels.csv"), 'w') as labels_file:
    #     labels = csv.writer(labels_file)
    #     labels.writerows(train_labels)


if __name__ == "__main__":
    main()
