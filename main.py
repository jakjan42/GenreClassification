import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
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
import torchaudio.transforms as aT
import csv

def normalize01(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min()) 

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    # plt.imshow(Ep.transpose(npimg, (1, 2, 0)))
    t = np.transpose(npimg, (1, 2, 0))
    plt.imshow(t, cmap='gray', vmin=0, vmax=1)
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


def get_MINST_set(root="./data", transform=transforms.ToTensor()):
    testset = torchvision.datasets.mnist.MNIST(root=root, train=False,
                                           download=True, transform=transform)
    trainset = torchvision.datasets.mnist.MNIST(root=root, train=True,
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

    # transform = transforms.Compose([
    #     transforms.CenterCrop((218, 336)),
    #     transforms.ToTensor(),
    #     transforms.RandomCrop((218, 100)),
    #     transforms.Normalize((0.5,), (0.5,))
    # ])
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5,), (0.5)),
    # ])

    sr = 22050
    n_mels = 256
    transform = transforms.Compose([
        aT.MelSpectrogram(sr, n_fft=2048, hop_length=512, n_mels=n_mels),
        transforms.RandomCrop((n_mels, 200)),
        aT.AmplitudeToDB(80),
        transforms.Lambda(normalize01),
        transforms.Normalize((0.5,), (0.5))
    ])

    batch_size = 32

    trainset = gc.AudioDataset(subset="train", sr=sr, transform=transform)
    testset = gc.AudioDataset(subset="test", sr=sr, transform=transform)
    # trainset, testset = get_spectrogram_set(transform=transform)
    trainloader, testloader = get_data_loaders(trainset, testset, batch_size=batch_size)
    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = next(iter(trainloader))
    images, labels = data[0].to(device), data[1].to(device)
    
    imshow(torchvision.utils.make_grid(images.cpu()))

    cnn_path = 'spect_cnn.pth'
    # net = gc.SpectCnn(num_channels=3).to(device)
    # net.load_state_dict(torch.load(cnn_path, weights_only=True))

    # transform = transforms.Normalize((0.5), (0.5))
    # transform = None
    # ds = gc.NumericalFeatureDataset('features/features_30_sec.csv',
    #                                 transform=transform)
    # v, l = ds[0]
    # print(v)
    # print(l)
    # print(v.shape)
    # trainloader = torch.utils.data.DataLoader(ds, batch_size=batch_size,
    #                                           shuffle=True, num_workers=2)
    # net = gc.FeatureNetwork(num_features=57).to(device=device)
    net = gc.SpectCnn(img_w=200, img_h=n_mels, num_channels=1).to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, weight_decay=1e-5)

    epochs = 30
    net.train_model(trainloader, epochs=epochs,
                    loss_threshold=0.1,
                    optimizer=optimizer,
                    loss=criterion,
                    device=device)

    print('Finished Training')
    torch.save(net.state_dict(), cnn_path)
    print(f'Train accuracy: {net.validate(trainloader, device=device)}')
    # print(f'Test accuracy: {net.validate(testloader, device=device)}')


if __name__ == "__main__":
    main()
