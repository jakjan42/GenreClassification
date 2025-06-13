import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.datasets.mnist
#import genre_classifier as gc
import datasets as ds
import models as mod
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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
    trainset = ds.SpectDataset(
        annotations_file=os.path.join(root, "train", "labels.csv"),
        img_dir=os.path.join(root, "train"),
        transform=transform
    )

    testset = ds.SpectDataset(
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

    # sr = 22050
    # n_mels = 256
    # transform = transforms.Compose([
    #     aT.MelSpectrogram(sr, n_fft=2048, hop_length=512, n_mels=n_mels),
    #     transforms.RandomCrop((n_mels, 200)),
    #     aT.AmplitudeToDB(80),
    #     transforms.Lambda(normalize01),
    #     transforms.Normalize((0.5,), (0.5))
    # ])

    batch_size = 32

    # trainset = ds.AudioDataset(subset="train", sr=sr, transform=transform)
    # testset = ds.AudioDataset(subset="test", sr=sr, transform=transform)
    # # trainset, testset = get_spectrogram_set(transform=transform)
    # trainloader, testloader = get_data_loaders(trainset, testset, batch_size=batch_size)
    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # data = next(iter(trainloader))
    # images, labels = data[0].to(device), data[1].to(device)
    
    # imshow(torchvision.utils.make_grid(images.cpu()))

    # cnn_path = 'spect_cnn.pth'
    # net = mod.SpectCnn(num_channels=3).to(device)
    # net.load_state_dict(torch.load(cnn_path, weights_only=True))

    full_df = pd.read_csv('features/features_30_sec.csv')

    feature_columns = [col for col in full_df.columns if col not in ['filename', 'label']]
    X_full = full_df[feature_columns]
    y_full = full_df['label']

    # splitting into test and train
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    train_df_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    train_df_scaled['label'] = y_train.values

    test_df_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    test_df_scaled['label'] = y_test.values

    train_file_path = 'features/temp_train_features_scaled.csv'
    test_file_path = 'features/temp_test_features_scaled.csv'
    
    # temporary csvs
    train_df_scaled.to_csv(train_file_path, index=False)
    test_df_scaled.to_csv(test_file_path, index=False)
    
    print(f"Number of training samples: {len(train_df_scaled)}")
    print(f"Number of test samples: {len(test_df_scaled)}")
    print(f"Number of features: {X_train_scaled.shape[1]}")

    trainset = ds.NumericalFeatureDataset(train_file_path, transform=None)
    testset = ds.NumericalFeatureDataset(test_file_path, transform=None)

    # transform = transforms.Normalize((0.5), (0.5))
    # # transform = None
    # numerical = ds.NumericalFeatureDataset('features/features_30_sec.csv',
    #                                 transform=transform)
    # v, l = numerical[0]
    v, l = trainset[0]
    print(v)
    print(l)
    print(v.shape)

    actual_num_features = v.shape[0]

    trainloader, testloader = get_data_loaders(trainset, testset, batch_size=batch_size)

    # trainloader = torch.utils.data.DataLoader(numerical, batch_size=batch_size,
    #                                           shuffle=True, num_workers=2)
    # net = mod.FeatureNetwork(num_features=57).to(device=device)
    num_classes = 10
    net = mod.FeatureNetwork(num_features=actual_num_features, num_classes=num_classes).to(device=device)
    #net = mod.SpectCnn(img_w=200, img_h=n_mels, num_channels=1).to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, weight_decay=1e-5)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode='max',       # look for max
    #     factor=0.1,       # reduce lr
    #     patience=10      # wait 10 epochs without improvement before reducing lr
    # )

    epochs = 30
    net.train_model(trainloader, epochs=epochs,
                    loss_threshold=0.1,
                    optimizer=optimizer,
                    loss=criterion,
                    testloader=testloader,
                    device=device)

    print('Finished Training')
    #torch.save(net.state_dict(), cnn_path)
    feature_net_path = 'feature_network.pth' 
    torch.save(net.state_dict(), feature_net_path)
    print(f'Train accuracy: {net.validate(trainloader, device=device)}')
    print(f'Test accuracy: {net.validate(testloader, device=device)}')

    os.remove(train_file_path)
    os.remove(test_file_path)
    


if __name__ == "__main__":
    main()
