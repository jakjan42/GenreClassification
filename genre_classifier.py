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
import torchaudio
import torchaudio.transforms as aT
import audio_processor as ap
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

class AudioDataset(Dataset):
    def __init__(self,
                 root='audio', 
                 transform=None,
                 sr=None,
                 subset=None
        ):
        self.root = root
        self.transform = transform
        self.sr = sr
        self.subset = subset

        if not os.path.isdir(root):
            raise RuntimeError("Audio root directory does not exist")
        if len(os.listdir(root)) == 0:
            raise RuntimeError("Audio root directory empty")
        if subset is not None and subset not in ["train", "validation", "test"]:
            raise ValueError("Subset should be None or be equal to 'train', 'validation', or 'test'")

        if subset is not None: 
            if not os.path.isdir(os.path.join(root, subset)):
                raise RuntimeError(f"{subset} directory does not exist")
            
        self._walker = []
        self._class_ids = []
        self.classes = []
        for dir in os.listdir(root):
            if subset is not None:
                if dir != subset:
                    continue
            class_id = 0
            _classes = os.listdir(os.path.join(root, dir))
            if len(self.classes) != 0 and _classes != self.classes:
                raise RuntimeError("Class directories don't match")

            for c in _classes:
                self.classes.append(c)
                fulldir = os.path.join(root, dir, c)
                class_files = os.listdir(fulldir)
                class_files.sort()
                for fname in class_files:
                    self._walker.append(fname)
                    self._class_ids.append(class_id)
                class_id += 1


    def __len__(self):
        return len(self._walker)
    
    def __getitem__(self, index):
        filepath = ""
        class_id = self._class_ids[index]
        fname = self._walker[index]
        if self.subset is None:
            for dir in os.listdir(self.root):
                dir = os.path.join(self.root, dir)
                c = os.listdir(dir)[class_id]
                fulldir = os.path.join(dir, c)
                if fname in os.listdir(fulldir):
                    filepath = os.path.join(fulldir, fname)
        else:
            dir = os.path.join(self.root, self.subset)
            c = os.listdir(dir)[class_id]
            fulldir = os.path.join(dir, c)
            filepath = os.path.join(fulldir, fname)

        waveform, sample_rate = torchaudio.load(filepath)
        if self.sr is not None:
            resampler = aT.Resample(sample_rate, self.sr, dtype=waveform.dtype)
            sample_rate = self.sr
            waveform = resampler(waveform)
        if self.transform is not None:
            waveform = self.transform(waveform)

        return waveform, sample_rate, class_id        
    


        

        



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
        image = image.convert("RGB")
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
        print(self.labels)
        
        data = pd.read_csv(file, nrows=1)
        cols = list(data.columns)
        cols.remove("filename")
        cols.remove("label")
        cols.remove("length")
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
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(num_features),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 10)
        )

    def forward(self, x):
        x = self.layers(x)
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
        for epoch in range(epochs):
            self.train()
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
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.BatchNorm2d(32),
            # nn.Dropout(0.2),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.BatchNorm2d(32),
            # nn.Dropout(0.2),

            nn.Conv2d(32, 32, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            # nn.Dropout(0.2),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(9152, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
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
        for epoch in range(epochs):
            self.train()
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
    

    if __name__ == "__main__":
        sr = 22050
        # transform = None
        transform = transforms.Compose([
            aT.MelSpectrogram(sr, n_fft=2048, hop_length=512, n_mels=128),
            aT.AmplitudeToDB(80),
            transforms.RandomCrop((128, 200)),
            transforms.Normalize((0.5,), (0.5))
        ])
        ds = AudioDataset(subset="train", sr=sr, transform=transform)
        # for i in range(len(ds)):
        #     n, l = ds[i]
        #     print(n, ds.classes[l])
        # print(len(ds))
        wf, sr, l = ds[200]
        wf -= wf.min()
        wf /= wf.max()
        print(wf, sr, ds.classes[l])