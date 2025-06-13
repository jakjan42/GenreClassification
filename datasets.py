import torch
from PIL import Image
import pandas as pd
import os
import torchaudio
import torchaudio.transforms as aT
from torch.utils.data import Dataset


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

        return waveform, class_id, sample_rate
    

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
        if "filename" in cols:
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