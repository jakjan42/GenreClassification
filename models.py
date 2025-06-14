import torch
from torch import optim
from torch import nn
import librosa
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


class FeatureNetwork(nn.Module):
    def __init__(self, num_classes=10, num_features=58):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(num_features),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Dropout(0.45),
            nn.Linear(512, 256),
            nn.ReLU(),
            # nn.Dropout(0.4),
            nn.Dropout(0.35),
            nn.Linear(256, 128),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Dropout(0.25),
            # nn.Linear(128, 16),
            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Dropout(0.15),
            # nn.Linear(16, 10)
            nn.Linear(64, 10)
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

        train_loss_history = []
        val_accuracy_history = []
        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs)
                loss_val = loss(outputs, labels)
                loss_val.backward()
                optimizer.step()
                running_loss += loss_val.item()

            avg_train_loss = running_loss / len(trainloader)
            train_loss_history.append(avg_train_loss)

            print_str = f'Epoch [{epoch+1}/{epochs}], Loss: {avg_train_loss:.4f}'

            if testloader is None:
                val_accuracy = ''
            else:
                self.eval()
                val_accuracy = self.validate(testloader, device=device)         
                val_accuracy_history.append(val_accuracy)
                self.train()
                print_str += f', Validation Accuracy: {val_accuracy:.2f}%'

            print(print_str)

            if loss_val.item() < loss_threshold:
                break

        return train_loss_history, val_accuracy_history
    
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
    
# plotting training loss and validation accuracy
def plot_training_history(train_loss_hist, val_accuracy_hist, epochs, title="Training History", save_path=None):
    plt.figure(figsize=(12, 6))

    # training Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_loss_hist) + 1), train_loss_hist, label='Training Loss', color='blue')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(val_accuracy_hist) + 1), val_accuracy_hist, label='Validation Accuracy', color='green')
    plt.title('Validation Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

# plot confusion matrix
def plot_confusion_matrix(model, dataloader, device, class_names, title="Confusion Matrix", save_path=None):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

# classification report
def print_classification_report(model, dataloader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print("\n--- Classification Report ---")
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)


class SpectCnn(nn.Module):
    def __init__(self, num_channels=1, num_classes=10,
                img_w=10, img_h=10):
        super(SpectCnn, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_channels, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # nn.Dropout(0.2),

            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # nn.Dropout(0.2),

            nn.Conv2d(32, 32, 2, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
            # nn.Dropout(0.2),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3072, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
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
    

    def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
        if ax is None:
            _, ax = plt.subplots(1, 1)
        if title is not None:
            ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")


    def normalize01(tensor):
        return (tensor - tensor.min()) / (tensor.max() - tensor.min()) 

    # if __name__ == "__main__":
        # sr = 22050
        # # transform = None
        # # mel_spectrogram = aT.MelSpectrogram(
        # #     sample_rate=sr,
        # #     n_fft=2048,
        # #     hop_length=512,
        # #     center=True,
        # #     pad_mode="reflect",
        # #     power=2.0,
        # #     norm="slaney",
        # #     n_mels=128,
        # #     mel_scale="htk",
        # # )
        # transform = transforms.Compose([
            # aT.MelSpectrogram(sr, n_fft=2048, hop_length=512, n_mels=128),
            # transforms.RandomCrop((128, 200)),
            # aT.AmplitudeToDB(80),
            # transforms.Lambda(normalize01),
            # transforms.Normalize((0.5,), (0.5))
        # ])
        # ds = AudioDataset(subset="train", sr=sr, transform=transform)
        # wf, sr, l = ds[0]
        # import main as m
        # m.imshow(wf)
        # print(wf, sr,
               # ds.classes[l])
        # print(wf.shape)