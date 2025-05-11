import numpy as np
import librosa
import os
import shutil
import cv2
import simpleaudio as sa
import csv



def convert_to_spectrogram(path, sr=22050, n_fft=2048, hop_length=512):
    signal, sr = librosa.load(path, sr=sr)
    mel_signal = librosa.feature.melspectrogram(y=signal, sr=sr,
                                                hop_length=hop_length,
                                                n_fft=n_fft)
    spectrogram = np.abs(mel_signal)

    return librosa.power_to_db(spectrogram, ref=np.max)


def audio_to_spectorgram_dataset(data_dir='data', audio_dir='audio'):
    if not os.path.isdir(audio_dir):
        raise Exception("audio directory does not exit")

    if len(os.listdir(audio_dir)) == 0:
        raise Exception("audio directory empty")

    if os.path.isdir(data_dir):
        shutil.rmtree(data_dir)
    os.mkdir(data_dir)

    train_path = os.path.join(data_dir, "train")
    test_path = os.path.join(data_dir, "test")
    os.mkdir(train_path)
    os.mkdir(test_path)

    genre_dict = {}

    i = 0
    for dir in os.listdir(audio_dir):
        genre_dict[dir] = i
        i += 1

    test_labels = []
    train_labels = []
    for genre in os.listdir(audio_dir):
        samples_count = len(os.listdir(audio_dir))
        i = 0
        for sample in os.listdir(os.path.join(audio_dir, genre)):
            is_train_set = i / samples_count > 0.8
            dir_path = train_path if is_train_set else test_path
            data_file = os.path.splitext(sample)[0] + ".png"
            sample_path = os.path.join(audio_dir, genre, sample)
            data_path = os.path.join(dir_path, data_file)

            spectorgram = convert_to_spectrogram(sample_path)

            cv2.imwrite(data_path, (spectorgram + 80.0).astype(np.uint8))

            if is_train_set:
                train_labels.append([data_file, genre_dict[genre]])
            else:
                test_labels.append([data_file, genre_dict[genre]])

            i += 1

    with open(os.path.join(test_path, "labels.csv"), 'w') as labels_file:
        labels = csv.writer(labels_file)
        labels.writerows(test_labels)
    with open(os.path.join(train_path, "labels.csv"), 'w') as labels_file:
        labels = csv.writer(labels_file)
        labels.writerows(train_labels)

    return genre_dict


def load_spectrogram(path):
    mel_db = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return mel_db


def get_audio_from_spectrogram(mel_db, sr=22050, n_fft=2028, hop_length=512):
    mel_signal = librosa.db_to_power(mel_db - 80.0)

    return librosa.feature.inverse.mel_to_audio(
        mel_signal, sr=sr, n_fft=n_fft, hop_length=hop_length)


def play_audio(audio, sr=22050):
    play_obj = sa.play_buffer(audio, 1, 4, sr)
    play_obj.wait_done()
