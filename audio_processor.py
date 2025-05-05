import numpy as np
import librosa
import os
import shutil
import cv2
import simpleaudio as sa


def convert_to_spectrogram(path, sr=22050, n_fft=2048, hop_length=512):
    signal, sr = librosa.load(path)
    mel_signal = librosa.feature.melspectrogram(y=signal, sr=sr,
                                                hop_length=hop_length,
                                                n_fft=n_fft)
    spectrogram = np.abs(mel_signal)

    return librosa.power_to_db(spectrogram, ref=np.max)


def audio_to_spectorgram(data_dir='data', audio_dir='audio'):
    if not os.path.isdir(audio_dir):
        raise Exception("audio directory does not exit")

    if len(os.listdir(audio_dir)) == 0:
        raise Exception("audio directory empty")

    if os.path.isdir(data_dir):
        shutil.rmtree(data_dir)
    os.mkdir(data_dir)

    for dir in os.listdir(audio_dir):
        os.mkdir(os.path.join(data_dir, dir))

    for genre in os.listdir(audio_dir):
        for sample in os.listdir(os.path.join(audio_dir, genre)):
            sample_name = os.path.splitext(sample)[0]
            sample_path = os.path.join(audio_dir, genre, sample)
            data_path = os.path.join(data_dir, genre, sample_name + ".png")

            spectorgram = convert_to_spectrogram(sample_path)

            cv2.imwrite(data_path, (spectorgram + 80.0).astype(np.uint8))


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
