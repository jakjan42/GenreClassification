import genre_classifier as gc
import matplotlib.pyplot as plt
import tensorflow as tf


def main():

    # signal, sr = librosa.load(librosa.util.example('brahms'))
    # CUDA_VISIBLE_DEVICES = 0, 1
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # print(gpus)

    data = tf.keras.utils.image_dataset_from_directory('data',
                                                       color_mode="grayscale",
                                                       image_size=(128, 1293),
                                                       batch_size=10)

    data = data.map(lambda x, y: (x/255, y))

    train, val, test = gc.get_datasets(data)
    model = gc.get_model_spectrogram()
    model.summary()
    gc.compile_model_spectrogram(model)
    epochs = 20
    history = gc.train_model_spectrogram(model, train_data=train,
                                         validation_data=val,
                                         epochs=epochs)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


if __name__ == "__main__":
    main()
