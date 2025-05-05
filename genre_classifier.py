import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


def get_model_spectrogram(data):
    model = Sequential()

    model.add(Conv2D(16, (3, 3), 1, activation='relu',
              input_shape=(128, 1293, 1)))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10))
    return model


def compile_model_spectrogram(model):
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])


def train_model_spectrogram(model, train_data=None,
                            validataion_data=None, epochs=10):
    history = model.fit(
        train_data,
        validation_data=validataion_data,
        epochs=epochs
    )

    return history


def get_datasets(data):
    train_size = int(len(data)*.7)
    val_size = int(len(data)*.2)
    test_size = int(len(data)*.1)

    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size+val_size).take(test_size)

    return train, val, test
