import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization

def create_model(training_shape):
    """
    Conv1D (1-D Convolution):
        - Applies a convolution operation over a 1-D input
        - In this context, it's processing a sequence or time-series data
        - Parameters:
            - 256/128,64: Number of output filters (feature detectors)
            - kernel_size=5: Size of the sliding window (5 time steps)
            - strides=1: how much the kernel moves each step
            - padding='same': ensures the output has same length as input
            - activation='relu': introduces non-linearity
    MaxPooling1D:
        - Downsamples the input by taking the maximum value in each pooling window
        - Reduces spatial dimensions of the input
        - Parameters:
            - pool_size=5: size of the pooling window
            - strides=2: how much the pooling window moves
            - padding='same': ensures the output has similar dimensions to input
    - Each Conv1D layer learns different features from the input sequence
    - MaxPooling1D reduces computational complexity and helps capture most important features
    - Dropout layers prevent overfitting
    - Final layers flatten the output and use dense layers for classification
    :param training_shape:
    :return:
    """
    model = tf.keras.models.Sequential()
    model.add(
        Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(training_shape.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))

    model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))

    model.add(Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))

    model.add(Flatten())
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(units=8, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    return model