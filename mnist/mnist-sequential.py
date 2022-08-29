
# Use control-shift-p in order to put editor in your desired virtual environment.

# Using Sequential method to build model

import tensorflow
#import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, Input, BatchNormalization, Dropout, Dense, MaxPool2D, Flatten, GlobalAvgPool2D


model = keras.Sequential(
    [
        Input(shape=(28, 28, 1)),  #images are 28x28  with a channel of 1 s0 28x28x1
        Conv2D(32, (3, 3), activation='relu'),  # Here are 32 filters of size 3x3 with relu activation
        Conv2D(64, (3, 3), activation='relu'),
        MaxPool2D(),
        BatchNormalization(),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPool2D(),
        BatchNormalization(),

        GlobalAvgPool2D(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ]
)

model.summary()

# create a helper function to disply some samples


def display_some_examples(examples, labels):
    plt.figure(figsize=(10, 10))

    # plot 25 images
    for i in range(25):

        # choose between 0 and 60k-1
        idx = np.random.randint(0, examples.shape[0]-1)  # Randomly choose images.
        img = examples[idx]
        label = labels[idx]

        plt.subplot(5,5, i+1)
        plt.title(str(label))
        plt.tight_layout()    #adds more space to make it easier to see layout
        plt.imshow(img, cmap='gray')  # because this is a gray scale image we use cmap

    plt.show()


if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()

    print("x_train.shape= ", x_train.shape)
    print("x_train.shape= ", y_train.shape)
    print("x_train.shape= ", x_test.shape)
    print("x_train.shape= ", y_test.shape)

    if False:
        display_some_examples(x_train, y_train)

    x_train = x_train.astype('float32') / 255.00  # Normalize the training data
    x_test = x_test.astype('float32') / 255.00  # Normalize the test data

    x_train = np.expand_dims(x_train, axis=-1) # because we want 28x28x1 and not just 28x28.   -1 adds dimension at the end 
    x_test = np.expand_dims(x_test, axis=-1)    # because we want 28x28x1 and not just 28x28.  -1 adds demension at the end

    

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics='accuracy')
    # Not using one-hot encoding so must use sparse_categorical
    # learn more about crossentropy on tensorflow website and other resources

    #If we wanted to do one-hot encoding we would do:
    #y_train = tensorflow.keras.utils.to_categorical(y_train, 10)
    #y_test = tensorflow.keras.utils.to_categorical(y_test, 10)


    history = model.fit(x_train, y_train, batch_size=64,
                        epochs=10, validation_split=0.2)
    # validation is used at the end of ever epoch to check how the model is doing on un-trained data

    model.evaluate(x_test, y_test, batch_size=64)
