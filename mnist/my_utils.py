
# Use control-shift-p in order to put editor in your desired virtual environment.

# Using Sequential method to build model

import tensorflow
#import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, Input, BatchNormalization, Dropout, Dense, MaxPool2D, Flatten, GlobalAvgPool2D


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
    pass

    

    

