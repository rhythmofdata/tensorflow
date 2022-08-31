
# Use control-shift-p in order to put editor in your desired virtual environment.

# Using Sequential method to build model

import tensorflow
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, Input, BatchNormalization, Dropout, Dense, MaxPool2D, Flatten, GlobalAvgPool2D

from keras import Model
#from deeplearning_models import street_signs_model



class myCustomModel(tensorflow.keras.Model):
    def __init__(self):
        super().__init__()

        self.conv1 = Conv2D(32, (3, 3), activation='relu')
        self.conv2 = Conv2D(64, (3, 3), activation='relu')
        self.maxpool1 = MaxPool2D()
        self.batchnorm1 = BatchNormalization()

        self.conv3 = Conv2D(128, (3, 3), activation='relu')
        self.maxpool2 = MaxPool2D()
        self.batchnorm2 = BatchNormalization()

        self.globalavgpool1 = GlobalAvgPool2D()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(10, activation='softmax')


    def call(self,cnn_input):         #use forward in pyTorch

        x = self.conv1(cnn_input)
        x = self.conv2(x)
        x = self.maxpool1(x) 
        x = self.batchnorm1(x)

        x = self.conv3(x) 
        x = self.maxpool2(x) 
        x = self.batchnorm2(x) 

        x = self.globalavgpool1(x)
        x = self.dense1(x)
        x = self.dense2(x) 

        return x





#Use functional model
def functional_model():

    
    cnn_input = Input(shape=(28, 28,1))
    x = Conv2D(32, (3, 3), activation='relu')(cnn_input)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = GlobalAvgPool2D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)
    
    model = tensorflow.keras.Model(inputs=cnn_input, outputs = x)
    model.summary()
    return model





def street_signs_model(num_classes):
    cnn_input = Input(shape=(60,60,3))

    x = Conv2D(32, (3, 3), activation='relu')(cnn_input)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    #x = GlobalAvgPool2D()(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)
    
    

    return Model(inputs=cnn_input, outputs = x)




if __name__=="__main__":          # now we can do testing from here if we want

    model = street_signs_model(10)
    model.summary()
