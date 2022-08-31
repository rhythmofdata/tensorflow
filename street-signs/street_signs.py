
# Use control-shift-p in order to put editor in your desired virtual environment.

# Using Functional method to build model


import glob
import shutil
import os
from tensorflow import keras 
from keras.callbacks import ModelCheckpoint, EarlyStopping
from my_utils import create_generators, splitData, order_test_set

from deeplearning_models import street_signs_model







if __name__ == '__main__':


    path_to_train = '/home/usojourn/tensorflow/tensorflow-projects/sign-recognition/training_data/train'
    path_to_val = '/home/usojourn/tensorflow/tensorflow-projects/sign-recognition/training_data/val'
    path_to_test = '/home/usojourn/tensorflow/tensorflow-projects/sign-recognition/Test'

    batch_size = 64
    epochs = 15



    train_generator, val_generator, test_generator = create_generators(batch_size,path_to_train,path_to_val,path_to_test)
    nbr_classes = train_generator.num_classes

    #Using callbacks to save model
    path_to_saved_model = "./saved_models"
    model_saver = ModelCheckpoint(
        path_to_saved_model,
        monitor="val_accuracy",
        mode='max',
        save_best_only=True,
        save_freq='epoch',
        verbose=1
    )



    early_stop = EarlyStopping(monitor='val_accuracy',patience=10)

    model = street_signs_model(nbr_classes)

    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(train_generator,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=val_generator,
              callbacks=[model_saver]
              )