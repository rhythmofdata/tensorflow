
# Use control-shift-p in order to put editor in your desired virtual environment.

# Using Sequential method to build model


import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import shutil
from sklearn.model_selection import train_test_split
import csv


from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, Input, BatchNormalization, Dropout, Dense, MaxPool2D, Flatten, GlobalAvgPool2D
from keras.preprocessing.image import ImageDataGenerator




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




def splitData(path_to_data, path_to_save_train, path_to_save_val, split_size=0.1):
    folders = os.listdir(path_to_data)
    
    for folder in folders:
        full_path = os.path.join(path_to_data, folder)
        images_paths = glob.glob(os.path.join(full_path, '*.png'))  # see inside folder and load files
        
        x_train, x_val = train_test_split(images_paths, test_size=split_size)
        
        for x in x_train:
            path_to_folder = os.path.join(path_to_save_train, folder)
            
            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)
                
            shutil.copy(x,path_to_folder)
            
            
            
        for x in x_val:
            path_to_folder = os.path.join(path_to_save_val, folder)
            
            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)
                
            shutil.copy(x,path_to_folder)




def order_test_set(path_to_images, path_to_csv):

    try:
        with open(path_to_csv, 'r') as csvfile:

            reader = csv.reader(csvfile, delimiter=',')

            for i, row in enumerate(reader):

                if i==0:
                    continue        # skip over header row

                img_name = row[-1].replace('Test/', '')
                label = row[-2]

                path_to_folder = os.path.join(path_to_images, label)

                if not os.path.isdir(path_to_folder):
                    os.makedirs(path_to_folder)

                img_full_path = os.path.join(path_to_images, img_name)
                #print(img_full_path)
                shutil.move(img_full_path, path_to_folder)

    except:
        print('[INFO] : Error reading csv file')   

    

def create_generators(batch_size, train_data_path, val_data_path, test_data_path):

    train_preprocessor = ImageDataGenerator(   #this way we augmentate on just the training data
        rescale = 1/255.,
        rotation_range=10,
        width_shift_range=0.1
    )


    test_preprocessor = ImageDataGenerator(
        rescale = 1/255.
    )


    train_generator = train_preprocessor.flow_from_directory(
        train_data_path,
        class_mode = "categorical",
        target_size=(60,60),
        color_mode = 'rgb',
        shuffle=True,
        batch_size=batch_size,
    )



    val_generator = test_preprocessor.flow_from_directory(
        val_data_path,
        class_mode = "categorical",
        target_size=(60,60),
        color_mode = 'rgb',
        shuffle=False,
        batch_size=batch_size,
    )


    test_generator = test_preprocessor.flow_from_directory(
        test_data_path,
        class_mode = "categorical",
        target_size=(60,60),
        color_mode = 'rgb',
        shuffle=False,
        batch_size=batch_size,
    )


    return (train_generator, val_generator, test_generator)