
# Use control-shift-p in order to put editor in your desired virtual environment.




import weakref
import tensorflow as tf
import numpy as np






def predict_from_model(model, img_path):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image,dtype=tf.float32)
    image = tf.image.resize(image,[60,60])
    image = tf.expand_dims(image,axis=0)    #expects(1,60,60,3)     1 is in axix 0

    predictions= model.predict(image)  #[0.005,0.00003.0.99.0.00 ...]
    predictions = np.argmax(predictions)  # gives index of max value

    return predictions




if __name__ == '__main__':

    img_path = '/home/usojourn/tensorflow/tensorflow-projects/sign-recognition/Test/4/00014.png'

    model = tf.keras.models.load_model('./saved_models')
    prediction = predict_from_model(model, img_path)

    

    #Although the class directories appear as numbers, they are actually sorted as strings,
    #ImageDataGenerator.flow_from_directory sorts them as alphanumeric string so we
    #need to do a conversion so that we can print out the correct class number.
    prediction_dict = {
    0 :0,
    1 :1 ,
    2 :10,
    3 :11,
    4 :12,
    5 :13,
    6 :14,
    7 :15,
    8 :16,
    9 :17,
    10:18,
    11:19,
    12:2 ,
    13:20,
    14:21,
    15:22,
    16:23,
    17:24,
    18:25,
    19:26,
    20:27,
    21:28,
    22:29,
    23:3 ,
    24:30,
    25:31,
    26:32,
    27:33,
    28:34,
    29:35,
    30:36,
    31:37,
    32:38,
    33:39,
    34:4 ,
    35:40,
    36:41,
    37:42,
    38:5 ,
    39:6 ,
    40:7 ,
    41:8 ,
    42:9 ,
}


print(f"prediction = {prediction_dict[prediction]}") 
