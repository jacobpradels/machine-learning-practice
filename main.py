import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils
import os
from os import listdir
from random import randrange
import json

def preProcessImage(filename):
    """
    Method used to process an image before being put through the tensorflow algorithm

    Parameter
    ---------
    filename: The relative file location of the image to be processed
    """
    img = image.load_img(filename, target_size=(224,224))
    resized_image = image.img_to_array(img)
    final_image = np.expand_dims(resized_image,axis=0)
    final_image = tf.keras.applications.mobilenet_v2.preprocess_input(final_image)
    return final_image

def predictAnimal(processed_image, application):
    """
    Will return a processed prediction array of the algorithm's guess of what it is identifying

    Parameters
    ---------
    processed_image: the image that is properly formatted to be processed by tensorflow
    application: the tensorflow application to process the image
    """
    predictions = application.predict(processed_image)
    processed_predictions = imagenet_utils.decode_predictions(predictions)
    return processed_predictions

def main():
    """
    The main method controling primary logic
    """
    NUMBER_OF_RESULTS = 100

    result_dict = {}
    application = tf.keras.applications.mobilenet_v2.MobileNetV2()
    dog_choices = os.listdir('Images')
    
    for x in range(NUMBER_OF_RESULTS):
        selection = dog_choices[randrange(0,len(dog_choices))]
        selected_image = os.listdir('Images\\' + selection)[randrange(0,len(selection))]
        filename = "Images\\" + selection + "\\" + selected_image

        final_image = preProcessImage(filename)
        results = predictAnimal(final_image, application)
        
        breed = results[0][0][1]
        if (breed in result_dict):
            result_dict[breed] += 1
        else:
            result_dict[breed] = 1


    sorted_results = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
    with open("output.json","w") as output_file:
        json.dump(sorted_results, output_file, indent=4)

main()