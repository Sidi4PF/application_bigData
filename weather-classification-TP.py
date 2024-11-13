#!/usr/bin/env python
# coding: utf-8

# Imports
import keras
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

# Data
from tensorflow.image import resize
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.utils import load_img, img_to_array

# Data Viz
import seaborn as sns
import matplotlib.pyplot as plt

# TL Model
from tensorflow.keras.applications import ResNet50, ResNet50V2, InceptionV3, Xception, ResNet152, ResNet152V2

# Model
from keras import Sequential
from keras.layers import Dense, GlobalAvgPool2D, Dropout
from keras.models import load_model

# Callbacks 
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Model Performance
from sklearn.metrics import classification_report

# Model Viz
from tensorflow.keras.utils import plot_model

# Categories
class_names = {0: 'cloudy', 1: 'foggy', 2: 'rainy', 3: 'shine', 4: 'sunrise'}

# Function to load image
def load_image(path):
    '''
    Takes in path of the image and load it
    '''
    img = resize(img_to_array(load_img(path))/255., (256,256))
    return img

# Function to show image with title
def show_image(image, title=None):
    '''
    Takes in an Image and plot it with Matplotlib
    '''
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')

# Function to load data
def load_data(img_paths):
    X = np.zeros(shape=(len(img_paths), 256,256,3))

    for i, path in tqdm(enumerate(img_paths), desc="Loading"):
        X[i] = load_image(path)
    
    return X

# Main script
if __name__ == "__main__":
    # Load images
    image_paths = sorted(glob('*.jpg'))
    print(f"Total Number of Images : {len(image_paths)}")
    print(image_paths[:5])

    # Load images data
    images = load_data(image_paths)

    # Data Visualization
    plt.figure(figsize=(10,10))
    for i in range(25):
        if i > len(images)-1:
            break
        image = images[i]
        plt.subplot(5,5,i+1)
        show_image(image, title=f"Image : {i}")
    plt.tight_layout()
    plt.show()

    # Load pre-trained model
    model_v3 = load_model('ResNet152V2-Weather-Classification-03.h5')

    # Make Predictions
    preds = np.argmax(model_v3.predict(images), axis=-1)

    # Show results with predictions
    plt.figure(figsize=(15,20))
    for i, im in enumerate(images):
        pred = class_names[list(preds)[i]]
        plt.subplot(5,5,i+1)
        show_image(im, title=f"Pred : {pred}")
    plt.tight_layout()
    plt.show()
