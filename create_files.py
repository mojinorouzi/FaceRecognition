import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt


# import tensorflow dependencies - functional apis   

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer , Conv2D  , Dense , MaxPooling2D , Input , Flatten
import tensorflow as tf 

# Import uuid library for generate unique image names

import uuid 

# setup path

POS_PATH = os.path.join('data' , 'position')

NEG_PATH = os.path.join('data' , 'negative')

ANC_PATH = os.path.join('data' , 'anchor')

# create directories
os.makedirs(POS_PATH)
os.makedirs(NEG_PATH)
os.makedirs(ANC_PATH)

# move lfw images to data/negative dir

for directory in os.listdir('lfw'):
    for file in os.listdir(os.path.join('lfw' , directory)):
        Ex_PATH = os.path.join('lfw' , directory, file)
        NEW_PATH = os.path.join(NEG_PATH , file)
        os.replace(Ex_PATH, NEW_PATH)


  