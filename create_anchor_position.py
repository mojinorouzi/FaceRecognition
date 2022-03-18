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




# establish a connection to the webcam

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret , frame = cap.read()
    # CUT DOWN FRAME to 250 * 250 pixels
    frame = frame[:250 , :250 , :]
    cv2.imshow('Image collection' , frame)
    # collect anchors 
    if cv2.waitKey(1) & 0XFF == ord('p'):
        img_name = os.path.join(POS_PATH , '{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(img_name , frame)
    if cv2.waitKey(2) & 0XFF == ord('a'):
        img_name = os.path.join(ANC_PATH , '{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(img_name , frame)
    # collect positives
    # show images
    if cv2.waitKey(3) & 0XFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
