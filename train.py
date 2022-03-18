import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt


# import tensorflow dependencies - functional apis   

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer , Conv2D  , Dense , MaxPooling2D , Input , Flatten
import tensorflow as tf 
# import metrix evaluations
from tensorflow.keras.metrics import Precision , Recall


# Import uuid library for generate unique image names

import uuid 

# setup path

POS_PATH = os.path.join('data' , 'position')

NEG_PATH = os.path.join('data' , 'negative')

ANC_PATH = os.path.join('data' , 'anchor')



anchor = tf.data.Dataset.list_files(ANC_PATH+'/*.jpg').take(300)

positive = tf.data.Dataset.list_files(POS_PATH+'/*.jpg').take(300)

negative = tf.data.Dataset.list_files(NEG_PATH+'/*.jpg').take(300)
  


def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img,(100 , 100))
    img = img / 255.0
    return img


positives = tf.data.Dataset.zip((anchor , positive , tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor , negative , tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)

def preprocess_twin(input_img , validation_img , label):
    return(preprocess(input_img), preprocess(validation_img),label)

#build dataloader pipeline
data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=1024)

# training partion
train_data = data.take(round(len(data)*.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

# testing partion
test_data = data.skip(round(len(data)*.7))
test_data = test_data.take(round(len(data)*.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8) 



def make_embedding():
    inp = Input(shape=(100 , 100 , 3) , name='input_image')
    # First block
    c1 = Conv2D(64 , (10 , 10) , activation='relu')(inp)
    m1 = MaxPooling2D(64 , (2 , 2) , padding='same')(c1)
    # Seconde block
    c2 = Conv2D(128 , (7 , 7) , activation='relu')(m1)
    m2 = MaxPooling2D(64 , (2 , 2) , padding='same')(c2)
    # Third block
    c3 = Conv2D(128 , (4 , 4) , activation='relu')(m2)
    m4 = MaxPooling2D(64 , (2 , 2) , padding='same')(c3)
    # finall embedding layer
    c4 = Conv2D(256 , (4 , 4) , activation='relu')(m4)
    f1 = Flatten()(c4)
    d1 = Dense(4094 , activation='sigmoid')(f1)
    
    
    return Model(inputs=[inp],outputs=[d1],name='embedding')

embedding = make_embedding()

class L1Dist(Layer):
    def __init__(self,**kwargs):
        super().__init__()
        
    def call(self , input_embedding, validation_embedding):
        return tf.math.abs( input_embedding - validation_embedding)
    
    
    
def make_siamese_model():
    # anchor image input in the network
    input_image = Input(name='input_img' , shape=(100 , 100 , 3))
    # validation image in the network
    validation_image = Input(name='validation_img' , shape=(100 , 100 , 3))
    
    #combine siamese distance components
    siamese_layer = L1Dist()
    
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))
    
    
    #classification layer
    
    classifier = Dense(1 , activation='sigmoid')(distances) 
    
    return Model(inputs=[input_image, validation_image] , outputs=[classifier] , name= 'SiameseNetwork')

# create model
siamese_model = make_siamese_model()
# Loss function

binary_cross_loss = tf.losses.BinaryCrossentropy()


# Optimization function
opt = tf.keras.optimizers.Adam(1e-4)


# checkpoint 

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir , 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt , siamese_model=siamese_model)


@tf.function
def train_step(batch):
    
    with tf.GradientTape() as tape:
        
        #get anchor and pos/neg image 
        
        X = batch[:2]
        #get label
        y = batch[2]
        
        # forward pass
        yhat = siamese_model(X , training=True)
        #calculate loss 
        loss = binary_cross_loss(y , yhat)
    print(loss)
        # calculate gradients
    grad = tape.gradient(loss , siamese_model.trainable_variables)
    
    # calculate updated weights and apply to siamese models
    
    opt.apply_gradients(zip(grad , siamese_model.trainable_variables))
    
    return loss


def train(data , EPOCHS):
    # loop through epochs
    
    for epoch in range(1 , EPOCHS+1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))
        
        # loop through each batch
        
        for idx , batch in enumerate(train_data):
            train_step(batch)
            progbar.update(idx+1)
            
        if epoch % 10 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
            
            
EPOCHS = 50
train(train_data , EPOCHS)


            
            
    
    
    
# TEST        
    
    
# make predictions
test_input , test_val , y_true = test_data.as_numpy_iterator().next()
y_hat = siamese_model([test_input , test_val])

m = Recall()
m.update_state(y_true , y_hat)
print(m.result().numpy())


p = Precision()
p.update_state(y_true , y_hat)
print(p.result().numpy())




# save weights
siamese_model.save('siamesemodel.h5')


         
        


 
    
    




    
    
    

