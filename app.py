import cv2
import os
import numpy as np
from tensorflow.keras.layers import Layer
import tensorflow as tf 





def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img,(100 , 100))
    img = img / 255.0
    return img



class L1Dist(Layer):
    def __init__(self,**kwargs):
        super().__init__()
        
    def call(self , input_embedding, validation_embedding):
        return tf.math.abs( input_embedding - validation_embedding)
    
    
# Loss function

binary_cross_loss = tf.losses.BinaryCrossentropy()

  








model = tf.keras.models.load_model('siamesemodel.h5' , custom_objects={'L1Dist':L1Dist , 'BinaryCrossentropy':binary_cross_loss})


def verify(model , detection_threshold , verification_threshold):
    results = []
    for image in os.listdir(os.path.join('application_data', 'verification_images')):
        input_img = preprocess(os.path.join('application_data' , 'input_image' , 'input_image.jpg'))
        if image[-4:] != ".jpg" : continue 
        validation_img = preprocess(os.path.join('application_data' , 'verification_images' , image))
        
        result = model.predict(list(np.expand_dims([input_img , validation_img] , axis=1)))
        results.append(result)
    detection = np.sum(np.array(results) > detection_threshold)  
    
    verification = detection / len(os.listdir(os.path.join('application_data' , 'verification_images'))) 
    
    verified = verification > verification_threshold
    
    return results , verified
    
     
cap = cv2.VideoCapture(0)
result = False
while cap.isOpened():
    ret , frame = cap.read()
    # CUT DOWN FRAME to 250 * 250 pixels
    frame = frame[:250 , :250 , :]
    if result :
        cv2.putText(frame , "verified" , (20 , 50) , cv2.FONT_HERSHEY_COMPLEX , 1 , 255 , 1)
    else:
        cv2.putText(frame , "Not verified" , (20 , 50) , cv2.FONT_HERSHEY_COMPLEX , 1 , 255 , 1)
        
    cv2.imshow('Verification' , frame)
    
    if cv2.waitKey(1) & 0XFF == ord('v'):
        #save input image
        cv2.imwrite(os.path.join('application_data' , 'input_image' , 'input_image.jpg') , frame)
        # run verification
        results , verified = verify(model , 0.5 , 0.5)
        result = verified
        
    if cv2.waitKey(2) & 0XFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()




        
         
        


 
    
    




    
    
    

