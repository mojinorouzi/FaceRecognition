# Face Recognition
## Siamese Neural Network

### Anchor images , negative images and position images

we use Siamese neural network for compare anchor images with negative images and position images 

position images includes images of that person you want recognize 

negative images includes images of another persons 

we use this collection for labeling and training 

## implement Siamese Neural Network

The following article gives you a good idea of this neural network 

https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf


### Setup commands

## Downdload lfw.tgz

lfw is Labeled Faces in the Wild, a database of face photographs designed for studying the problem of unconstrained face recognition.
so dowload dataset from flolowing link:

http://vis-www.cs.umass.edu/lfw/

we use this dataset as negative images

* save in root of project

## Install Dependencies

pip install tensorflow==2.4.1 tensorflow-gpu==2.4.1 opencv-python matplotlib


## Run create_files.py

run this script for create anchor , negative and position directories and also move lfw dataset to negative directory 

python create_files.py

## Run create_ancher_position.py

for collecting anchor images and position images with webcam ,  run create_anchor_position.py 

- press 'p' when you want save images for position dataset
- press 'a' when you want save anchor dataset  

python create_ancher_position.py

## Run train script

python train.py

- after training this will save model 

## Run App with siamesemodel.h5

we use siamesemodel.h5 that we've saved model in training process

python app.py