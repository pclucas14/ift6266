from __future__ import print_function

import sys
import os
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
from helper import * 
from PIL import Image
import glob

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.models import model_from_json
import os
import matplotlib.pyplot as plt
#from keras.utils.visualize_util import plot


# ################## Download and prepare the MNIST dataset ##################
# This is just some way of getting the MNIST dataset from an online location
# and loading it into numpy arrays. It doesn't involve Lasagne at all.
def keras_model(load=False):
    if load : 
        print('loading model')
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights('model_new3.h5')
        print('loaded model from disk')

        # remains to compile the model : 
        loaded_model.compile(loss='categorical_crossentropy',
                        optimizer='adadelta',
                        metrics=['accuracy'])

        return loaded_model
    else :                          
        print('building model...')
        input_shape = (64,64,3)
        model = Sequential()

        model.add(Convolution2D(32, 5, 5, border_mode='same',
                                input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Convolution2D(3, 4, 4, border_mode='same'))
        model.add(Activation('relu'))

        model.add(Convolution2D(3, 3, 3, border_mode='same'))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(3, 3, 3, border_mode='same'))
        model.add(Activation('relu'))
        
        model.compile(loss='mse',
                optimizer='rmsprop',
                metrics=['accuracy'])

        return model

def save_model(model):
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
    print("saved model to disk")



# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

X_train, y_train, X_valid, Y_valid, X_test, y_test = load_dataset()

model = keras_model(load=False)
#plot(model, show_shapes=True, to_file='3dcnn.png')

import pdb; #pdb.set_trace()
for i in range(3): # restart at 6
    model.fit(X_train, y_train, batch_size=128, nb_epoch=1,
          verbose=1)

    pred = model.predict(X_valid)
    
    pred_0 = (pred[0]).astype('uint8')
    Image.fromarray(pred_0).show()
    Image.fromarray(Y_valid[0]).show()
    Image.fromarray(X_valid[0]).show()
    pdb.set_trace()

save_model(model)
pdb.set_trace()


