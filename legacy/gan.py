import keras.models as models
from keras.models import Model
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Input,merge
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten,MaxoutDense
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD, Adam
from keras.datasets import mnist
from sklearn.utils import shuffle
import numpy as np
from PIL import Image
import argparse
import math
import os
import random
import theano
from cnn import load_dataset
import matplotlib
import matplotlib.pyplot as plt
import pdb
matplotlib.use('Agg')

#s.environ["THEANO_FLAGS"] = "exception_verbosity=high"
#theano.config.compute_test_value = 'warn'

use_mnist = True
threshold_update = 0.03
save_model = True
load_weights = False
verbose = True
coffee_break = 10

################################################################################
#######################          MODEL CREATION          #######################
################################################################################

def generator_model(input_noise, load_weights=False):
    # input is a 1d tensor (vector in Theano) used as noise for
    # image generation. Expected shape : (100,)
    if use_mnist : 
        hid_chan_mult = 14
        output_chan_size = 1
    else : 
        hid_chan_mult = 16
        output_chan_size = 3
    
    num_hidden_channels = 50 * 3
    model = Dense(num_hidden_channels*hid_chan_mult*hid_chan_mult, init='glorot_normal')(input_noise)
    model = BatchNormalization(mode=2,axis=1)(model)
    model = Activation('relu')(model)
    model = Reshape([num_hidden_channels, hid_chan_mult, hid_chan_mult])(model)
    model = UpSampling2D(size=(2,2), dim_ordering='th')(model)
    model = Convolution2D(int(num_hidden_channels/2), 3, 3, border_mode='same', 
                    init='glorot_uniform', dim_ordering='th')(model)
    model = BatchNormalization(mode=2,axis=1)(model)
    model = Activation('relu')(model)
    model = Convolution2D(int(num_hidden_channels/4), 3, 3, border_mode='same', 
                    init='glorot_uniform', dim_ordering='th')(model)
    model = BatchNormalization(mode=2,axis=1)(model)
    model = Activation('relu')(model)
    model = Convolution2D(output_chan_size, 1, 1, border_mode='same', 
                    init='glorot_uniform', dim_ordering='th')(model)
    model = Activation('tanh')(model)
    generator = Model(input_noise,model)
    generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4))
    #generator.summary()
    if load_weights:
        path = 'models/generator_2.h5'
        generator.load_weights(path)
        print('loaded generator weights from cache.')
    return generator


def discriminator_model(generated_image,load_weights=False):
    H = Convolution2D(256, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu')(generated_image)
    H = LeakyReLU(0.2)(H)
    H = Dropout(0.25)(H)
    H = Convolution2D(512, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu')(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(0.25)(H)
    H = Flatten()(H)
    H = Dense(256)(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(0.25)(H)
    d_V = Dense(1,activation='sigmoid')(H)
    discriminator = Model(generated_image,d_V)
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-5))
    # discriminator.summary()
    if load_weights:
        path = 'models/discriminator_?.h5'
        generator.load_weights(path)
        print('loaded discriminator weights from cache.')
    return discriminator


def gan_model(generator_input, discriminator_output):
    gan = Model(generator_input, discriminator_output)
    gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-5), metrics=['accuracy'])
    #gan.summary()
    return gan


########################################################
#####           Useful helper methods             ######
########################################################

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

def pre_train_discriminator(discriminator, generator, sample_size, dataX):
    trainidx = random.sample(range(0,dataX.shape[0]), sample_size)
    XT = X_train[trainidx,:,:,:]
    noise_gen = np.random.uniform(0,1,size=[XT.shape[0],100])
    generated_images = generator.predict(noise_gen)
    X = np.concatenate((XT, generated_images))
    n = XT.shape[0]
    y = [1] * n + [0] * n

    make_trainable(discriminator,True)
    discriminator.fit(X,y, nb_epoch=1, batch_size=128)
    y_hat = discriminator.predict(X)

def saveImage(imageData, imageName, epoch):
	f, ax = plt.subplots(16, 8)
	k = 0
	for i in range(16):
		for j in range(8):
			pltImage = imageData[k][0]
			ax[i,j].imshow(pltImage, interpolation='nearest',cmap='gray_r')
			ax[i,j].axis('off')
			k = k+1
	f.set_size_inches(18.5, 10.5)
	f.savefig('images/'+imageName+'_after_'+str(epoch)+'_epoch.png', dpi = 100, bbox_inches='tight', pad_inches = 0)
	plt.close(f)
	return None

######################################################
######################################################

gen_input = Input(shape=[100])
if use_mnist:
     disc_input = Input(shape=(1,28,28))
else:
    disc_input = Input(shape=(3,32,32))
gan_input = Input(shape=[100])

# creating generator
generator = generator_model(gen_input,load_weights=load_weights)

# creating discriminator
discriminator = discriminator_model(disc_input)

# Freeze weights in the discriminator for stacked training
make_trainable(discriminator, False)

# feeding input into generator (get a TENSOR instead of model)
generator_T = generator(gan_input)

# connecting gen_output to disc_input (get a TENSOR instead of model)
gan_T = discriminator(generator_T)

# TODO : Q : so as soon as you feed a tensor to a model, (line above, said model becomes tensor?)
# wrapping up
GAN = gan_model(gan_input, gan_T)


if not os.path.exists('images'):
    os.makedirs('images')
if not os.path.exists('models'):
    os.makedirs('models')
if not os.path.exists('metrics'):
    os.makedirs('metrics')

dLoss = []
gLoss = []
batchSize = 128
nbEpoch = 200
decayIter = 100

if use_mnist : 
    (X_train, y_train), _ = mnist.load_data()
    X_train = X_train.reshape((-1,1,28,28))
    X_train = X_train.astype('float32')
    X_train -= - 127.5
    X_train /= 127.5
    print(str(X_train.shape[0]) + ' MNIST images in training data set.')
    pre_train_discriminator(discriminator, generator, int(X_train.shape[0] / 10), X_train)

else : 
    Y_train, X_train, Y_test, X_test, _, _ = load_dataset(ds_split=(0.8,0.2,0.), shuffle=True)

numExamples = (X_train.shape)[0]
numBatches = int(numExamples/float(batchSize))

# generating noise from input
X_noise = []
npRandom = np.random.RandomState(18)
for i in range(X_train.shape[0]):
    randomNoise = npRandom.uniform(-1,1,100)
    X_noise.append(randomNoise)
X_noise = np.array(X_noise)
print('Random Noise Data: ', X_noise.shape)

for epoch in range(1, nbEpoch + 1):
    print('Epoch: ', epoch)

    for i in range(numBatches):

        noisePredictBatch = X_noise[np.random.randint(numExamples, size = batchSize)]
        noiseDataBatch = generator.predict(noisePredictBatch)
        origDataBatch = X_train[batchSize*i:batchSize*(i+1)]
        #[np.random.randint(numExamples, size = batchSize)]
        noiseLabelsBatch, origLabelsBatch = np.zeros(batchSize).astype(int), np.ones(batchSize).astype(int)

        trainBatch = np.concatenate((noiseDataBatch, origDataBatch), axis = 0)
        trainLabels = np.concatenate((noiseLabelsBatch, origLabelsBatch))
        trainBatch, trainLabels = shuffle(trainBatch, trainLabels)

        # to make sure the discriminator does not become too powerful, only perform update when disc loss if big enough
        # note that the discriminator has already been pre trained to have a general idea 

        if len(dLoss) == 0 or ((len(dLoss) > 0 and dLoss[-1] > threshold_update)) or np.random.rand() < 0.1 : 
            discriminatorLoss = discriminator.train_on_batch(trainBatch, trainLabels)
            dLoss.append(discriminatorLoss)
            if verbose : print('discriminator loss : ' + str(discriminatorLoss))

        dcganLabels = np.ones(batchSize).astype(int)

        make_trainable(discriminator, False)
        dcganLoss = GAN.train_on_batch(noisePredictBatch, dcganLabels)
        if verbose : print ('dcgan Loss: ', dcganLoss)
        make_trainable(discriminator, True)

        gLoss.append(dcganLoss)

    print('after epoch: ', epoch)
    saveImage(noiseDataBatch, 'generated', epoch)

    
    if (epoch % coffee_break == 1):  
        #pdb.set_trace()   
        generator.save('models/generator_'+str(epoch)+'.h5')
        discriminator.save('models/discriminator_'+str(epoch)+'.h5')
    '''
    if epoch > decayIter :
        lrD = discriminator.optimizer.lr.get_value()
        lrG = generator.optimizer.lr.get_value()
        discriminator.optimizer.lr.set_value((lrD - lr/decayIter).astype(np.float32))
        generator.optimizer.lr.set_value((lrG - lr/decayIter).astype(np.float32))
        print('learning rate linearly decayed')
    '''
    #np.save('metrics/dLoss.npy', np.array(dLoss))
    #np.save('metrics/gLoss.npy', np.array(gLoss))



