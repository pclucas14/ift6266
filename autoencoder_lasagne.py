# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 11:54:25 2017

@author: pcluc
"""


import numpy as np
import numpy.random as rng
from PIL import Image
import os
#os.environ[ 'CUDA_LAUNCH_BLOCKING' ] = '1'
#os.environ[ 'THEANO_FLAGS' ] = 'mode=DebugMode'

import theano
import theano.tensor as T
from helper import *
from keras.datasets import mnist
import lasagne
from model import *

# theano.config.exception_verbosity = 'high'
# theano.config.allow_gc = False
# theano.config.openmp = True

# Hyperparameters :   
batch_size = 256
num_epochs = 50

input_ = T.tensor4()
output_ = T.tensor4()

lambda_sqr = 0.5
lambda_cr = 0.5
clip = 0.01

#%%
model = Model(version=1, batch_size=batch_size)
auto_encoder = model.model['output']
critic = model.critic['output']

ae_output = lasagne.layers.get_output(auto_encoder)
critic_output = lasagne.layers.get_output(critic)

# Create expression for passing real data through the critic
real_out = lasagne.layers.get_output(critic)

# Create expression for passing fake data through the critic
# TODO : modify this to only pass in the modified version
#critic_input = 
fake_out = lasagne.layers.get_output(critic,
                            inputs=ae_output)
'''
Optionally, you can define the input(s) to propagate through the network instead 
of using the input variable(s) associated with the network’s input layer(s).
'''
# Create score expressions to be maximized (i.e., negative losses)
ae_score = fake_out.mean()

critic_score = real_out.mean() - fake_out.mean()

# Create update expressions for training
ae_params = lasagne.layers.get_all_params(auto_encoder, trainable=True)

critic_params = lasagne.layers.get_all_params(critic, trainable=True)

critic_loss = -critic_score

ae_sqr_loss = lasagne.objectives.squared_error(ae_output,
                                           model.output_)
ae_cr_loss = -ae_score

ae_loss =  lambda_sqr * T.mean(ae_sqr_loss) + lambda_cr * ae_cr_loss
#%%

ae_updates = lasagne.updates.adam(ae_loss, ae_params)
critic_updates = lasagne.updates.adam(critic_loss, critic_params)

# Clip critic parameters in a limited range around zero (except biases)
for param in lasagne.layers.get_all_params(critic, trainable=True,
                                           regularizable=True):
        critic_updates[param] = T.clip(critic_updates[param], -clip, clip)



train_critic = theano.function(inputs=[model.input_, model.input_c], 
                               outputs=[critic_loss], 
                               updates=critic_updates,
                               name='train_critic')

train_ae = theano.function(inputs=[model.input_, model.output_], 
                           outputs=[ae_loss, ae_output], 
                           updates = ae_updates,
                           name='train_ae')

test_ae = theano.function(inputs=[model.input_, model.output_], 
                           outputs=[ae_loss, ae_output], 
                           name='test_ae')

#%%

trainx, trainy, testx, testy, _, _ = load_dataset(sample=False)
'''
trainx = trainx.reshape((-1, 3, 64, 64)).astype('float32')
trainy = trainy.reshape((-1,3,64, 64)).astype('float32')
testx = trainx.reshape((-1, 3, 64, 64)).astype('float32')
testy = trainy.reshape((-1,3,64, 64)).astype('float32')
'''
trainx = np.transpose(trainx, axes=[0,3,1,2]).astype('float32')
trainy = np.transpose(trainy, axes=[0,3,1,2]).astype('float32')
testx = np.transpose(testx, axes=[0,3,1,2]).astype('float32')
testy = np.transpose(testy, axes=[0,3,1,2]).astype('float32')
#%%
num_batches = trainx.shape[0] / batch_size

for epoch in range(0,60) : 
    print epoch
    for i in range(num_batches-1):
        batch_x = trainx[i*batch_size : (i+1)*batch_size, :, :, :]
        batch_y = trainy[i*batch_size : (i+1)*batch_size, :, :, :] 
        
        if i == num_batches-2 :
            # test perfomance
            _, test_pred_ae = test_ae(batch_x, batch_y)
            out_img = combine_tensors(batch_y, test_pred_ae)
            out_img = np.transpose(out_img, axes=[0,2,3,1])
            saveImage(out_img.astype('uint8'), 'test', epoch) 
            
        else :     
            loss_ae, pred_ae = train_ae(batch_x, batch_y)
            loss_cr = train_critic(batch_y, pred_ae)
            
            if i == num_batches-3 :
                out_img = combine_tensors(batch_y, pred_ae)
                out_img = np.transpose(out_img, axes=[0,2,3,1])
                saveImage(out_img.astype('uint8'), 'train', epoch) 
                
        if epoch % 5 == 4 : model.save(epoch=epoch)
            
    print loss_ae
#%%
z = combine_tensors(trainy[0:batch_size], trainy[2*batch_size: 3*batch_size])
z = np.transpose(z, axes=[0,2,3,1])
x = Image.fromarray(z[10].astype('uint8')).show()
            #img.show()   


