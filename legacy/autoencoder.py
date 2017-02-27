# -*- coding: utf-8 -*-
import os
#os.environ[ 'CUDA_LAUNCH_BLOCKING' ] = '1'
os.environ[ 'THEANO_FLAGS' ] = 'mode=DebugMode'
import theano
import theano.tensor as T
import numpy as np
import numpy.random as rng
from keras.datasets import mnist
import lasagne
from helper import load_dataset

theano.config.exception_verbosity = 'high'
theano.config.allow_gc = False
theano.config.openmp = True

# Hyperparameters : 
scale = 0.01   
batch_size = 20
num_epochs = 50

input_ = T.tensor4()
output_ = T.tensor4()
filter_size = 3

#%%
model = dict()
# encoder  weights
model['W1'] = theano.shared(scale * rng.normal(size = (32,3,filter_size,filter_size)).astype('float32'), name='w1')
model['b1'] = theano.shared(scale * np.zeros((32,)).astype('float32'), name='b1')
model['W2'] = theano.shared(scale * rng.normal(size = (64,32,filter_size,filter_size)).astype('float32'), name='w2')
model['b2'] = theano.shared(scale * np.zeros((64,)).astype('float32'), name='b2')
model['W3'] = theano.shared(scale * rng.normal(size = (256,64,filter_size,filter_size)).astype('float32'), name='w3')
model['b3'] = theano.shared(scale * np.zeros((256,)).astype('float32'), name='b3')

# decoder weights
model['b3_t'] = theano.shared(scale * np.zeros((64,)).astype('float32'), name='b3_t')
model['W3_t'] = theano.shared(scale * rng.normal(size = (64,256,filter_size,filter_size)).astype('float32'), name='w3_t')
model['b2_t'] = theano.shared(scale * np.zeros((32,)).astype('float32'), name='b2_t')
model['W2_t'] = theano.shared(scale * rng.normal(size = (32,64,filter_size,filter_size)).astype('float32'), name='w2_t')
model['b1_t'] = theano.shared(scale * np.zeros((3,)).astype('float32'), name='b1_t')
model['W2_t'] = theano.shared(scale * rng.normal(size = (3,32,filter_size,filter_size)).astype('float32'), name='w2_t')

# batch normalization parameters
#model['beta'] = theano.shared(np.zeros((1,1,1,1)).astype('float32'), name='beta')
#model['gamma'] = theano.shared(np.ones((1,1,1,1)).astype('float32'), name='gamma')

#%%
# encoder 
h = T.nnet.conv2d(input_, model['W1'], input_shape=(batch_size, 3, 64, 64), filter_shape=(32, 3, filter_size, filter_size), border_mode='half', subsample=(2, 2))
h = h + model['b1'].dimshuffle('x', 0, 'x', 'x')
#h = T.nnet.bn.batch_normalization(h , model['gamma'], model['beta'], h.mean(0, keepdims=True), h.mean(0, keepdims=True))
h = T.nnet.relu(h)

h = T.nnet.conv2d(h, model['W2'], input_shape=(batch_size, 32, 32, 32), filter_shape=(64, 32, filter_size, filter_size), border_mode='half', subsample=(2, 2))
h = h + model['b2'].dimshuffle('x', 0, 'x', 'x')
#h = T.nnet.bn.batch_normalization(h, model['gamma'], model['beta'], h.mean(0, keepdims=True), h.mean(0, keepdims=True))
h = T.nnet.relu(h)

h = T.nnet.conv2d(h, model['W3'], input_shape=(batch_size, 64, 16, 16), filter_shape=(256, 64, filter_size, filter_size), border_mode='half', subsample=(2, 2))
h = h + model['b3'].dimshuffle('x', 0, 'x', 'x')
#h = T.nnet.bn.batch_normalization(h, model['gamma'], model['beta'], h.mean(0, keepdims=True), h.mean(0, keepdims=True))
h = T.nnet.relu(h)

# decoder 
h = T.nnet.abstract_conv.conv2d_grad_wrt_inputs(h, model['W3_t'], (batch_size, 256, 8, 8), filter_shape=(64,256,filter_size,filter_size), border_mode='half', subsample=(2, 2))
h = h + model['b3_t'].dimshuffle('x', 0, 'x', 'x')
#h = T.nnet.bn.batch_normalization(h, model['gamma'], model['beta'], h.mean(0, keepdims=True), h.mean(0, keepdims=True))
h = T.nnet.relu(h)

h = T.nnet.abstract_conv.conv2d_grad_wrt_inputs(h, model['W2_t'], (batch_size, 64, 16, 16), filter_shape=(32,64,filter_size,filter_size), border_mode='half', subsample=(2, 2))
h = h + model['b2_t'].dimshuffle('x', 0, 'x', 'x')
#h = T.nnet.bn.batch_normalization(h , model['gamma'], model['beta'], h.mean(0, keepdims=True), h.mean(0, keepdims=True))
h = T.nnet.relu(h)

h = T.nnet.abstract_conv.conv2d_grad_wrt_inputs(h, model['W2_t'], (batch_size, 32, 32, 32), filter_shape=(3,32,filter_size,filter_size), border_mode='half', subsample=(2, 2))
output = T.tanh(h + model['b1_t'].dimshuffle('x', 0, 'x', 'x'))

#%%
loss = T.mean(T.sqr(output - output_))
updates = lasagne.updates.adam(loss, model.values())

train_function = theano.function(inputs=[input_, output_], outputs=[loss, output], updates=updates, name='train_fct')
test_function = theano.function(inputs=[input_, output_], outputs=[loss, output], name='test_fct')

#%%

trainx, trainy, testx, testy, _, _ = load_dataset()
trainx = trainx.reshape((-1, 3, 64, 64)).astype('float32')
trainy = trainy.reshape((-1,3,64, 64)).astype('float32')
testx = trainx.reshape((-1, 3, 64, 64)).astype('float32')
testy = trainy.reshape((-1,3,64, 64)).astype('float32')

#%%
num_batches = trainx.shape[0] / batch_size

for epoch in range(num_epochs) : 
    for i in range(num_batches):
        batch_x = trainx[i*batch_size : (i+1)*batch_size, :, :, :]
        batch_y = trainy[i*batch_size : (i+1)*batch_size, :, :, :]
        loss_train, predictions = train_function(batch_x, batch_y)
        print loss_train
    
    Image.fromarray(trainx[-1]).show()
    Image.fromarray(predictions[-1]).show()
    

