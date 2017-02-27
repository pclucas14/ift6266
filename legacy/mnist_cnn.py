# -*- coding: utf-8 -*-

import theano
import theano.tensor as T
import numpy as np
import numpy.random as rng
from keras.datasets import mnist
import lasagne

#%%

# 1) Hyperparameters : 
scale = 0.01   
batch_size = 512
num_epochs = 50
cnn = True
 
    
# 2) creating necessary placeholders
input_tensor = T.tensor4()
output_labels = T.ivector()

# 3) creating shared variables
model = dict()

if cnn : 
    model['W1'] = theano.shared(scale * rng.normal(size = (32,1,4,4)).astype('float32'))
    model['b1'] = theano.shared(scale * np.zeros((32,)).astype('float32'))
    model['W2'] = theano.shared(scale * rng.normal(size = (64,32,4,4)).astype('float32'))
    model['b2'] = theano.shared(scale * np.zeros((64,)).astype('float32'))
    model['W3'] = theano.shared(scale * rng.normal(size = (64*7*7,10)).astype('float32'))
    model['b3'] = theano.shared(scale * np.zeros((10,)).astype('float32'))
    
    h = T.nnet.conv2d(input_tensor, model['W1'], border_mode='half')
    h = T.nnet.relu(h + model['b1'].dimshuffle('x', 0, 'x', 'x'))
    h = T.signal.pool.pool_2d(h, (2,2), ignore_border=True)
    
    h = T.nnet.conv2d(h, model['W2'], border_mode='half')
    h = T.nnet.relu(h + model['b2'].dimshuffle('x', 0, 'x', 'x'))
    h = T.signal.pool.pool_2d(h, (2,2),ignore_border=True)
       
    h = T.dot(T.flatten(h,outdim=2),model['W3'])
    output = T.nnet.softmax(h + model['b3'])
    
else : 
    model['W1'] = theano.shared(scale * rng.normal(size = (784,512)).astype('float32'))
    model['b1'] = theano.shared(scale * np.zeros((512,)).astype('float32'))
    model['W2'] = theano.shared(scale * rng.normal(size = (512,100)).astype('float32'))
    model['b2'] = theano.shared(scale * np.zeros((100,)).astype('float32'))
    model['W3'] = theano.shared(scale * rng.normal(size = (100,10)).astype('float32'))
    model['b3'] = theano.shared(scale * np.zeros((10,)).astype('float32'))
    
    # 4) creating model 
    two_d_input = T.flatten(input_tensor, outdim=2)
    pre_act = T.dot(two_d_input, model['W1']) + model['b1']
    act = T.nnet.relu(pre_act)
    
    pre_act = T.dot(act, model['W2']) + model['b2']
    act = T.nnet.relu(pre_act)
    
    pre_act = T.dot(act, model['W3']) + model['b3']
    output = T.nnet.softmax(pre_act)

#%%
# 5) loss function
cross_entropy_train = -T.mean(T.log(output)[T.arange(output_labels.shape[0]),output_labels])
accuracy_train = T.mean(T.eq(T.argmax(output, axis=1), output_labels))

updates = lasagne.updates.adam(cross_entropy_train, model.values())

train_function = theano.function(inputs=[input_tensor, output_labels], outputs=[cross_entropy_train, accuracy_train], updates=updates, name='train_fct')
test_function = theano.function(inputs=[input_tensor, output_labels], outputs=[cross_entropy_train, accuracy_train], name='test_fct')

#%%
# 6) training

(trainx, trainy), (validx, validy) = mnist.load_data()
trainx = trainx.reshape((-1, 1, 28, 28)).astype('float32')
validx = validx.reshape((-1, 1, 28, 28)).astype('float32')

#%% training
num_batches = trainx.shape[0] / batch_size

for epoch in range(num_epochs) : 
    for i in range(num_batches):
        batch_x = trainx[i*batch_size : (i+1)*batch_size, :, :]
        batch_y = trainy[i*batch_size : (i+1)*batch_size]
        loss_train, acc_train = train_function(batch_x, batch_y)
        print loss_train, acc_train
    
#%% testing
for i in range(validx.shape[0] / batch_size):
    batch_x = validx[i*batch_size : (i+1)*batch_size, :, :]
    batch_y = validy[i*batch_size : (i+1)*batch_size]
    loss_test, acc_test = test_function(batch_x, batch_y)
    print loss_test, acc_test




