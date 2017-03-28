# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 11:54:25 2017
@author: pcluc
"""


import numpy as np
import numpy.random as rng
from PIL import Image
import os
import logging
#os.environ[ 'CUDA_LAUNCH_BLOCKING' ] = '1'
#os.environ[ 'THEANO_FLAGS' ] = 'mode=DebugMode'

import theano
import theano.tensor as T
import lasagne
import lasagne.layers as ll
from helper import *
from GAN import *

# theano.config.exception_verbosity = 'high'
# theano.config.allow_gc = False
# theano.config.openmp = True

# Hyperparameters :   
batch_size = 64
num_epochs = 50
version = '_c2'
name = 'lsgan' + str(version)
load_params = False 

initial_eta = 1e-4
encode_input = False

logging.basicConfig(filename=(name + ".log"), level=logging.INFO)
logging.info('Logging start')
home_dir = '/home2/ift6ed47/' #'/home/lucas/Desktop/'

GAN = GAN(version=2, batch_size=batch_size, encode=encode_input)
generator = GAN.generator[-1]
critic = GAN.critic[-1]

if load_params:
    logging.info('loaded params from file'); print 'loading params from file' 
    last_saved_epoch = '10'
    update_model_params(generator, home_dir + 'models/' + name + '_gen_' + last_saved_epoch + '.npz')
    update_model_params(critic, home_dir + ' models/' + name + '_disc_' + last_saved_epoch + '.npz')
    logging.info('parameters successfully loaded'); print 'parameters successfully loaded' 	

#%%
logging.info('param setup')
gen_output = ll.get_output(generator)
gen_params = ll.get_all_params(generator, trainable=True)

eta = theano.shared(lasagne.utils.floatX(initial_eta))

critic_output = ll.get_output(critic)

# Create expression for passing real data through the critic
real_out = ll.get_output(critic, inputs=GAN.input_c)


critic_input = combine_tensor_images(GAN.input_, gen_output, batch_size)
fake_out = ll.get_output(critic, inputs=critic_input)

'''
# hidden layers for feature matching
hid_real = ll.get_output(GAN.critic[-3], inputs=GAN.input_c, deterministic=False)
hid_fake = ll.get_output(GAN.critic[-3], inputs=critic_input, deterministic=False)
m1 = T.mean(hid_real,axis=0)
m2 = T.mean(hid_fake,axis=0)
loss_gen_fm = lasagne.objectives.squared_error(m1, m2).mean()
'''


# border reconstruction
loss_gen_border = lasagne.objectives.squared_error(GAN.input_ * (1. - GAN.mask), gen_output * (1. - GAN.mask)).mean()
gen_border_updates = lasagne.updates.adam(loss_gen_border, gen_params, learning_rate=eta)


a, b, c = 0, 1, 1

# Create update expressions for training
critic_params = ll.get_all_params(critic, trainable=True)
critic_loss = (lasagne.objectives.squared_error(real_out, b).mean() +
		 lasagne.objectives.squared_error(fake_out, a).mean()) 


gen_loss = lasagne.objectives.squared_error(fake_out, c).mean() 

print 'loss and function setup'
logging.info('loss and function setup')

gen_grads = theano.grad(gen_loss, wrt=gen_params)
critic_grads = theano.grad(critic_loss, wrt=critic_params)

gen_grads_norm = sum(T.sum(T.square(grad)) for grad in gen_grads) / len(gen_grads)
critic_grads_norm = sum(T.sum(T.square(grad)) for grad in critic_grads) / len(critic_grads)

gen_updates= lasagne.updates.adam(
    gen_grads, gen_params, learning_rate=eta)

critic_updates = lasagne.updates.adam(
    critic_grads, critic_params, learning_rate=eta)

train_critic = theano.function(inputs=[GAN.input_, GAN.input_c], 
                               outputs=[real_out.mean(),
                                        fake_out.mean(),
                                        critic_loss,
                                        critic_grads_norm], 
                               updates=critic_updates,
                               name='train_critic')

train_gen = theano.function(inputs=[GAN.input_], #input_c is for feature matching
                           outputs=[(fake_out > .5).mean(), 
                                    gen_grads_norm,
                                    gen_loss, 
                                    gen_output], 
                           updates = gen_updates,
                           name='train_gen')

train_gen_border = theano.function(inputs=[GAN.input_], 
		                   outputs=[loss_gen_border],
				   updates = gen_border_updates,
			           name='train_border_gen')

test_gen = theano.function(inputs=[GAN.input_],
                           outputs=[critic_input, gen_output], 
                           name='test_gen')

#%%
logging.info('loading data')
trainx, trainy, trainz, testx, testy, testz = load_dataset(sample=False, extra=4, normalize=True)
logging.info('data loaded')

batches = iterate_minibatches(trainx, trainz, batch_size, shuffle=True,
                              forever=True)

input_test, target_test = testx[:batch_size], testz[:batch_size]


num_batches = trainx.shape[0] / batch_size

for epoch in range(0, 3000) :
    print epoch
    updates_gen = 0
    updates_critic = 0
    disc_err = 0
    gen_err = 0
    gen_iter = 1
    border_epoch = 50   
    critic_iter = 1 if epoch > border_epoch else 0    

    for _ in range(50):
        # train autoencoder
        for _ in range(gen_iter):
            _, target = next(batches)
	    if epoch < border_epoch : 
		gen_err += np.array(train_gen_border(target))
	    else : 
            	acc_gen, gen_grad_norm, loss_gen, pred = train_gen(target)#(input, target)#, target)
            	gen_err += np.array([acc_gen, gen_grad_norm, loss_gen])
            updates_gen += 1

	# train discriminator
        for _ in range(critic_iter):
            _, target = next(batches)
            disc_err += np.array([train_critic(target, target)])
            #critic_clip_fn()
	    updates_critic += 1

    # test it out 
    samples, raw_samples = test_gen(target_test)#(input_test)
    samples = samples * 88.3 + 86.3
    raw_samples = raw_samples * 88.3 + 86.3
    # result = samples*77.3 + 86.3#fill_middle_extra(input_test, samples)* 77.3 + 86.3

    raw_result = raw_samples.transpose(0,2,3,1).astype('uint8')
    result = samples.transpose(0,2,3,1).astype('uint8')
    
    saveImage(result, 'samples_' + name, epoch) 
    saveImage(raw_result, 'samples_raw_' + name, epoch)

    if epoch % 25 == 0 :
        np.savez(home + 'models/' + str(name) + '_disc_' + str(epoch) + '.npz', *ll.get_all_param_values(critic))
        np.savez(home + 'models/' + str(name) + '_gen_' + str(epoch) + '.npz', *ll.get_all_param_values(generator))
    
    # Then we print the results for this epoch:

    logging.info("epoch : " + str(epoch))
    logging.info("  generator loss:\t\t{}".format(gen_err / updates_gen))
    print("  generator loss:\t\t{}".format(gen_err / updates_gen))
    
    if epoch > border_epoch : 
    	print("  discriminator loss:\t\t{}".format(disc_err / updates_critic))
    	logging.info("  discriminator loss:\t\t{}".format(disc_err / updates_critic))

          

