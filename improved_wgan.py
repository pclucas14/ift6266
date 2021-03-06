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


# Hyperparameters :   
batch_size = 64
num_epochs = 50
version = ''
name = 'improved_wgan' + str(version)
load_params = False
last_saved_epoch = 0
initial_eta = 1e-4

logging.basicConfig(filename=(name + ".log"), level=logging.INFO)
logging.info('Logging start')
home_dir = '/home/ml/lpagec/' 

GAN = GAN(version=2, batch_size=batch_size, encode=False)
generator = GAN.generator[-1] 
critic = GAN.critic[-1]

if load_params:
    logging.info('loaded params from file'); print 'loading params from file' 
    last_saved_epoch_str = str(last_saved_epoch)
    path_gen =  'gen_eor1.npz'#name + '_gen_' + last_saved_epoch_str + '.npz'
    path_disc = 'disc_eor1.npz'#name + '_disc_'+ last_saved_epoch_str + '.npz'
    update_model_params(generator, home_dir + 'models/' + path_gen)
    update_model_params(critic, home_dir + 'models/' + path_disc) 
    logging.info('parameters successfully loaded'); print 'parameters successfully loaded' 	
    version = '_c1_adam_'

#%%
logging.info('param setup')
gen_output = ll.get_output(generator)
gen_params = ll.get_all_params(generator, trainable=True)
eta = theano.shared(lasagne.utils.floatX(initial_eta))

critic_output = ll.get_output(critic)

# Create expression for passing real data through the critic
real_out = ll.get_output(critic, inputs=GAN.input_c)
critic_input = gen_output#combine_tensor_images(GAN.input_, gen_output, batch_size)
fake_out = ll.get_output(critic, inputs=critic_input)

generator_score = fake_out.mean()
critic_score = real_out.mean() - fake_out.mean() 

# Create update expressions for training
critic_params = ll.get_all_params(critic, trainable=True)

print 'loss and function setup'
logging.info('loss and function setup')

critic_loss = - critic_score 
gen_loss = -generator_score

#%%
gen_grads = theano.grad(gen_loss, wrt=gen_params)
critic_grads = theano.grad(critic_loss, wrt=critic_params)

# get gradient norm to apply gradient l2 penalty (adapted from Ishaan's repo)
differences = fake_out - real_out
interpolates = real_data + (ALPHA*differences)

import pdb; pdb.set_trace()

gen_grads_norm = sum(T.sum(T.square(grad)) for grad in gen_grads) / len(gen_grads)
critic_grads_norm = sum(T.sum(T.square(grad)) for grad in critic_grads) / len(critic_grads)

gen_updates= lasagne.updates.rmsprop(
    gen_grads, gen_params, learning_rate=initial_eta)

critic_updates = lasagne.updates.rmsprop(
    critic_grads, critic_params, learning_rate=initial_eta)


train_critic = theano.function(inputs=[GAN.input_, GAN.input_c], 
                               outputs=[(real_out ).mean(),
                                        (fake_out ).mean(),
                                        critic_loss,
                                        critic_grads_norm], 
                               updates=critic_updates,
                               name='train_critic', on_unused_input='warn')

train_gen = theano.function(inputs=[GAN.input_, GAN.input_c],#@input_c is for feature matching
                           outputs=[(fake_out).mean(), 
                                    gen_grads_norm,
                                    gen_loss, 
                                    gen_output], 
                           updates = gen_updates,
                           name='train_gen', on_unused_input='warn')
'''
train_gen_border = theano.function(inputs=[GAN.input_], 
		                   outputs= [loss_gen_border],
				   updates = gen_border_updates,
			           name='train_border_gen', on_unused_input='warn')
'''
test_gen = theano.function(inputs=[GAN.input_],
                           outputs=[critic_input, gen_output], 
                           name='test_gen', on_unused_input='warn')


#import pdb; pdb.set_trace()
logging.info('loading data')
#trainz, testz = load_dataset(sample=False, extra=4, normalize=True)
dataset, dataset_size = load_dataset(sample=False, normalize=True)
num_batches = dataset_size / batch_size - 2
batches = iterate_minibatches(dataset[:num_batches * batch_size], batch_size, shuffle=True, forever=True)

logging.info('data loaded')
last_index = num_batches + 1

for epoch in range(0, 30000) :
    print epoch
    updates_gen = 0
    updates_critic = 0
    updates_critic_small = 0
    disc_err = 0
    disc_err_small = 0
    gen_err = 0
    gen_iter = 1
    border_epoch = 0   
    critic_iter = 1 #if epoch > border_epoch else 0    
    index_int = 0
    step = 0.08
    pit_stop = 150

    for _ in range(50):
        # train generator
	target = next(batches)
        for _ in range(gen_iter):
	    if epoch < border_epoch : 
		gen_err += np.array(train_gen_border(target))
	    else : 
            	#if updates_gen % 10 == 0 : train_gen_border(index_int)
		acc_gen, gen_grad_norm, loss_gen, pred = train_gen(target, target)#(input, target)#, target)
            	gen_err += np.array([acc_gen, loss_gen, gen_grad_norm])
	        updates_gen += 1

	# train discriminator on whole image 
        for _ in range(critic_iter):
	    #target = next(batches)
	    disc_err += np.array([train_critic(target, target)])
	    updates_critic += 1
	'''
	# train small disc on center only 
	for _ in range(0):
	    disc_err_small += np.array(train_critic_small(target, target))
	    updates_critic_small += 1
   	'''

    # update mixing coefficients
    if epoch % pit_stop == 0: 
        #import pdb; pdb.set_trace()
        GAN.update_mixing_coefs(step=0.08)
    
    # test it out 
    for i in range(1):
        test_batch = target#np.tile(target[i].reshape((1, 3, 64, 64)), (batch_size, 1, 1, 1))
        samples, raw_samples = test_gen(test_batch)
        samples = samples * 67.75 + 113.91
        raw_samples = raw_samples * 67.75 + 113.91
        raw_result = raw_samples.transpose(0,2,3,1).astype('uint8')
        result = samples.transpose(0,2,3,1).astype('uint8')
    
        saveImage(result, 'samples2_' + str(i) + '_' +  name, epoch, side=8) 
    #isaveImage(raw_result, 'samples2_raw_' + name, epoch)

    if epoch % 25 == 0 :
        np.savez(home + 'models/' + str(name) + '_disc_' + str(epoch) + '.npz', *ll.get_all_param_values(critic))
        np.savez(home + 'models/' + str(name) + '_gen_' + str(epoch) + '.npz', *ll.get_all_param_values(generator))
    
    # Then we print the results for this epoch:

    logging.info("epoch : " + str(epoch))
    logging.info("  generator loss:\t\t{}".format(gen_err / updates_gen))
    print("  generator loss:\t\t{}".format(gen_err / updates_gen))
    
    if epoch > border_epoch : 
        print("  discriminator loss:\t\t{}".format(disc_err / updates_critic))
    	#print("  discriminator small loss:\t\t{}".format(disc_err_small / updates_critic_small))
    	#logging.info("  discriminator loss:\t\t{}".format(disc_err / updates_critic))


