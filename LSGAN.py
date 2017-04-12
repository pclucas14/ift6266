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
version = 'cond_ae_full'
name = 'lsgan' + str(version)
load_params = True
last_saved_epoch = 325
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
    path_gen = name + '_gen_' + last_saved_epoch_str + '.npz'
    path_disc = name + '_disc_'+ last_saved_epoch_str + '.npz'
    update_model_params(generator, home_dir + 'models/' + path_gen)
    #update_model_params(critic, home_dir + 'models/' + path_disc) 
    logging.info('parameters successfully loaded'); print 'parameters successfully loaded' 	
    version = '_cond_ae_2'
    name = 'lsgan' + str(version)

#%%
logging.info('param setup')
gen_output = ll.get_output(generator)
gen_params = ll.get_all_params(generator, trainable=True)
eta = theano.shared(lasagne.utils.floatX(initial_eta))


# Create expression for passing real data through the critic
real_out = ll.get_output(critic, inputs=GAN.input_c)
critic_input = combine_tensor_images(GAN.input_, gen_output, batch_size)
fake_out = ll.get_output(critic, inputs=gen_output)


# hidden layers for "feature matching" on hidden layer
hid_real = ll.get_output(GAN.critic[4], inputs=GAN.input_c, deterministic=False)
hid_dim = ll.get_output_shape(GAN.critic[4])
hid_fake = ll.get_output(GAN.critic[4], inputs=gen_output, deterministic=False)

hid_real_center = hid_real#extract_middle_tensor(hid_real, hid_dim)
hid_fake_center = hid_fake#extract_middle_tensor(hid_fake, hid_dim)

loss_gen_fm = lasagne.objectives.squared_error(hid_real_center, hid_fake_center).mean()

#loss_gen_fm = lasagne.objectives.squared_error(gen_output, GAN.input_c).mean() # classic MSE


a, b, c = 0, 1, 1
index = T.lscalar() 

# Create update expressions for training
critic_params = ll.get_all_params(critic, trainable=True)
critic_loss = (lasagne.objectives.squared_error(real_out, b).mean() +
		 lasagne.objectives.squared_error(fake_out, a).mean()) 
#critic_loss = (lasagne.objectives.binary_crossentropy(real_out, 0.9)
#+ lasagne.objectives.binary_crossentropy(fake_out, 0)).mean() 
'''
critic_loss_small = (lasagne.objectives.squared_error(real_out_small, b).mean() +
		 lasagne.objectives.squared_error(fake_out_small, a).mean()) 

critic_params_small =ll.get_all_params(small_critic, trainable=True)
'''
gen_loss = lasagne.objectives.squared_error(fake_out, c).mean()# + 10 * loss_gen_fm
'''
# adding extra penalty for MSE contour 
pred_contour = extract_contour_tensor(gen_output)
real_contour = extract_contour_tensor(GAN.input_c)

loss_gen_contour = lasagne.objectives.squared_error(pred_contour, real_contour).mean()
gen_loss += 20 * loss_gen_contour
'''
print 'loss and function setup'
logging.info('loss and function setup')


#%%
gen_grads = theano.grad(gen_loss, wrt=gen_params)
critic_grads = theano.grad(critic_loss, wrt=critic_params)
gen_grads_norm = sum(T.sum(T.square(grad)) for grad in gen_grads) / len(gen_grads)
critic_grads_norm = sum(T.sum(T.square(grad)) for grad in critic_grads) / len(critic_grads)

gen_updates= lasagne.updates.rmsprop(
    gen_grads, gen_params, learning_rate=initial_eta)

critic_updates = lasagne.updates.adam(
    critic_grads, critic_params, learning_rate=initial_eta)

#critic_updates_small = lasagne.updates.adam(
#    critic_loss_small, critic_params_small, learning_rate=initial_eta)

train_critic = theano.function(inputs=[GAN.input_, GAN.input_c], 
                               outputs=[(real_out ).mean(),
                                        (fake_out ).mean(),
                                        critic_loss,
                                        critic_grads_norm], 
                               updates=critic_updates,
                               name='train_critic', on_unused_input='warn')
'''

train_critic_small = theano.function(inputs=[GAN.input_, GAN.input_c], 
                               outputs=[(real_out_small  ).mean(),
                                        (fake_out_small  ).mean(),
                                        critic_loss_small],
                               updates=critic_updates_small,
                               name='train_critic_small', on_unused_input='warn')

'''
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
dataset, dataset_size = load_dataset(sample=False, normalize=False)
dataset /= 255.
num_batches = dataset_size / batch_size - 2
batches = iterate_minibatches(dataset[:num_batches * batch_size], batch_size, shuffle=True, forever=True)

logging.info('data loaded')
last_index = num_batches + 1

for epoch in range(300000) :
    print epoch
    updates_gen = 0
    updates_critic = 0
    updates_critic_small = 0
    disc_err = 0
    disc_err_small = 0
    gen_err = 0
    gen_iter = 1
    border_epoch = 0   
    critic_iter = 2 if epoch < 2 else 1    
    index_int = 0
    step = 0.08
    pit_stop = 150

    for _ in range(50):
	#GAN.mixing_coefs[0].set_value(np.float32(np.random.rand(1) * 0.5 / 0.90 )[0])
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
    if epoch % pit_stop == -1: 
        #import pdb; pdb.set_trace()
        GAN.update_mixing_coefs(step=0.08)
    
    # test it out 
    for i in range(1):
        test_batch = target
        test_batch = dataset[num_batches *batch_size : (num_batches + 1) * batch_size]
	samples, raw_samples = test_gen(test_batch)
        samples = samples * 255.#67.75 + 113.91
        raw_samples = raw_samples *255.# 67.75 + 113.91
        raw_result = raw_samples.transpose(0,2,3,1).astype('uint8')
        result = samples.transpose(0,2,3,1).astype('uint8')
	original = (test_batch * 67.75 + 113.91).transpose(0,2,3,1).astype('uint8') 
	#blended = batch_poisson_blending(original, raw_samples)
        #saveImage(original, 'original', epoch, side=8)
	#saveImage(blended, 'blended', epoch, side=8)
        saveImage(result, 'samples_' +  name, epoch, side=8) 
    	saveImage(raw_result, 'samples_raw_' + name, epoch, side=8)

	#np.savez(home + 'images/original' + str(epoch) + '.npz', original)
	#np.savez(home + 'images/prediction' + str(epoch) + '.npz', result)
	#np.savez(home + 'images/raw' + str(epoch) + '.npz', raw_result)
    if epoch % 25 == 0 :
        np.savez(home + 'models/' + str(name) + '_disc_' + str(epoch) + '.npz', *ll.get_all_param_values(critic))
        np.savez(home + 'models/' + str(name) + '_gen_' + str(epoch) + '.npz', *ll.get_all_param_values(generator))
    
    # Then we print the results for this epoch:

    logging.info("epoch : " + str(epoch))
    logging.info("  generator loss:\t\t{}".format(gen_err / updates_gen))
    print("  generator loss:\t\t{}".format(gen_err / updates_gen))
    
    if epoch >  border_epoch : 
        print("  discriminator loss:\t\t{}".format(disc_err / updates_critic))
    	#print("  discriminator small loss:\t\t{}".format(disc_err_small / updates_critic_small))
    	#logging.info("  discriminator loss:\t\t{}".format(disc_err / updates_critic))


