# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 15:35:59 2017
@author: pcluc
"""
from helper import *
import numpy as np
import numpy.random as rng
from PIL import Image
import theano as th
import theano.tensor as T
import lasagne
import lasagne.layers as ll
from lasagne.init import Normal
#from layers import *
from theano.sandbox.rng_mrg import MRG_RandomStreams
import nn

global mask
mask = None

'''
class representing an actual model (lasagne/theano)
'''
class GAN :
    def __init__(self, name='GAN', version=1, batch_size=64, encode=False):
        self.name = name
        self.version = version
        self.batch_size = batch_size
        
        self.input_ = T.tensor4('generator input')
        self.output_ = T.tensor4('generator output')       
        self.input_c = T.tensor4('critic/disc input')
        
        self.generator = self.build_generator(encode=encode)
        self.critic = self.build_critic()
	#self.small_critic = self.build_small_critic()
    '''
    save parameters of lasagne model
    TODO : update this to work with new format
    '''
    def save(self, epoch=None):
        
        if epoch != None : 
            ext = 'epoch_' + str(epoch) + '.npz'
        else : 
            ext = '.npz'
        np.savez('models/' + self.name + ext, *lasagne.layers.get_all_param_values(self.model['output']))
        
        if self.critic != None : 
            np.savez('models/' + self.name + '_critic_' + ext, *lasagne.layers.get_all_param_values(self.critic['output']))
        
    '''
    load parameters of previously trained lasagne model
    TODO : update this to work with new format
    '''
    def load(self, epoch=''):
        assert self.generator != None
   
        with np.load('models/' + self.name + '.npz') as file_desc:
            param_values = [file_desc['arr_%d' % i] for i in range(len(file_desc.files))]
            lasagne.layers.set_all_param_values(self.model['output'], param_values)
        
        if self.critic != None:     
            with np.load('models/' + self.name + '_critic_' + '.npz') as file_desc:
                param_values = [file_desc['arr_%d' % i] for i in range(len(file_desc.files))]
                lasagne.layers.set_all_param_values(self.critic['output'], param_values)
        
    
    '''
    contruct generator. Architeture based on DCGAN (if you neglect the encoder part). 
    '''
    # generator takes as input uncropped image and crops it himself
    def build_generator(self, version=1, encode=False):

	#from lasagne.layers import TransposedConv2DLayer as Deconv2DLayer
        global mask
        
	if mask is None : 
            mask = T.zeros(shape=(self.batch_size, 1, 64, 64), dtype=theano.config.floatX)
            mask = T.set_subtensor(mask[:, :, 16:48, 16:48], 1.)
	    self.mask = mask
    
       	
        noise_dim = (self.batch_size, 100)
        theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))
        noise = theano_rng.uniform(size=noise_dim)        
        # mask_color = T.cast(T.cast(theano_rng.uniform(size=(self.batch_size,), low=0., high=2.), 'int16').dimshuffle(0, 'x', 'x', 'x') * mask, dtype=theano.config.floatX)
	input = ll.InputLayer(shape=noise_dim, input_var=noise)

        cropped_image = T.cast(T.zeros_like(self.input_) * mask + (1. - mask) * self.input_ , dtype=theano.config.floatX)
	encoder_input = T.concatenate([cropped_image, mask], axis=1) # shoudl concat wrt channels

        if version == 1:  
            if encode : 
                gen_layers = [ll.InputLayer(shape=(self.batch_size, 4, 64, 64), input_var=encoder_input)]                   #  3 x 64 x 64 -->  64 x 32 x 32

                gen_layers.append(nn.batch_norm(ll.Conv2DLayer(gen_layers[-1], 64, 4, 2, pad=1, nonlinearity=nn.lrelu)))    #  64 x 32 x 32 --> 128 x 16 x 16
                
                gen_layers.append(nn.batch_norm(ll.Conv2DLayer(gen_layers[-1], 128, 4, 2, pad=1, nonlinearity=nn.lrelu)))   # 128 x 16 x 16 -->  256 x 8 x 8 
                
                gen_layers.append(nn.batch_norm(ll.Conv2DLayer(gen_layers[-1], 256, 4, 2, pad=1, nonlinearity=nn.lrelu)))   # 256 x 8 x 8 --> 512 x 4 x 4 
                
                gen_layers.append(nn.batch_norm(ll.Conv2DLayer(gen_layers[-1], 512, 4, 2, pad=1, nonlinearity=nn.lrelu)))   # 512 x 4 x 4 --> 1024 x 2 x 2
                
                gen_layers.append(nn.batch_norm(ll.Conv2DLayer(gen_layers[-1], 1024, 4, 2, pad=1, nonlinearity=nn.lrelu)))  # 1024 x 2 x 2 --> 2048 x 1 x 1       
            
                gen_layers.append(nn.batch_norm(ll.Conv2DLayer(gen_layers[-1], 2048, 4, 2, pad=1, nonlinearity=nn.lrelu)))          
                # flatten this out
                gen_layers.append(ll.FlattenLayer(gen_layers[-1]))
                # concat with noise
                #gen_layers.append(ll.ConcatLayer([input, gen_layers[-1]]))
                latent_size = 2048 # + 100

            else : 
                gen_layers = [input]
                latent_size = 100	
		
	    # TODO : put batchorm back on all layers, + g=None
	    gen_layers.append(ll.DenseLayer(gen_layers[-1], 128*8*4*4, W=Normal(0.02)))
            gen_layers.append(ll.ReshapeLayer(gen_layers[-1], (self.batch_size,128*8, 4, 4)))

	    # creating array of mixing coefficients (shared Theano floats) that will be used for mixing generated_output and image at each layer
	    mixing_coefs = [ theano.shared(lasagne.utils.floatX(0.5))  for i in range(3)]
	    mixing_coefs.append(theano.shared(lasagne.utils.floatX(1.)))
            border = 2
            gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (self.batch_size, 128*4, 8, 8), (5, 5), W=Normal(0.02),nonlinearity=nn.relu), g=None))  # 4 -> 8
            

            gen_layers.append(nn.ResetDeconvLayer(gen_layers[-1], cropped_image, mixing_coefs[0], border=border))
	    #layer_a = nn.ResetDeconvLayer(gen_layers[-1], cropped_image, mixing_coefs[0]) # all new
	    #layer_concat_a = ll.ConcatLayer([layer_a, gen_layers[-1]], axis=1)
	    #gen_layers.append(layer_concat_a)        

            gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (self.batch_size, 128*2, 16, 16), (5, 5), W=Normal(0.02),nonlinearity=nn.relu), g=None))  # 8 -> 16
	    
	    gen_layers.append(nn.ResetDeconvLayer(gen_layers[-1], cropped_image, mixing_coefs[1], border=border*2))
	    #layer_b = nn.ResetDeconvLayer(gen_layers[-1], cropped_image, mixing_coefs[1]) # all new
	    #layer_concat_b = ll.ConcatLayer([layer_b, gen_layers[-1]], axis=1)
	    #gen_layers.append(layer_concat_b)        

            gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (self.batch_size, 128, 32, 32), (5, 5), W=Normal(0.02),nonlinearity=nn.relu), g=None))  # 16 -> 32            

	    gen_layers.append(nn.ResetDeconvLayer(gen_layers[-1], cropped_image, mixing_coefs[2], border=border*2*2))
	    #layer_c = nn.ResetDeconvLayer(gen_layers[-1], cropped_image, mixing_coefs[1]) # all new
	    #layer_concat_c = ll.ConcatLayer([layer_c, gen_layers[-1]], axis=1)
	    #gen_layers.append(layer_concat_c)        
            
            gen_layers.append(nn.Deconv2DLayer(gen_layers[-1], (self.batch_size, 3, 64, 64), (5, 5), W=Normal(0.02),nonlinearity=lasagne.nonlinearities.sigmoid))  # 32 -> 64
	    
	    gen_layers.append(nn.ResetDeconvLayer(gen_layers[-1], cropped_image, mixing_coefs[3], border=border*2*2*2))

        for layer in gen_layers : 
            print layer.output_shape
        print ''

	GAN.mixing_coefs = mixing_coefs
                
        return gen_layers
                        



    '''
    construct critic/discriminator. Architecture based on DCGAN. However, Least Squares loss is used. 
    '''
    def build_critic(self, version=1, extra_middle=True, only_middle=False):
        assert self.generator != None
        
	if version==1:
	    from lasagne.nonlinearities import sigmoid
            disc_layers = [ll.InputLayer(shape=(self.batch_size, 3, 64, 64), input_var=self.input_c)]
            # b_s x 3 x 64 x 64 --> b_s x 32 x 32 x 32
            disc_layers.append(nn.batch_norm(ll.Conv2DLayer(disc_layers[-1], 128, (3,3), pad=1, stride=2,  W=Normal(0.03), nonlinearity=nn.lrelu)))#nn.weight_norm
            
            # disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.5))
            # b_s x 32 x 32 x 32 --> b_s x 64 x 16 x 16 
            disc_layers.append(nn.batch_norm(ll.Conv2DLayer(disc_layers[-1], 256, (3,3), pad=1, stride=2,  W=Normal(0.03), nonlinearity=nn.lrelu)))#nn.weight_norm
            # b_s x 64 x 16 x 16 --> b_s x 128 x 8 x 8
	    # note : this layer will be used to compare statistics (output of disc_layer[3])
            disc_layers.append(nn.batch_norm(ll.Conv2DLayer(disc_layers[-1], 512, (3,3), pad=1, stride=2,  W=Normal(0.03),  nonlinearity=nn.lrelu)))#nn.weight_norm
            # b_s x 128 x 8 x 8 --> b_s x 256 x 4 x 4
            disc_layers.append(nn.batch_norm(ll.Conv2DLayer(disc_layers[-1], 1024, (3,3), pad=1, stride=2,  W=Normal(0.03), nonlinearity=nn.lrelu)))#nn.weight_norm
            
            disc_layers.append(ll.GlobalPoolLayer(disc_layers[-1]))
            disc_layers.append(nn.MinibatchLayer(disc_layers[-1], num_kernels=100))

	    if extra_middle or only_middle:  
		# goal : add mechanism that checks for mode collapse, but only for the middle part
           
	        layer = nn.ExtractMiddleLayer(disc_layers[0], extra=2)
		# two convolutions
	        print layer.output_shape
	        layer = nn.batch_norm(ll.Conv2DLayer(layer, 128, 5, stride=2, pad='same',
					   nonlinearity=nn.lrelu))

	        print layer.output_shape
	        layer = nn.batch_norm(ll.Conv2DLayer(layer, 256, 5, stride=2, pad='same',
					   nonlinearity=nn.lrelu))

	        print layer.output_shape
	        layer = nn.batch_norm(ll.Conv2DLayer(layer, 512, 5, stride=2, pad='same',
					   nonlinearity=nn.lrelu))

	        #layer = nn.batch_norm(ll.Conv2DLayer(layer, 1024, 5, stride=2, pad='same',
		#			   nonlinearity=nn.lrelu))
		
		#disc_layers.append(layer)

	        #print layer.output_shape
	        layer = ll.GlobalPoolLayer(layer)	    
	    
	        #print layer.output_shape
	        layer = nn.MinibatchLayer(layer, num_kernels=400)
		
		if only_middle : 
		     disc_layers = []
                     disc_layers.append((ll.DenseLayer(layer, num_units=1, W=Normal(0.03), nonlinearity=None)))
		     return disc_layers

		disc_layers.append(ll.ConcatLayer([layer, disc_layers[-1]]))

            disc_layers.append((ll.DenseLayer(disc_layers[-1], num_units=1, W=Normal(0.03), nonlinearity=None)))

	    # seeing how there is strong intra-batch normalization, I think it would be good to do minibatch discrimination solely on the center of the images. 
	    # I think that the statistics of the border are throwing 
            
            for layer in disc_layers : 
                print layer.output_shape
            print ''
            
            return disc_layers


    def update_mixing_coefs(self, step=0.0005):
        '''index = 0 
        while index < len(self.mixing_coefs) and self.mixing_coefs[index].get_value() >= .99 :
            index += 1

        if index < len(self.mixing_coefs):
            to_be_updated = self.mixing_coefs[index]
            to_be_updated.set_value(to_be_updated.get_value() + np.float32(step))
        print [x.get_value() for x in self.mixing_coefs]
	'''
	for i in range(len(self.mixing_coefs)):
	    current_value = self.mixing_coefs[i].get_value()
	    self.mixing_coefs[i].set_value(min(current_value + np.float32(step), np.float32(1.)))
            
        print [x.get_value() for x in self.mixing_coefs]
