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

'''
class representing an actual model (lasagne/theano)
'''
class GAN :
    def __init__(self, name='GAN', version=1, batch_size=64):
        self.name = name
        self.version = version
        self.batch_size = batch_size
        
        self.input_ = T.tensor4('generator input')
        self.output_ = T.tensor4('generator output')       
        self.input_c = T.tensor4('critic/disc input')
        
        self.generator = self.build_generator()
        self.critic = self.build_critic()
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
    def build_generator(self, version=1):
    
        from lasagne.layers import TransposedConv2DLayer as Deconv2DLayer
        
        if version == 1: 
            # encoder
            '''
            gen_layers = [ll.InputLayer(shape=data_size, input_var=self.model_input)]
            # b_s x 3 x 64 x 64 --> b_s x 64 x 32 x 32
            gen_layers.append(nn.batch_norm(ll.Conv2DLayer(gen_layers[-1], 64, 4, 2, pad=1, nonlinearity=nn.lrelu)))
            # b_s x 64 x 32 x 32 --> b_s x 64 x 16 x 16
            gen_layers.append(nn.batch_norm(ll.Conv2DLayer(gen_layers[-1], 64, 4, 2, pad=1, nonlinearity=nn.lrelu)))
            # b_s x 64 x 16 x 16 --> b_s x 128 x 8 x 8
            gen_layers.append(nn.batch_norm(ll.Conv2DLayer(gen_layers[-1], 128, 4, 2, pad=1, nonlinearity=nn.lrelu)))
            # b_s x 128 x 8 x 8 --> b_s x 256 x 4 x 4 
            gen_layers.append(nn.batch_norm(ll.Conv2DLayer(gen_layers[-1], 256, 4, 2, pad=1, nonlinearity=nn.lrelu)))
            # b_s x 128 x 8 x 8 --> b_s x 512 x 2 x 2 
            #gen_layers.append(nn.batch_norm(ll.Conv2DLayer(gen_layers[-1], 2048, 4, 2, pad='same', nonlinearity=nn.lrelu)))
            # b_s x 256 x 4 x 4 --> b_s x 2048 x 1 x 1
            
            gen_layers.append(nn.batch_norm(ll.Conv2DLayer(gen_layers[-1], 2048, 4, 4, pad=1, nonlinearity=nn.lrelu)))          
            '''
            noise_dim = (self.batch_size, 100)
            theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))
            noise = theano_rng.uniform(size=noise_dim)
            gen_layers = [ll.InputLayer(shape=noise_dim, input_var=noise)]
            
	    #gen_layers.append(nn.batch_norm(ll.DenseLayer(gen_layers[-1], num_units=5 * 5 * 512, W=Normal(0.05), nonlinearity=nn.relu), g=None))
            gen_layers.append(ll.ReshapeLayer(gen_layers[-1], (self.batch_size, 100, 1, 1)))
            gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (self.batch_size, 256, 2, 2), (5, 5), W=Normal(0.02),nonlinearity=nn.relu), g=None))  # 1 -> 2
            gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (self.batch_size, 128, 4, 4), (5, 5), W=Normal(0.02),nonlinearity=nn.relu), g=None))  # 2 -> 4
	    gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (self.batch_size, 64, 8, 8), (5, 5), W=Normal(0.02),nonlinearity=nn.relu), g=None))  # 4 -> 8
	    gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (self.batch_size, 64, 16, 16), (5, 5), W=Normal(0.02),nonlinearity=nn.relu), g=None))  # 8 -> 16
	    gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (self.batch_size, 32, 32, 32), (5, 5), W=Normal(0.02),nonlinearity=nn.relu), g=None))  # 8 -> 16
            gen_layers.append((nn.Deconv2DLayer(gen_layers[-1], (self.batch_size, 3, 64, 64), (5, 5), W=Normal(0.02),nonlinearity=T.tanh)))  # 16 -> 32
            
	    for layer in gen_layers : 
                print layer.output_shape
            print ''
                
            return gen_layers
                        

    '''
    construct critic/discriminator. Architecture based on DCGAN. However, Least Squares loss is used. 
    '''
    def build_critic(self, version=1):
        assert self.generator != None
        
	if version==1:
	    from lasagne.nonlinearities import sigmoid
            disc_layers = [ll.InputLayer(shape=(None, 3, 64, 64), input_var=self.input_c)]
            # b_s x 3 x 40 x 40 --> b_s x 32 x 20 x 20
            disc_layers.append(nn.batch_norm(ll.Conv2DLayer(disc_layers[-1], 32, (3,3), pad=1, stride=2,  W=Normal(0.03), nonlinearity=nn.lrelu)))#nn.weight_norm
            #disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.5))
            # b_s x 32 x 20 x 20 --> b_s x 64 x 10 x 10 
            disc_layers.append(nn.batch_norm(ll.Conv2DLayer(disc_layers[-1], 64, (3,3), pad=1, stride=2,  W=Normal(0.03), nonlinearity=nn.lrelu)))#nn.weight_norm
            #disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.5))
            # b_s x 64 x 10 x 10 --> b_s x 128 x 5 x 5
            disc_layers.append(nn.batch_norm(ll.Conv2DLayer(disc_layers[-1], 128, (3,3), pad=1, stride=2,  W=Normal(0.03),  nonlinearity=nn.lrelu)))#nn.weight_norm
            # b_s x 128 x 5 x 5 --> b_s x 256 x 5 x 5
            disc_layers.append(nn.batch_norm(ll.Conv2DLayer(disc_layers[-1], 256, (3,3), pad=1, stride=2,  W=Normal(0.03), nonlinearity=nn.lrelu)))#nn.weight_norm
            
            disc_layers.append(nn.batch_norm(ll.Conv2DLayer(disc_layers[-1], 512, (3,3), pad=1, stride=2,  W=Normal(0.03), nonlinearity=nn.lrelu)))#nn.weight_norm
	    disc_layers.append(ll.GlobalPoolLayer(disc_layers[-1]))
            disc_layers.append(nn.MinibatchLayer(disc_layers[-1], num_kernels=100))
            disc_layers.append((ll.DenseLayer(disc_layers[-1], num_units=1, W=Normal(0.03), nonlinearity=None)))#nn.weight_norm, train_g=True, init_stdv=0.1))#nn.weight_norm
            
            for layer in disc_layers : 
                print layer.output_shape
            print ''
            
            return disc_layers
            
#%%
