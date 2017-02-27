# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 15:35:59 2017

@author: pcluc
"""
from helper import *
import numpy as np
import numpy.random as rng
from PIL import Image
import theano
import theano.tensor as T
from keras.datasets import mnist
import lasagne

'''
class representing an actual model (lasagne/theano)
'''
class Model : 
    def __init__(self, name='lasagne_model', version=1, batch_size=64):
        self.name = name
        self.version = version
        self.batch_size = batch_size
        
        self.input_ = T.tensor4('auto encoder input')
        self.output_ = T.tensor4('auto encoder output')       
        self.input_c = T.tensor4('critic input')
        
        self.model = self.build_model(version=version)
        self.critic = self.build_critic()
    '''
    save parameters of lasagne model
    '''
    def save(self, epoch=None):
        
        if epoch != None : 
            ext = 'epoch_' + str(epoch) + '.npz'
        else : 
            ext = '.npz'
        np.savez('models/' + self.name + ext, *lasagne.layers.get_all_param_values(self.model['output']))
        
        if self.critic != None : 
            np.savez('models/' + self.name + '_critic_' + ext, *lasagne.layers.get_all_param_values(self.model['output']))
        
    '''
    load parameters of previously trained lasagne model
    '''
    def load(self):
        assert self.model != None
           
        with np.load('models/' + self.name + '.npz') as file_desc:
            param_values = [file_desc['arr_%d' % i] for i in range(len(file_desc.files))]
            lasagne.layers.set_all_param_values(self.model['output'], param_values)
            
        if self.critic != None:
            
            with np.load('models/' + self.name + '_critic.npz') as file_desc:
                param_values = [file_desc['arr_%d' % i] for i in range(len(file_desc.files))]
                lasagne.layers.set_all_param_values(self.critic['output'], param_values)

    '''
    general method to build current/future models. 
    '''
    def build_model(self,version):
    
        # first model is a regular auto encodeder with Fully Connected layer in 
        # the middle. See blog for more detailed architecture comments. 

        if version == 1: 
            net = {}
            data_size=(None,3,64,64) # Batch size x Img Channels x Height x Width

            # encoder
            net['input'] = lasagne.layers.InputLayer(data_size, 
                                            input_var=self.input_)
            
            net['c1'] = lasagne.layers.Conv2DLayer(net['input'], 
                                             num_filters=16, 
                                             filter_size=5,
                                             stride=(2,2),
                                             pad='same')
            net['bn1'] = lasagne.layers.BatchNormLayer(net['c1'])
            
            net['c2'] = lasagne.layers.Conv2DLayer(net['bn1'], 
                                             num_filters=32, 
                                             filter_size=3,
                                             stride=(2,2),
                                             pad='same')
            net['bn2'] = lasagne.layers.BatchNormLayer(net['c2'])
            
            net['c3'] = lasagne.layers.Conv2DLayer(net['bn2'], 
                                             num_filters=64, 
                                             filter_size=3,
                                             stride=(2,2),
                                             pad='same')
            net['bn3'] = lasagne.layers.BatchNormLayer(net['c3'])
            
            net['dense'] = lasagne.layers.DenseLayer(net['bn3'], num_units=2048)
            # TODO 
            net['resh'] = lasagne.layers.ReshapeLayer(net['dense'], 
                                                      shape=(self.batch_size, 
                                                             32, 8, 8))
            
            # decoder 
            net['c3_t'] = lasagne.layers.TransposedConv2DLayer(net['resh'],
                                                       num_filters=32,
                                                       filter_size=4,
                                                       stride=(2,2),
                                                       crop=1)
            net['bn3_t'] = lasagne.layers.BatchNormLayer(net['c3_t'])
            
            net['c2_t'] = lasagne.layers.TransposedConv2DLayer(net['bn3_t'],
                                                       num_filters=16,
                                                       filter_size=4,
                                                       stride=(2,2),
                                                       crop=1)
            net['bn2_t'] = lasagne.layers.BatchNormLayer(net['c2_t'])
            
            net['output'] = lasagne.layers.TransposedConv2DLayer(net['bn2_t'],
                                                       num_filters=3,
                                                       filter_size=4,
                                                       stride=(2,2),
                                                       crop=1)
            '''
            net['output'] = lasagne.layers.TransposedConv2DLayer(net['bn1_t'],
                                                       num_filters=3,
                                                       filter_size=3,
                                                       stride=(1,1),
                                                       crop=1)
            '''
        # quick safety check 
        for name, layer in net.iteritems():
            print name, lasagne.layers.get_output_shape(layer)
        
        print "\n"
        return net
    
    
    '''
    construct critic to get adversarial loss gradient
    '''
    def build_critic(self):
        assert self.model != None
        
        net = dict()
        data_size=(self.batch_size,3,64,64) # Batch size x Img Channels x Height x Width
        lrelu = lasagne.nonlinearities.LeakyRectify(0.2)
        
        net['input'] = lasagne.layers.InputLayer(data_size, 
                                                 input_var=self.input_c)
        net['c1'] = lasagne.layers.Conv2DLayer(net['input'], 
                                             num_filters=16, 
                                             filter_size=5,
                                             stride=(2,2),
                                             pad='same')
        net['bn1'] = lasagne.layers.BatchNormLayer(net['c1'])
        
        net['c2'] = lasagne.layers.Conv2DLayer(net['bn1'], 
                                             num_filters=128, 
                                             filter_size=3,
                                             stride=(2,2),
                                             pad='same')
        net['bn2'] = lasagne.layers.BatchNormLayer(net['c1'])
        
        net['fc1'] = lasagne.layers.DenseLayer(net['bn1'],
                                               1024,
                                               nonlinearity=lrelu)
        net['bn3'] = lasagne.layers.BatchNormLayer(net['fc1'])
        
        net['output'] = lasagne.layers.DenseLayer(net['bn3'],
                                                  1,
                                                  nonlinearity=None,
                                                  b=None)
        
        for name, layer in net.iteritems():
            print name, lasagne.layers.get_output_shape(layer)
            
        return net
        
        
