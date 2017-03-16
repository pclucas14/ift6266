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
from layers import *
from theano.sandbox.rng_mrg import MRG_RandomStreams
import nn

'''
class representing an actual model (lasagne/theano)
'''
class Model :
    def __init__(self, name='lasagne_model', version=1, batch_size=64, full_img=False):
        self.name = name
        self.version = version
        self.batch_size = batch_size
        
        self.input_ = T.tensor4('auto encoder input')
        self.output_ = T.tensor4('auto encoder output')       
        self.input_c = T.tensor4('critic input')
        
        self.model = self.build_model(version=version)
        self.critic = self.build_critic(full_img)
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
            np.savez('models/' + self.name + '_critic_' + ext, *lasagne.layers.get_all_param_values(self.critic['output']))
        
    '''
    load parameters of previously trained lasagne model
    '''
    def load(self, epoch=''):
        assert self.model != None
   
        with np.load('models/' + self.name + '.npz') as file_desc:
            param_values = [file_desc['arr_%d' % i] for i in range(len(file_desc.files))]
            lasagne.layers.set_all_param_values(self.model['output'], param_values)
        
        if self.critic != None:     
            with np.load('models/' + self.name + '_critic_' + '.npz') as file_desc:
                param_values = [file_desc['arr_%d' % i] for i in range(len(file_desc.files))]
                lasagne.layers.set_all_param_values(self.critic['output'], param_values)
        
    '''
    general method to build current/future models. 
    '''
    def build_model(self,version):
    
        from lasagne.layers import TransposedConv2DLayer as Deconv2DLayer
        # first model is a regular auto encodeder with Fully Connected layer in 
        # the middle. See blog for more detailed architecture comments. 
        
        # lets try and fill that hole with some random noise
        self.noise_var = T.tensor4('noise')
        self.model_input = self.input_#combine_tensor_images(self.input_, self.noise_var, self.batch_size)
        data_size=(self.batch_size,3,64,64) # Batch size x Img Channels x Height x Width
        
        if version == 2: 
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
            gen_layers.append(nn.batch_norm(
                ll.DenseLayer(gen_layers[-1], num_units=5 * 5 * 512, W=Normal(0.05), nonlinearity=nn.relu), g=None))
            gen_layers.append(ll.ReshapeLayer(gen_layers[-1], (self.batch_size, 512, 5, 5)))
            gen_layers.append(nn.batch_norm(
                nn.Deconv2DLayer(gen_layers[-1], (self.batch_size, 256, 10, 10), (5, 5), W=Normal(0.05),
                                 nonlinearity=nn.relu), g=None))  # 4 -> 8
            gen_layers.append(nn.batch_norm(
                nn.Deconv2DLayer(gen_layers[-1], (self.batch_size, 128, 20, 20), (5, 5), W=Normal(0.05),
                                 nonlinearity=nn.relu), g=None))  # 8 -> 16
            gen_layers.append(nn.weight_norm(
                nn.Deconv2DLayer(gen_layers[-1], (self.batch_size, 3, 40, 40), (5, 5), W=Normal(0.05),
                                 nonlinearity=T.tanh), train_g=True, init_stdv=0.1))  # 16 -> 32
            '''
            # decoder
            #gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (self.batch_size,1024,2,2), (5,5), W=Normal(0.05), nonlinearity=nn.relu), g=None)) # 2 -> 4
            gen_layers.append(nn.batch_norm(ll.Deconv2DLayer(gen_layers[-1], (self.batch_size,512,5,5), (5,5), W=Normal(0.05), nonlinearity=nn.relu), g=None)) # 4 -> 8        
            gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (self.batch_size,256,4,4), (5,5), W=Normal(0.05), nonlinearity=nn.relu), g=None)) # 4 -> 8
            gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (self.batch_size,128,8,8), (5,5), W=Normal(0.05), nonlinearity=nn.relu), g=None)) # 8 -> 16
            gen_layers.append(nn.weight_norm(nn.Deconv2DLayer(gen_layers[-1], (self.batch_size,3,40,40), (5,5), W=Normal(0.05), nonlinearity=T.tanh), train_g=True, init_stdv=0.1)) # 16 -> 32
            '''
            for layer in gen_layers : 
                print layer.output_shape
            print ''
                
            return gen_layers
                        
        
        elif version == 1: 
            net = {}
            

            # encoder
            net['input'] = lasagne.layers.InputLayer(data_size, 
                                            input_var=self.model_input)
            
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
            
            net['dense'] = channel_wise_fc_layer(net['bn3'])
            net['prop'] =  lasagne.layers.Conv2DLayer(net['dense'], 
                                             num_filters=64, 
                                             filter_size=3,
                                             stride=(1,1),
                                             pad='same')
            
            # decoder 
            net['c3_t'] = lasagne.layers.TransposedConv2DLayer(net['prop'],
                                                       num_filters=32,
                                                       filter_size=4,
                                                       stride=(2,2),
                                                       crop=1)
            net['bn3_t'] = lasagne.layers.BatchNormLayer(net['c3_t'])
            
            net['c2_t'] = lasagne.layers.TransposedConv2DLayer(net['bn3_t'],
                                                       num_filters=3,
                                                       filter_size=4,
                                                       stride=(2,2),
                                                       crop=1)
            net['bn2_t'] = lasagne.layers.BatchNormLayer(net['c2_t'])


            return net['output']





    '''
    construct critic to get adversarial loss gradient
                                                       stride=(1,1),
                                                       crop=1)
    '''
    def build_critic(self, full_img, version=4):
        #assert self.model != None
        
        if version==4:
            from lasagne.nonlinearities import sigmoid
            disc_layers = [ll.InputLayer(shape=(None, 3, 40, 40), input_var=self.input_c)]
            # b_s x 3 x 40 x 40 --> b_s x 32 x 20 x 20
            disc_layers.append(nn.weight_norm(ll.Conv2DLayer(disc_layers[-1], 32, (3,3), pad=1, stride=2,  W=Normal(0.03), nonlinearity=nn.lrelu)))#nn.weight_norm
            #disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.5))
            # b_s x 32 x 20 x 20 --> b_s x 64 x 10 x 10 
            disc_layers.append(nn.weight_norm(ll.Conv2DLayer(disc_layers[-1], 64, (3,3), pad=1, stride=2,  W=Normal(0.03), nonlinearity=nn.lrelu)))#nn.weight_norm
            #disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.5))
            # b_s x 64 x 10 x 10 --> b_s x 128 x 5 x 5
            disc_layers.append(nn.weight_norm(ll.Conv2DLayer(disc_layers[-1], 128, (3,3), pad=1, stride=2,  W=Normal(0.03),  nonlinearity=nn.lrelu)))#nn.weight_norm
            # b_s x 128 x 5 x 5 --> b_s x 256 x 5 x 5
            disc_layers.append(nn.weight_norm(ll.Conv2DLayer(disc_layers[-1], 256, (3,3), pad=1, stride=2,  W=Normal(0.03), nonlinearity=nn.lrelu)))#nn.weight_norm
            #disc_layers.append(ll.GlobalPoolLayer(disc_layers[-1]))
            disc_layers.append(nn.MinibatchLayer(disc_layers[-1], num_kernels=40))
            disc_layers.append((ll.DenseLayer(disc_layers[-1], num_units=1, W=Normal(0.03), nonlinearity=None)))#nn.weight_norm, train_g=True, init_stdv=0.1))#nn.weight_norm
            
            for layer in disc_layers : 
                print layer.output_shape
            print ''
            
            return disc_layers
            
        # openAI improved techniques for GAN discriminator 
        if version==3:
            from lasagne.nonlinearities import sigmoid
            disc_layers = [ll.InputLayer(shape=(None, 3, 40, 40), input_var=self.input_c)]
            disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.2))
           # disc_layers.append(nn.weight_norm(ll.Conv2DLayer(disc_layers[-1], 96, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
           # disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 96, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
            disc_layers.append(nn.weight_norm(ll.Conv2DLayer(disc_layers[-1], 96, (3,3), pad=1, stride=2, W=Normal(0.05), nonlinearity=nn.lrelu))) #nn.weight_norm
            disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.5))
            disc_layers.append((ll.Conv2DLayer(disc_layers[-1], 192, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu))) #nn.weight_norm
            #disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 192, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
            disc_layers.append(nn.weight_norm(ll.Conv2DLayer(disc_layers[-1], 192, (3,3), pad=1, stride=2, W=Normal(0.05), nonlinearity=nn.lrelu))) #nn.weight_norm
            disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.5))
            #disc_layers.append(nn.weight_norm(ll.Conv2DLayer(disc_layers[-1], 192, (3,3), pad=0, W=Normal(0.05), nonlinearity=nn.lrelu)))
            disc_layers.append(nn.weight_norm(ll.NINLayer(disc_layers[-1], num_units=192, W=Normal(0.05), nonlinearity=nn.lrelu))) #
            #disc_layers.append(nn.weight_norm(ll.NINLayer(disc_layers[-1], num_units=192, W=Normal(0.05), nonlinearity=nn.lrelu)))
            disc_layers.append(ll.GlobalPoolLayer(disc_layers[-1]))
            disc_layers.append(nn.MinibatchLayer(disc_layers[-1], num_kernels=40))
            #disc_layers.append(nn.weight_norm(ll.DenseLayer(disc_layers[-1], num_units=1, W=Normal(0.05), nonlinearity=sigmoid)) train_g=True, init_stdv=0.1) 
            
            for layer in disc_layers : 
                print layer.output_shape
            print ''
            
            return disc_layers
            
                        
        
        '''
        discriminator taken from f0k's DCGAN implementation
        '''
        if version==2:
            net = dict()
            if full_img : 
                data_size = (self.batch_size, 3, 64, 64)
            else : 
                data_size=(self.batch_size,3,32,32) # Batch size x Img Channels x Height x Width
    
            from lasagne.layers import (InputLayer, Conv2DLayer, ReshapeLayer,
                                    DenseLayer)
            try:
                from lasagne.layers.dnn import batch_norm_dnn as batch_norm
            except ImportError:
                from lasagne.layers import batch_norm
            from lasagne.nonlinearities import LeakyRectify, sigmoid
            lrelu = LeakyRectify(0.2)
            # input: (None, 1, 28, 28)
            layer = InputLayer(shape=data_size, input_var=self.input_c)
            # two convolutions
            layer = batch_norm(Conv2DLayer(layer, 64, 5, stride=2, pad='same',
                                           nonlinearity=lrelu))
            print ("critic output:", layer.output_shape)
            layer = batch_norm(Conv2DLayer(layer, 128, 5, stride=2, pad='same',
                                           nonlinearity=lrelu))
            print ("critic output:", layer.output_shape)
            # fully-connected layer
            #layer = MinibatchLayer(layer, 15)
            #print ("critic output:", layer.output_shape)
            layer = batch_norm(DenseLayer(layer, 1024, nonlinearity=lrelu))
            # output layer (linear and without bias)
            print ("critic output:", layer.output_shape)
            layer = DenseLayer(layer, 1, nonlinearity=sigmoid)
            print ("critic output:", layer.output_shape)
            
            net['output'] = layer
            return net
            
        
        # direct implementation of inpainting paper 
        elif version == 1 : 
            net = dict()
            data_size=(self.batch_size,3,32,32) # Batch size x Img Channels x Height x Width
            lrelu = lasagne.nonlinearities.LeakyRectify(0.2)
            
            net['input'] = lasagne.layers.InputLayer(data_size, 
                                                     input_var=self.input_c)
            net['c1'] = lasagne.layers.Conv2DLayer(net['input'], 
                                                 num_filters=16, 
                                                 filter_size=5,
                                                 stride=(2,2),
                                                 nonlinearity=lrelu,
                                                 pad='same')
            net['bn1'] = lasagne.layers.BatchNormLayer(net['c1'])
            
            net['c2'] = lasagne.layers.Conv2DLayer(net['bn1'], 
                                                 num_filters=32, 
                                                 filter_size=3,
                                                 stride=(2,2),
                                                 nonlinearity=lrelu,
                                                 pad='same')
            net['bn2'] = lasagne.layers.BatchNormLayer(net['c2'])
            
            net['fc1'] = lasagne.layers.DenseLayer(net['bn2'],
                                                   1024,
                                                   nonlinearity=lrelu)
            net['bn3'] = lasagne.layers.BatchNormLayer(net['fc1'])
            
            net['output'] = lasagne.layers.DenseLayer(net['bn3'],
                                                      1,
                                                      nonlinearity=lasagne.nonlinearities.sigmoid)
                                                      #b=None)
            return net  
                
 
        
#%%
