"""
neural network stuff, intended to be used with Lasagne
"""
import theano
import numpy as np
import theano as th
import theano.tensor as T
import lasagne
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.signal import pool

def relu(x):
    return T.maximum(x, 0)
    
def lrelu(x, a=0.2):
    return T.maximum(x, a*x)

def centered_softplus(x):
    return T.nnet.softplus(x) - np.cast[th.config.floatX](np.log(2.))

def log_sum_exp(x, axis=1):
    m = T.max(x, axis=axis)
    return m+T.log(T.sum(T.exp(x-m.dimshuffle(0,'x')), axis=axis))

def adam_updates(params, cost, lr=0.001, mom1=0.9, mom2=0.999):
    updates = []
    grads = T.grad(cost, params)
    t = th.shared(np.cast[th.config.floatX](1.))
    for p, g in zip(params, grads):
        v = th.shared(np.cast[th.config.floatX](p.get_value() * 0.))
        mg = th.shared(np.cast[th.config.floatX](p.get_value() * 0.))
        v_t = mom1*v + (1. - mom1)*g
        mg_t = mom2*mg + (1. - mom2)*T.square(g)
        v_hat = v_t / (1. - mom1 ** t)
        mg_hat = mg_t / (1. - mom2 ** t)
        g_t = v_hat / T.sqrt(mg_hat + 1e-8)
        p_t = p - lr * g_t
        updates.append((v, v_t))
        updates.append((mg, mg_t))
        updates.append((p, p_t))
    updates.append((t, t+1))
    return updates

class WeightNormLayer(lasagne.layers.Layer):
    def __init__(self, incoming, b=lasagne.init.Constant(0.), g=lasagne.init.Constant(1.),
                 W=lasagne.init.Normal(0.05), train_g=False, init_stdv=1., nonlinearity=relu, **kwargs):
        super(WeightNormLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = nonlinearity
        self.init_stdv = init_stdv
        k = self.input_shape[1]
        if b is not None:
            self.b = self.add_param(b, (k,), name="b", regularizable=False)
        if g is not None:
            self.g = self.add_param(g, (k,), name="g", regularizable=False, trainable=train_g)
        if len(self.input_shape)==4:
            self.axes_to_sum = (0,2,3)
            self.dimshuffle_args = ['x',0,'x','x']
        else:
            self.axes_to_sum = 0
            self.dimshuffle_args = ['x',0]

        # scale weights in layer below
        incoming.W_param = incoming.W
        #incoming.W_param.set_value(W.sample(incoming.W_param.get_value().shape))
        if incoming.W_param.ndim==4:
            if isinstance(incoming, Deconv2DLayer):
                W_axes_to_sum = (0,2,3)
                W_dimshuffle_args = ['x',0,'x','x']
            else:
                W_axes_to_sum = (1,2,3)
                W_dimshuffle_args = [0,'x','x','x']
        else:
            W_axes_to_sum = 0
            W_dimshuffle_args = ['x',0]
        if g is not None:
            incoming.W = incoming.W_param * (self.g/T.sqrt(1e-6 + T.sum(T.square(incoming.W_param),axis=W_axes_to_sum))).dimshuffle(*W_dimshuffle_args)
        else:
            incoming.W = incoming.W_param / T.sqrt(1e-6 + T.sum(T.square(incoming.W_param),axis=W_axes_to_sum,keepdims=True))

    def get_output_for(self, input, init=False, **kwargs):
        if init:
            m = T.mean(input, self.axes_to_sum)
            input -= m.dimshuffle(*self.dimshuffle_args)
            inv_stdv = self.init_stdv/T.sqrt(T.mean(T.square(input), self.axes_to_sum))
            input *= inv_stdv.dimshuffle(*self.dimshuffle_args)
            self.init_updates = [(self.b, -m*inv_stdv), (self.g, self.g*inv_stdv)]
        elif hasattr(self,'b'):
            input += self.b.dimshuffle(*self.dimshuffle_args)
            
        return self.nonlinearity(input)

def weight_norm(layer, **kwargs):
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = lasagne.nonlinearities.identity
    if hasattr(layer, 'b'):
        del layer.params[layer.b]
        layer.b = None
    return WeightNormLayer(layer, nonlinearity=nonlinearity, **kwargs)

class Deconv2DLayer(lasagne.layers.Layer):
    def __init__(self, incoming, target_shape, filter_size, stride=(2, 2),
                 W=lasagne.init.Normal(0.05), b=lasagne.init.Constant(0.), nonlinearity=relu, **kwargs):
        super(Deconv2DLayer, self).__init__(incoming, **kwargs)
        self.target_shape = target_shape
        self.nonlinearity = (lasagne.nonlinearities.identity if nonlinearity is None else nonlinearity)
        self.filter_size = lasagne.utils.as_tuple(filter_size, 2, int)
        self.stride = lasagne.utils.as_tuple(stride, 2, int)
        self.target_shape = target_shape

        self.W_shape = (incoming.output_shape[1], target_shape[1], filter_size[0], filter_size[1])
        self.W = self.add_param(W, self.W_shape, name="W")
        if b is not None:
            self.b = self.add_param(b, (target_shape[1],), name="b")
        else:
            self.b = None

    def get_output_for(self, input, **kwargs):
        op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(imshp=self.target_shape, kshp=self.W_shape, subsample=self.stride, border_mode='half')
        activation = op(self.W, input, self.target_shape[2:])

        if self.b is not None:
            activation += self.b.dimshuffle('x', 0, 'x', 'x')

        return self.nonlinearity(activation)

    def get_output_shape_for(self, input_shape):
        return self.target_shape

# minibatch discrimination layer
class MinibatchLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_kernels, dim_per_kernel=5, theta=lasagne.init.Normal(0.05),
                 log_weight_scale=lasagne.init.Constant(0.), b=lasagne.init.Constant(-1.), **kwargs):
        super(MinibatchLayer, self).__init__(incoming, **kwargs)
        self.num_kernels = num_kernels
        num_inputs = int(np.prod(self.input_shape[1:]))
        self.theta = self.add_param(theta, (num_inputs, num_kernels, dim_per_kernel), name="theta")
        self.log_weight_scale = self.add_param(log_weight_scale, (num_kernels, dim_per_kernel), name="log_weight_scale")
        self.W = self.theta * (T.exp(self.log_weight_scale)/T.sqrt(T.sum(T.square(self.theta),axis=0))).dimshuffle('x',0,1)
        self.b = self.add_param(b, (num_kernels,), name="b")
        
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], np.prod(input_shape[1:])+self.num_kernels)

    def get_output_for(self, input, init=False, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)
        
        activation = T.tensordot(input, self.W, [[1], [0]])
        abs_dif = (T.sum(abs(activation.dimshuffle(0,1,2,'x') - activation.dimshuffle('x',1,2,0)),axis=2)
                    + 1e6 * T.eye(input.shape[0]).dimshuffle(0,'x',1))

        if init:
            mean_min_abs_dif = 0.5 * T.mean(T.min(abs_dif, axis=2),axis=0)
            abs_dif /= mean_min_abs_dif.dimshuffle('x',0,'x')
            self.init_updates = [(self.log_weight_scale, self.log_weight_scale-T.log(mean_min_abs_dif).dimshuffle(0,'x'))]
        
        f = T.sum(T.exp(-abs_dif),axis=2)

        if init:
            mf = T.mean(f,axis=0)
            f -= mf.dimshuffle('x',0)
            self.init_updates.append((self.b, -mf))
        else:
            f += self.b.dimshuffle('x',0)

        return T.concatenate([input, f], axis=1)

class ExtractMiddleLayer(lasagne.layers.Layer):
    def __init__(self, incoming, extra=0, **kwargs):
	super(ExtractMiddleLayer, self).__init__(incoming, **kwargs)
	self.incoming_dim = lasagne.layers.get_output_shape(incoming)
	self.extra = extra

    def get_output_shape_for(self, input_shape):
	extra = self.extra
	return (input_shape[0], 3, 32 + 2*extra, 32 + 2*extra)#input_shape[1], input_shape[2]/2, input_shape[3]/3)

    def get_output_for(self, input, **kwargs):
        extra = self.extra
        print self.incoming_dim, self.incoming_dim[0]
        middle_part = T.zeros(shape=(self.incoming_dim[0], 3, 32 + 2*extra, 32 + 2*extra), dtype=theano.config.floatX)
        middle_part = T.set_subtensor(middle_part[:, :, :, :], input[:, :, 16-extra:48+extra, 16-extra:48+extra])
	return middle_part
	

class BatchNormLayer(lasagne.layers.Layer):
    def __init__(self, incoming, b=lasagne.init.Constant(0.), g=lasagne.init.Constant(1.), nonlinearity=relu, **kwargs):
        super(BatchNormLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = nonlinearity
        k = self.input_shape[1]
        if b is not None:
            self.b = self.add_param(b, (k,), name="b", regularizable=False)
        if g is not None:
            self.g = self.add_param(g, (k,), name="g", regularizable=False)
        self.avg_batch_mean = self.add_param(lasagne.init.Constant(0.), (k,), name="avg_batch_mean", regularizable=False, trainable=False)
        self.avg_batch_var = self.add_param(lasagne.init.Constant(1.), (k,), name="avg_batch_var", regularizable=False, trainable=False)
        if len(self.input_shape)==4:
            self.axes_to_sum = (0,2,3)
            self.dimshuffle_args = ['x',0,'x','x']
        else:
            self.axes_to_sum = 0
            self.dimshuffle_args = ['x',0]

    def get_output_for(self, input, deterministic=False, **kwargs):
        if deterministic:
            norm_features = (input-self.avg_batch_mean.dimshuffle(*self.dimshuffle_args)) / T.sqrt(1e-6 + self.avg_batch_var).dimshuffle(*self.dimshuffle_args)
        else:
            batch_mean = T.mean(input,axis=self.axes_to_sum).flatten()
            centered_input = input-batch_mean.dimshuffle(*self.dimshuffle_args)
            batch_var = T.mean(T.square(centered_input),axis=self.axes_to_sum).flatten()
            batch_stdv = T.sqrt(1e-6 + batch_var)
            norm_features = centered_input / batch_stdv.dimshuffle(*self.dimshuffle_args)

            # BN updates
            new_m = 0.9*self.avg_batch_mean + 0.1*batch_mean
            new_v = 0.9*self.avg_batch_var + T.cast((0.1*input.shape[0])/(input.shape[0]-1),th.config.floatX)*batch_var
            self.bn_updates = [(self.avg_batch_mean, new_m), (self.avg_batch_var, new_v)]

        if hasattr(self, 'g'):
            activation = norm_features*self.g.dimshuffle(*self.dimshuffle_args)
        else:
            activation = norm_features
        if hasattr(self, 'b'):
            activation += self.b.dimshuffle(*self.dimshuffle_args)

        return self.nonlinearity(activation)

def batch_norm(layer, b=lasagne.init.Constant(0.), g=lasagne.init.Constant(1.), **kwargs):
    """
    adapted from https://gist.github.com/f0k/f1a6bd3c8585c400c190
    """
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = lasagne.nonlinearities.identity
    else:
        nonlinearity = lasagne.nonlinearities.identity
    if hasattr(layer, 'b'):
        del layer.params[layer.b]
        layer.b = None
    return BatchNormLayer(layer, b, g, nonlinearity=nonlinearity, **kwargs)

class GaussianNoiseLayer(lasagne.layers.Layer):
    def __init__(self, incoming, sigma=0.1, **kwargs):
        super(GaussianNoiseLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
        self.sigma = sigma

    def get_output_for(self, input, deterministic=False, use_last_noise=False, **kwargs):
        if deterministic or self.sigma == 0:
            return input
        else:
            if not use_last_noise:
                self.noise = self._srng.normal(input.shape, avg=0.0, std=self.sigma)
            return input + self.noise


# incoming is the previous TransposedConvolutionLayer
# masked_image is a bs x 3 x 64 x 64 Tensor (NOT LASAGNE LAYER)
# mixing_coef is a theano shared int
class ResetDeconvLayer(lasagne.layers.Layer):

    def __init__(self, incoming, masked_image,mixing_coef, border=None, trainable=True, **kwargs):
	super(ResetDeconvLayer, self).__init__(incoming, **kwargs)
	self.incoming_dim = lasagne.layers.get_output_shape(incoming)
	#self.input_shape = self.incoming_dim
	self.masked_image = masked_image
	self.name = 'reset deconv layer'
	#mixing_coef = 1 * np.ones_like(self.incoming_dim)
	self.mixing_coef = mixing_coef#self.add_param(lasagne.init.Constant(mixing_coef), self.incoming_dim , name="mix_coef", trainable=trainable, regularizable=False)
	#self.input_layer = incoming.input_layer
	self.border = border

    def get_output_for(self, input, **kwargs):
	# first we check by max pooling ratio : 
    	b_s, channel_gen_out, height_gen_out, width_gen_out = self.incoming_dim
    	# TODO : fetch this dynamically 
	channel_img, height_img, width_img = 3, 64, 64

        assert height_img % height_gen_out == 0

    	pool_size = height_img / height_gen_out
    	channel_repeat = channel_gen_out / channel_img
    	repeat_red, repeat_green, repeat_blue = channel_repeat, channel_repeat, channel_repeat
    	repeat_blue += channel_gen_out % channel_repeat
	
	# only apply MaxPool if input.shape[2:] != masked_image.shape[2:]
	if pool_size != 1: 
	    print 'pooling by' + str(pool_size)
	    pooled_image = pool.pool_2d(self.masked_image, (pool_size, pool_size), ignore_border=True)
	else : 
	    print 'no pooling (should only be for last layer)'
	    pooled_image = self.masked_image

	# now we need to adjust the depth of the original image
	if repeat_green != 1 : 
	    print 'adding depth to original image'
	    downsized_image = T.concatenate([
        	T.repeat(pooled_image[:, 0, None, :, :], repeat_red, axis=1),
        	T.repeat(pooled_image[:, 1, None, :, :], repeat_green, axis=1),
        	T.repeat(pooled_image[:, 2, None, :, :], repeat_blue, axis=1)], axis=1)
	else : 
	    print 'no depth added (should only to the last layer)'
	    downsized_image = self.masked_image

        # here, downsized_image and input should be 2 tensors of equal shape
        # we simply average them out here. EDIT : only average out overlapping parts. 
        # for the center (mask) discard completely the downsized_image (as it is simply
        # a black hole, and keep all of input)

    	if height_gen_out >= 4:
            center = height_gen_out / 2
	    if self.border == None : 
		top = center + height_gen_out / 4
                bottom = center - height_gen_out / 4
                print height_gen_out, center, bottom, top

	        # middle_part should have borders = 0 and center = pooled image content
	        middle_part = T.zeros(shape=self.incoming_dim, dtype=theano.config.floatX)
                middle_part = T.set_subtensor(middle_part[:, :, bottom:top, bottom:top], input[:, :, bottom:top ,bottom:top])
                outer_part =  input - middle_part
	        # TODO : check if next line changes anythin (it shouldn't)
	        downsized_image_ = T.set_subtensor(downsized_image[:, :, bottom:top, bottom:top], 0. )

	    else :
		# TODO: finish this section
		# goal : take only a small outer border of the original image, and the rest leave as 
		print 'border', self.border
		top = height_gen_out - self.border
		bottom = self.border 
		inner_part = T.zeros(shape=self.incoming_dim, dtype=theano.config.floatX)
		generated_middle = T.set_subtensor(inner_part[:, :, bottom:top, bottom:top], input[:, :, bottom:top, bottom:top])
		generated_border = input - generated_middle
		downsized_image_border = T.set_subtensor(downsized_image[:, :, bottom:top, bottom:top], 0.)
		avgd_ = generated_middle + self.mixing_coef * downsized_image_border + (1. - self.mixing_coef) * generated_border
		return avgd_		
        

            # avgd_ = (downsized_image_ + middle_part) * mixing_coef + (1. - mixing_coef) * gen_output_
	    avgd_ = middle_part + self.mixing_coef * downsized_image_  + (1. - self.mixing_coef) * outer_part

        else:
	    # TODO : remove this  
	    print 'should never get here'
            avgd_= (gen_output_ + downsized_image_) / 2

	return avgd_


    def get_output_shape_for(self, input_shape):
	    return input_shape
