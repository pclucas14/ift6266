#For running the model:
import os 
#os.environ["THEANO_FLAGS"] ="mode=DEBUG_MODE"
import lasagne
import theano
from theano import tensor as T
from keras.datasets import mnist
import numpy as np
from helper import load_dataset
from lasagne.layers import InputLayer, ReshapeLayer
import pickle
import matplotlib.pyplot as plt

# placeholders 
input_var = T.tensor4('input')
target_var = T.tensor4('targets')

# hyperparameters
lr = 1e-2
weight_decay = 1e-5


def create_model(input_var):
    net = {}
    data_size=(None,3,64,64) # Batch size x Img Channels x Height x Width

    #Input layer:
    net['data'] = lasagne.layers.InputLayer(data_size, input_var=input_var)

    #Convolution 
    # note that relu is default activation function after conv2d layers
    net['conv1'] = lasagne.layers.Conv2DLayer(net['data'], num_filters=16, filter_size=5, pad='same')
    net['drop1'] = lasagne.layers.DropoutLayer(net['conv1'], p=0.5)
    #net['conv2'] = lasagne.layers.Conv2DLayer(net['drop1'], num_filters=32, filter_size=4, pad='valid')
    #net['drop2'] = lasagne.layers.DropoutLayer(net['conv2'], p=0.5)
    net['conv3'] = lasagne.layers.Conv2DLayer(net['drop1'], num_filters=16, filter_size=4, pad='same')
    net['drop3'] = lasagne.layers.DropoutLayer(net['conv3'], p=0.5)
    net['pool1'] = lasagne.layers.Pool2DLayer(net['drop3'], pool_size=2)

    net['out'] = lasagne.layers.Conv2DLayer(net['pool1'], num_filters=3, filter_size=3, 
            nonlinearity = lasagne.nonlinearities.linear, pad='same')
    return net


def plot_predictions(test_samples):
    plt.figure(figsize=(3,2))
    f, axarr = plt.subplots(2,5)
    dim = test_samples[0].shape[2]
    for i in range(2):
        for j in range(5):
            index = i*5+j
            axarr[i,j].imshow(test_samples[index].reshape(dim, dim, 3),interpolation='nearest')
            axarr[i,j].axis('off')
    plt.show()


def save_model_weights(model):
    layer_weights = {}
    for name, layer in model.iteritems():
        layer_weights[name] = lasagne.layers.get_all_param_values(layer)
    # save dictionnary
	with open('layer_weights2.pickle', 'wb') as handle:
		pickle.dump(layer_weights, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print 'successfully saved model'


def load_model_weights(model):
    # load dictionnary
    with open('layer_weights2.pickle', 'rb') as handle:
        layer_weights = pickle.load(handle)

    for name, layer in model.iteritems():
        lasagne.layers.set_all_param_values(layer, layer_weights[name])
    print 'successfully loaded model'



net = create_model(input_var)
#load_model_weights(net)

'''
model setup / config 
'''

#Loss function: MSE for now 
prediction = lasagne.layers.get_output(net['out'])
loss_train = lasagne.objectives.squared_error(prediction,target_var)
loss = T.mean(loss_train)

# Also add weight decay to the cost function (L2 reg)
# weightsl2 = lasagne.regularization.regularize_network_params(net['out'], lasagne.regularization.l2)
# loss += weight_decay * weightsl2

#Get the update rule. Here we will use a more advanced optimization algorithm: ADAM [1]
params = lasagne.layers.get_all_params(net['out'], trainable=True)
updates = lasagne.updates.adam(loss, params)

test_prediction = lasagne.layers.get_output(net['out'], deterministic=True)
test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
test_loss = test_loss.mean()


#Note that train_fn has a "updates" rule. Whenever we call this function, it updates the parameters of the model.
train_fn = theano.function([input_var, target_var], [loss, prediction], updates=updates, name='train')
val_fn = theano.function([input_var, target_var], [test_loss, test_prediction], name='validation')
get_preds = theano.function([input_var], test_prediction, name='get_preds')

X_train, y_train, X_test, y_test, X_valid, y_valid = load_dataset()

x_train = X_train.reshape(-1, 3, 64, 64)
x_test = X_test.reshape(-1, 3, 64, 64)
y_train = y_train.reshape(-1, 3, 32, 32)
y_test = y_test.reshape(-1, 3, 32, 32)



import time
epochs = 10 #s 100 # 15  #You can reduce the number of epochs to run it  faster (or run it for longer for a better model)
batch_size=128

#Run the training function per mini-batches
n_examples = x_train.shape[0]
n_batches = n_examples / batch_size


'''
model training
'''
start_time = time.time()

cost_history = []
for epoch in xrange(epochs):
    st = time.time()
    batch_cost_history = []
    for batch in xrange(n_batches):
        x_batch = x_train[batch*batch_size: (batch+1) * batch_size]
        y_batch = y_train[batch*batch_size: (batch+1) * batch_size]
        
        this_cost, prediction = train_fn(x_batch, y_batch) # This is where the model gets updated
        batch_cost_history.append(this_cost)

    epoch_cost = np.mean(batch_cost_history)
    cost_history.append(epoch_cost)
    en = time.time()
    print('Epoch %d/%d, train error: %f. Elapsed time: %.2f seconds' % (epoch+1, epochs, epoch_cost, en-st))
    save_model_weights(net)

indices = np.random.choice(y_test.shape[0], replace=False, size=10)
loss, pred = val_fn(x_test[indices], y_test[indices])
plot_predictions(pred)
plot_predictions(x_test[indices])
plot_predictions(y_test[indices])

plt.plot(cost_history)
plt.show()


