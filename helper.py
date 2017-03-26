import numpy as np
from PIL import Image
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import theano.tensor as T
import theano
import lasagne

# taken from lasagne gitbub

home = '/home/lucas/Desktop/' #'/home2/ift6ed47/'
# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.
# Notice that this function returns only mini-batches of size `batchsize`.
# If the size of the data is not a multiple of `batchsize`, it will not
# return the last (remaining) mini-batch.

def iterate_minibatches(inputs, targets,  batchsize, full=None, shuffle=False, 
                        forever=False):
    assert len(inputs) == len(targets)
    
    while True : 
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]#, full[excerpt]
        if not forever: 
            break

def split_image(image,extra=0):
        img_array = np.array(image)
        ### Get input/target from the images
        center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
        if len(img_array.shape) == 3:
            input = np.copy(img_array)
            input[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = 0
            target = img_array[center[0]-16-extra:center[0]+16+extra, center[1] - 16-extra:center[1]+16+extra, :]
        else:
            input = np.copy(img_array)
            input[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = 0
            target = img_array[center[0]-16-extra:center[0]+16+extra, center[1] - 16-extra:center[1]+16+extra]

        return input,  target, img_array

def load_dataset(ds_split=(0.95,0.05,0.), shuffle=False, sample=False, resize=False, normalize=False, extra=False):
    print("Loading dataset")
    try :
        f = file(home + "data.bin","rb")
        trainx = np.load(f)
        trainy = np.load(f)
        trainz = np.load(f)
        testx = np.load(f)
        testy = np.load(f)
        testz = np.load(f)
        f.close()
        print('found cached version')
        return trainx, trainy, trainz, testx, testy, testz
    except : 
        data_path = home + "ift6266/inpainting"
        split="train2014"
        data_path = os.path.join(data_path, split)
        imgs = glob.glob(data_path + "/*.jpg")
    
        # sample a few TODO : remove this 
        if sample : imgs = imgs[14000:18000]
        
        X, Y, Z = [], [], []
        for i, img_path in enumerate(imgs):
            try : 
                img = Image.open(img_path)
                x, y, z = split_image(img, extra=extra)
                if resize : 
                    y = scipy.misc.imresize(y, (28,28,1))
                    #Image.fromarray(y).show()
    
                X.append(x)
                Y.append(y)
                Z.append(z)
            except Exception as e: 
                print e
                pass
    
        Image.fromarray(X[-1]).show()
        Image.fromarray(Y[-1]).show()
    
        X = np.array(X)
        Y = np.array(Y)
        Z = np.array(Z)
        
        X = np.transpose(X, axes=[0,3,1,2]).astype('float32')
        Y = np.transpose(Y, axes=[0,3,1,2]).astype('float32')
        Z = np.transpose(Z, axes=[0,3,1,2]).astype('float32')
        
        if normalize : 
            X = (X - np.mean(X)) / np.std(X)
            Y = (Y - np.mean(Y)) / np.std(Y)
            Z = (Z - np.mean(Z)) / np.std(Z)

        amt = X.shape[0]
        idx1 = int(ds_split[0]*amt)
        idx2 = int((ds_split[0] + ds_split[1])*amt)
        
        f = file(home + "data.bin","wb")
        np.save(f,X[:idx1])
        np.save(f,Y[:idx1])
        np.save(f,Z[:idx1])
        np.save(f,X[idx1:])
        np.save(f,Y[idx1:])
        np.save(f,Z[idx1:])
        f.close()
            
        return X[:idx1], Y[:idx1], Z[:idx1], X[idx1:], Y[idx1:], Z[idx1:]


#%%
def combine_images(real, pred):
    expected_shape= (64,64,3)
    center = (32,32)

    if real.shape != expected_shape : real = np.reshape(real, expected_shape)
    if pred.shape != expected_shape : pred = np.reshape(pred, expected_shape)

    mixed = np.copy(real)
    #print center
    #print pred.shape #[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] 
    
    mixed[center[0]-16:center[0]+16, center[0]-16:center[0]+16 , :] = pred[center[0]-16:center[0]+16, center[0]-16:center[0]+16 , :] 

    return mixed

#%%
def combine_tensor_images(real, pred, batch_size, masks=None):

    if masks == None : 
        mask, mask_rev = create_tensor_masks(batch_size)
    else : 
        mask, mask_rev = masks

    out =  real * mask + (1-mask) * pred
    return out
    

#%%

def fit_middle(contour, pred):
    #assert contour.shape == (-1, 3, 64, 64)
    #assert pred.shape == (-1, 3, 32, 32)
    center = (32,32)
    
    final  = np.copy(contour)
    final[:, :, center[0]-16:center[0]+16, center[0]-16:center[0]+16 ] = pred[:,:,:,]
    
    return final

def fit_middle_tensor(contour, pred, batch_size=256):
    #assert contour.shape == (-1, 3, 64, 64)
    #assert pred.shape == (-1, 3, 32, 32)
    center = (32,32)
    
    final  = contour
    indices = np.array([list(np.arange(0,batch_size)),[0,1,2], list(np.arange(center[0]-16,center[0]+16)), list(np.arange(center[0]-16,center[0]+16))])
    ind_t = tuple([slice(0, batch_size), slice(0,3), slice(center[0]-16,center[0]+16), slice(center[0]-16,center[0]+16)])
    
    final = T.set_subtensor(final[:, :, center[0]-16:center[0]+16, center[0]-16:center[0]+16],pred[:,:,16:48, 16:48])
    
    return final

def fit_middle_extra(contour, pred, extra=4):
    #assert contour.shape == (-1, 3, 64, 64)
    #assert pred.shape == (-1, 3, 32, 32)
    center = (32,32)
    
    final  = np.copy(contour)
    final[:, :, center[0]-16:center[0]+16, center[0]-16:center[0]+16 ] = pred[:,:,extra:-extra,extra:-extra]
    
    return final

def fit_middle_extra_tensor(contour, pred, extra=4):
    #assert contour.shape == (-1, 3, 64, 64)
    #assert pred.shape == (-1, 3, 32, 32)
    center = (32,32)
    
    final  = contour
    final = T.set_subtensor(final[:, :, center[0]-16:center[0]+16, center[0]-16:center[0]+16], pred[:,:,extra:-extra,extra:-extra])
    
    return final

def contour_delta_tensor(contour, pred, extra=4):
    # contour is the 64 x 64 image with a hole in the middle
    # prd ist the 40 x 40 patch. we want to extract contour - pred only where
    # they overlap
    center = (32,32)
    
    contour_p =  T.set_subtensor(contour[:, :, center[0]-16:center[0]+16, center[0]-16:center[0]+16],0)
    pred_p = T.set_subtensor(pred[:, :, extra:-extra, extra:-extra],0)
    contour_p = contour_p[:, :,  center[0]-16-extra:center[0]+16+extra, center[0]-16-extra:center[0]+16+extra]
    
    
    final = contour_p - pred_p
    
    return final


    
def create_tensor_masks(batch_size):
    center = (32,32)
    # note : the maskes must be built in the dimensions in which the image is
    # displayed. if not transpose operation fucks them up.
    expected_shape= (batch_size,3,64,64)
    mask = np.ones(expected_shape)
    mask_rev = np.zeros(expected_shape)
    mask[:,:,center[0]-16:center[0]+16, center[1]-16:center[1]+16] = 0
    mask_rev[:,:,center[0]-16:center[0]+16, center[1]-16:center[1]+16] = 1
    # now, reshape the masks in the proper dimensions 
    # mask = mask.reshape((-1, 3, 64, 64))
    # mask_rev = mask_rev.reshape((-1, 3, 64, 64))
    mask = mask.astype('float32')
    mask_rev = mask.astype('float32')
    
    
    return theano.shared(mask), theano.shared(mask_rev)
    
    

#%%
def saveImage(imageData, imageName, epoch):

    #creates a new empty image, RGB mode, and size 400 by 400.
    new_im = Image.new('RGB', (64*4,64*4))
    
    imageData = imageData.reshape((-1,64,64,3))
    
    #Iterate through a 4 by 4 grid with 100 spacing, to place my image
    index = 0
    for i in xrange(0,5*64,64):
        for j in xrange(0,5*64,64):
            #paste the image at location i,j:
            img = Image.fromarray(imageData[index])
            #img.show()
            new_im.paste(img, (i,j))
            index += 1

    new_im.save(home + 'images/' + imageName+ '_epoch' + str(epoch) + '.png')
    
def normalize(array):
    return (array - np.mean(array)) / np.std(array)


# taken from the lasagne wgan repo 
def rmsprop(cost, params, learning_rate, momentum=0.5, rescale=5.):
    
    grads = T.grad(cost=cost, wrt=params)
    
    running_square_ = [theano.shared(np.zeros_like(p.get_value(),dtype=p.dtype), broadcastable=p.broadcastable)
                      for p in params]
    running_avg_ = [theano.shared(np.zeros_like(p.get_value(),dtype=p.dtype), broadcastable=p.broadcastable)
                   for p in params]
    memory_ = [theano.shared(np.zeros_like(p.get_value(),dtype=p.dtype), broadcastable=p.broadcastable)
                       for p in params]
    
    grad_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), grads)))
    not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
    grad_norm = T.sqrt(grad_norm)
    scaling_num = rescale
    scaling_den = T.maximum(rescale, grad_norm)
    # Magic constants
    combination_coeff = 0.9
    minimum_grad = 1E-4
    updates = []
    for n, (param, grad) in enumerate(zip(params, grads)):
       grad = T.switch(not_finite, 0.1 * param,
                       grad * (scaling_num / scaling_den))
       old_square = running_square_[n]
       new_square = combination_coeff * old_square + (
           1. - combination_coeff) * T.sqr(grad)
       old_avg = running_avg_[n]
       new_avg = combination_coeff * old_avg + (
           1. - combination_coeff) * grad
       rms_grad = T.sqrt(new_square - new_avg ** 2)
       rms_grad = T.maximum(rms_grad, minimum_grad)
       memory = memory_[n]
       update = momentum * memory - learning_rate * grad / rms_grad

       update2 = momentum * momentum * memory - (
           1 + momentum) * learning_rate * grad / rms_grad
           
       updates.append((old_square, new_square))
       updates.append((old_avg, new_avg))
       updates.append((memory, update))
       updates.append((param, param + update2))
    return updates

# also taken from same reop
def adjust_hyperp(eta, epoch, num_epochs=100):
    # After half the epochs, we start decaying the learn rate towards zero
    if epoch >= num_epochs // 2:
        progress = float(epoch) / num_epochs
        eta.set_value(lasagne.utils.floatX(initial_eta*2*(1 - progress)))

def update_model_params(model, param_path):
    with np.load(param_path) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(model, param_values)  
#%%
# method to combine predictions 
x = T.tensor4()
y = T.tensor4()
#z = combine_tensor_images(x,y, 64)
#combine_tensors = theano.function([x,y],z)

a = fit_middle_tensor(x,y)
fill_middle = theano.function([x,y],a)

#%%
'''
result = fill_middle(input, samples)* 77 + 83
result = result.transpose(0,2,3,1).astype('uint8')
Image.fromarray(result[0]).show() 

#tx, ty, tz, _, _, _ = load_dataset(sample=True)
pred = tz[:,:,:40,:40]
z = get_contour(tx[100:200], pred[:100])

tbp = z.transpose(0,2,3,1).astype('uint8')
Image.fromarray(tbp[2]).show() 
'''

