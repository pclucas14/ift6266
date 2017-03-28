import numpy as np
from PIL import Image
import os
import glob
import matplotlib
import matplotlib.pyplot as plt
import theano.tensor as T
import theano
import lasagne
import lasagne.layers as ll
matplotlib.use('Agg')


home = '/home2/ift6ed47/' #'/home/lucas/Desktop/'  
global mask
mask = None

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
    name = 'data.bin' if normalize else 'data_og.bin'
    try :
        f = file(home + name,"rb")
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
	    print 'mean z', np.mean(Z)
            print 'std z', np.std(Z)

        amt = X.shape[0]
        idx1 = int(ds_split[0]*amt)
        idx2 = int((ds_split[0] + ds_split[1])*amt)
        
        f = file(home + name ,"wb")
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
# combines 2 64 x 64 images, real gives outer part, pred gives inner part
def combine_tensor_images(real, pred, batch_size):
    global mask
    if mask is None : 
        mask = T.zeros(shape=(batch_size, 1, 64, 64), dtype=theano.config.floatX)
        mask = T.set_subtensor(mask[:, :, 16:48, 16:48], 1.)

    # inner + outer part
    out =  pred * mask + (1-mask) * real
    return out

# method that takes the original image, downsizes it via pooling and layer repeat to 
# "fit" with intermediate generator's output 
# gen_output and masked_images are both Lasagne Layers (InputLayer, BatchNormLayer)
def reset_deconv(masked_image, gen_output):

    # first we check by max pooling ratio : 
    b_s, channel_gen_out, height_gen_out, width_gen_out = ll.get_output_shape(gen_output)
    b_s, channel_img, height_img, width_img = ll.get_output_shape(masked_image)

    assert height_img % height_gen_out == 0

    pool_size = height_img / height_gen_out 
    channel_repeat = channel_gen_out / channel_img
    repeat_red, repeat_green, repeat_blue = channel_repeat, channel_repeat, channel_repeat
    repeat_blue += channel_gen_out % channel_repeat
    # repeat_blue = 3

    # step 1 : apply max pooling to fit height/width
    pooled = ll.MaxPool2DLayer(masked_image, pool_size=pool_size)

    # step 2 : repeat colors of pooled layer
    # I just realized that Lasagne does not have a RepeatLayer module like Keras, 
    # So I'll have to switch back to Theano to do this. Thankfully, Philip Paquette
    # has already implemented something similar in his repo, so I'll take his. Give 
    # credit where it's due. 

    pooled_ = ll.get_output(pooled)
    gen_output_ = ll.get_output(gen_output)

    downsized_image_ = T.concatenate([
        T.repeat(pooled_[:, 0, None, :, :], repeat_red, axis=1),
        T.repeat(pooled_[:, 1, None, :, :], repeat_green, axis=1),
        T.repeat(pooled_[:, 2, None, :, :], repeat_blue, axis=1)], axis=1)

    # here, downsized_image_ and gen_output_ should be 2 tensors of equal shape
    # we simply average them out here. EDIT : only average out overlapping parts. 
    # for the center (mask) discard completely the downsized_image (as it is simply
    # a black hole, and keep all gen_output_)
    
    if height_gen_out >= 4:
        center = height_gen_out / 2
        bottom = center - height_gen_out / 4
        top = center + height_gen_out / 4
        # print height_gen_out, center, bottom, top
        middle_part = T.zeros(shape=ll.get_output_shape(gen_output), dtype=theano.config.floatX)
        middle_part = T.set_subtensor(middle_part[:, :, bottom:top, bottom:top], gen_output_[:, :, bottom:top ,bottom:top])
	downsized_image_ = T.set_subtensor(downsized_image_[:, :, bottom:top, bottom:top], 0. )
	avgd_ = (downsized_image_ + middle_part) 
        #avgd_ = (gen_output_ + downsized_image_  + middle_part) / 2
    
    else:
        avgd_= (gen_output_ + downsized_image_) / 2
    
    # put it back into lasagne format (InputLayer) and return it. 
    return ll.InputLayer(shape=ll.get_output_shape(gen_output), input_var=avgd_)
    
    
#%%
def fit_middle(contour, pred):
    #assert contour.shape == (-1, 3, 64, 64)
    #assert pred.shape == (-1, 3, 32, 32)
    center = (32,32)
    
    final  = np.copy(contour)
    final[:, :, center[0]-16:center[0]+16, center[0]-16:center[0]+16 ] = pred[:,:,:,]
    
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
'''
# method to combine predictions 
x = T.tensor4()
y = T.tensor4()
z = combine_tensor_images(x,y, 64)
combine_tensors = theano.function([x,y],z)

a, b, c, d, e, f = load_dataset(sample=True)

#%%

result = combine_tensors(a[:64], c[64:128])* 77 + 83
result = result.transpose(0,2,3,1).astype('uint8')
Image.fromarray(result[0]).show() 

#tx, ty, tz, _, _, _ = load_dataset(sample=True)
pred = tz[:,:,:40,:40]
z = get_contour(tx[100:200], pred[:100])

tbp = z.transpose(0,2,3,1).astype('uint8')
Image.fromarray(tbp[2]).show() 
'''

