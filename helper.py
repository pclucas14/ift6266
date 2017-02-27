import numpy as np
from PIL import Image
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import theano.tensor as T
import theano


# taken from lasagne gitbub

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

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def split_image(image):
        img_array = np.array(image)
        ### Get input/target from the images
        center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
        if len(img_array.shape) == 3:
            input = np.copy(img_array)
            input[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = 0
            target = img_array[center[0]-16:center[0]+16, center[1] - 16:center[1]+16, :]
        else:
            input = np.copy(img_array)
            input[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = 0
            target = img_array[center[0]-16:center[0]+16, center[1] - 16:center[1]+16]

        return input,  img_array

def load_dataset(ds_split=(0.9,0.1,0.), shuffle=False, sample=False):
    print("Loading dataset...")

    data_path = "C:/Users/pcluc/Desktop/School/UdeM/Deep Learning 5xx/project/data"
    split="train2014"
    data_path = os.path.join(data_path, split)
    imgs = glob.glob(data_path + "/*.jpg")

    # sample a few TODO : remove this 
    if sample : imgs = imgs[:2000]
    
    X, Y = [], []
    for i, img_path in enumerate(imgs):
        try : 
            img = Image.open(img_path)
            x, y = split_image(img)
            X.append(x)
            Y.append(y)
        except Exception as e: 
            print e
            pass

    Image.fromarray(X[-1]).show()
    Image.fromarray(Y[-1]).show()

    X = np.array(X)
    Y = np.array(Y)

    if shuffle : 
        X = X.reshape((-1, 3, 64, 64))
        Y = Y.reshape((-1,3,32,32))


    amt = X.shape[0]
    idx1 = int(ds_split[0]*amt)
    idx2 = int((ds_split[0] + ds_split[1])*amt)

    return X[:idx1], Y[:idx1], X[idx1:idx2], Y[idx1:idx2], X[idx2:], Y[idx2:] 

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

    out =  real * mask + pred * mask_rev
    return out
    
    
def create_tensor_masks(batch_size):
    center = (32,32)
    # note : the maskes must be built in the dimensions in which the image is
    # displayed. if not transpose operation fucks them up.
    expected_shape= (batch_size,64,64,3)
    mask = np.ones(expected_shape)
    mask_rev = np.zeros(expected_shape)
    mask[:,center[0]-16:center[0]+16, center[1]-16:center[1]+16,:] = 0
    mask_rev[:,center[0]-16:center[0]+16, center[1]-16:center[1]+16,:] = 1
    # now, reshape the masks in the proper dimensions 
    mask = mask.reshape((-1, 3, 64, 64))
    mask_rev = mask_rev.reshape((-1, 3, 64, 64))
    
    
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

    new_im.save('images/' + imageName+ '_epoch' + str(epoch) + '.png')
    
#%%
# method to combine predictions 
x = T.tensor4()
y = T.tensor4()
z = combine_tensor_images(x,y, 256)
combine_tensors = theano.function([x,y],z)
    





    

