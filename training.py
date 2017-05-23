
# coding: utf-8

# In[1]:

import numpy as np
np.random.seed(1234)
import cPickle as pickle
# import matplotlib
# matplotlib.use('SVG')
# import matplotlib.pyplot as plt
from subprocess import call

import theano.tensor as T
import theano
from lasagne.layers import get_output, InputLayer, DenseLayer, Upscale2DLayer, ReshapeLayer
print 'hello'
from lasagne.init import GlorotUniform
from lasagne.nonlinearities import rectify, leaky_rectify, tanh, sigmoid, softmax
from lasagne.updates import nesterov_momentum, adam
from lasagne.objectives import categorical_crossentropy, binary_crossentropy
from lasagne.layers import Conv2DLayer as Conv2DLayer
from lasagne.layers import MaxPool2DLayer as MaxPool2DLayer
import lasagne
import time
try:
    from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
    from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPool2DLayer
    print 'Using cuda_convnet (faster)'
except ImportError:
    from lasagne.layers import Conv2DLayer as Conv2DLayer
    from lasagne.layers import MaxPool2DLayer as MaxPool2DLayer
    print 'Using lasagne.layers (slower)'
    

from lasagne.layers import Layer
from lasagne import init
from lasagne import nonlinearities
import matplotlib.cm as cm
from math import floor
print 'hello'
import matplotlib
matplotlib.use('SVG')
import matplotlib.pyplot as plt

from scipy.misc import imread

print theano.config.floatX
# In[2]:

def load_StereoImages(dirpath='/home/tanuj/Desktop/major/notebook/Dataset'):
    from load_data import get_data,get_offset
    X_train,y_train,X_valid,y_valid = get_data()

    # X = np.concatenate((X_left,X_right),1)
  
    print X_train.shape,y_train.shape
    
    offset,scale,y_offset,y_scale = get_offset()
    
    # normalize to zero mean and unity variance
    if offset is None:
        offset = np.mean(X_train, 0)
        scale = np.std(X_train, 0).clip(min=1)
    
        y_offset = np.mean(y_train, 0)
        y_scale = y_train.max()
        # prefix_output = '/home/tanuj/Desktop/shapenet/Datasetv2/np/'
        # np.save(prefix_output + 'offset.npy',offset)
        # np.save(prefix_output + 'scale.npy',scale)
        # np.save(prefix_output + 'y_offset',y_offset)
        # np.save(prefix_output + 'y_scale',y_scale)
        print 'lol'
    
    # offset_left = np.mean(X_left_train, 0)
    # scale_left = np.std(X_left_train, 0).clip(min=1)
    # offset_right = np.mean(X_right_train, 0)
    # scale_right = np.std(X_right_train, 0).clip(min=1)
    
    X_train = (X_train - offset) / scale
    X_valid = (X_valid - offset) / scale
    
    # X_left_train = (X_left_train - offset_left) / scale_left
    # X_right_train = (X_right_train - offset_right) / scale_right

    # X_left_valid = (X_left_valid - offset_left) / scale_left
    # X_right_valid = (X_right_valid - offset_right) / scale_right
    
    y_train = y_train/y_scale
    y_valid = y_valid/y_scale
    
    
    return X_train,y_train, X_valid, y_valid
    

from lasagne.layers.normalization import batch_norm

# def batch_norm(layer, **kwargs):
#     nonlinearity = getattr(layer, 'nonlinearity', None)
#     if nonlinearity is not None:
#         layer.nonlinearity = nonlinearities.identity
#     if hasattr(layer, 'b') and layer.b is not None:
#         del layer.params[layer.b]
#         layer.b = None
#     layer = BatchNormLayer(layer, **kwargs)
#     if nonlinearity is not None:
#         from lasagne.layers import NonlinearityLayer
#         layer = NonlinearityLayer(layer, nonlinearity)
#     return layer


# In[6]:

def build_stereo_cnn(input_var=None):
    
    conv_num_filters1 = 16
    conv_num_filters2 = 32
    conv_num_filters3 = 64
    conv_num_filters4 = 128
    filter_size1 = 7
    filter_size2 = 5
    filter_size3 = 3
    filter_size4 = 3
    pool_size = 2
    scale_factor = 2
    pad_in = 'valid'
    pad_out = 'full'

    rows,cols,channels = 200,200,2
    # Input layer, as usual:                                                                                                                                                                                
    network = InputLayer(shape=(None,channels,rows,cols),input_var=input_var,name="input_layer")                                                                                                                             
        
    network = batch_norm(Conv2DLayer(
            network, num_filters=conv_num_filters1, filter_size=(filter_size1, filter_size1),pad=pad_in,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(),name="conv1"))
    
    network = MaxPool2DLayer(network, pool_size=(pool_size, pool_size),name="pool1")
    
    network = batch_norm(Conv2DLayer(
            network, num_filters=conv_num_filters2, filter_size=(filter_size2, filter_size2),pad=pad_in,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(),name="conv2"))
    
    network = MaxPool2DLayer(network, pool_size=(pool_size, pool_size),name="pool2")
                                                                                                                                     
    network = batch_norm(Conv2DLayer(
            network, num_filters=conv_num_filters3, filter_size=(filter_size3, filter_size3),pad=pad_in,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(),name="conv3"))
    
    network = MaxPool2DLayer(network, pool_size=(pool_size, pool_size),name="pool3")
                                                                                                                                     
    network = batch_norm(Conv2DLayer(
            network, num_filters=conv_num_filters4, filter_size=(filter_size4, filter_size4),pad=pad_in,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(),name="conv4"))
    
    network = batch_norm(Conv2DLayer(
            network, num_filters=32, filter_size=(filter_size4, filter_size4),pad=pad_out,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(),name="deconv1"))
    
    network = Upscale2DLayer(network, scale_factor=(pool_size, pool_size),name="upscale1")
    
    network = batch_norm(Conv2DLayer(
            network, num_filters=16, filter_size=(filter_size3, filter_size3),pad=pad_out,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(),name="deconv2"))
    
    network = Upscale2DLayer(network, scale_factor=(pool_size, pool_size),name="upscale2")
    
    network = batch_norm(Conv2DLayer(
            network, num_filters=8, filter_size=(filter_size2, filter_size2),pad=pad_out,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(),name="deconv3"))
    
    network = Upscale2DLayer(network, scale_factor=(pool_size, pool_size),name="upscale3")
    
    network = batch_norm(Conv2DLayer(
            network, num_filters=1, filter_size=(9,9),pad=pad_out,
            nonlinearity=lasagne.nonlinearities.sigmoid,
            W=lasagne.init.GlorotUniform(),name="deconv4"))
                                 
    return network


# In[7]:

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

def iterator(X, batchsize):
    indices = np.arange(len(X))
    for i in range(0, len(X) - batchsize + 1, batchsize):
        sli = indices[i:i+batchsize]
        yield X[sli]


# In[8]:

def save_params(model, fn):
    with open(fn, 'w') as wr:
        pickle.dump(lasagne.layers.get_all_param_values(model), wr)
        
def dice_loss(predictions, targets):
    dice_index = 2*T.sum(predictions*targets)/((T.sum(predictions)+T.sum(targets)))
    return -dice_index


# In[9]:

def main(model='cnn', num_epochs=200):
    # Load the dataset                                                                                                                                                                                      
    print("Loading data...")
    X_train, y_train, X_val, y_val = load_StereoImages()
    
    # indices = np.arange(len(X_train))
    # np.random.shuffle(indices)
    
    # y_train = y_train[...,3:y_train.shape[2]-3,...]
    # y_val = y_val[...,3:y_val.shape[2]-3,...]
    
    print 'X_train type and shape:', X_train.dtype, X_train.shape
    print 'X_train.min():', X_train.min()
    print 'X_train.max():', X_train.max()

    print 'X_val type and shape:', X_val.dtype, X_val.shape
    print 'X_val.min():', X_val.min()
    print 'X_val.max():', X_val.max()
    
    print 'y_train type and shape:', y_train.dtype, y_train.shape
    print 'y_train.min():', y_train.min()
    print 'y_train.max():', y_train.max()

    print 'y_val type and shape:', y_val.dtype, y_val.shape
    print 'y_val.min():', y_val.min()
    print 'y_val.max():', y_val.max()

    # Prepare Theano variables for inputs and targets                                                                                                                                                       
    input_var = T.tensor4('inputs',dtype=theano.config.floatX)
    target_var = T.tensor4('targets',dtype=theano.config.floatX)

    # Create neural network model (depending on first command line parameter)                                                                                                                               
    print("Building model and compiling functions...")

    network = build_stereo_cnn(input_var)
    laylist = lasagne.layers.get_all_layers(network)
    
    with np.load('model_stereo_latest.npz') as f:                                                                                                                                                                       
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]                                                                                                                                     
        lasagne.layers.set_all_param_values(network, param_values)  

    for l in laylist:
        print l.name, lasagne.layers.get_output_shape(l)
        
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    dloss = lasagne.objectives.squared_error(prediction, target_var)
    dloss = dloss.mean()
    
    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates2 = lasagne.updates.adam(
            dloss, params, learning_rate=0.001)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss2 = lasagne.objectives.squared_error(test_prediction,target_var)
    test_loss2 = test_loss2.mean()

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn2 = theano.function([input_var, target_var], dloss, updates=updates2)

    # Compile a second function computing the validation loss and accuracy:
    
    val_fn2 = theano.function([input_var, target_var], [test_loss2])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:                                                                                                                                                                               
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 16, shuffle=True):
            inputs, targets = batch
            train_err += train_fn2(inputs, targets)
            train_batches += 1
#             print train_fn2(inputs, targets)
        # And a full pass over the validation data:
        val_err = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 16, shuffle=False):
            inputs, targets = batch
            err = val_fn2(inputs, targets)
            val_err += err[0]
            val_batches += 1

        # Then we print the results for this epoch:                                                                                                                                                         
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  ratio:\t\t{:.6f}".format((train_err / train_batches)/(val_err / val_batches)))
#         filename = "model4_loss:\t\t{:.6f}".format(train_err / train_batches)
#         np.savez(filename, *lasagne.layers.get_all_param_values(network)) 
        np.savez('model_stereo_v1.2.npz', *lasagne.layers.get_all_param_values(network)) 
#         !git add model_stereo.npz
#         !git commit -m "new stereo model"
#         !git push -u origin master
        
    # Optionally, you could now dump the network weights to a file like this:                                                                                                                               
#     np.savez('model_stereo_v1.2.npz', *lasagne.layers.get_all_param_values(network))  
#     !git add model_stereo_v1.2.npz
# #     !git add Stereo_conv_net_AWS.ipynb
#     !git commit -m "new stereo model"
#     !git push -u origin master
    #                                                                                                                                                                                                       
    # And load them again later on like this:                                                                                                                                                               
    # with np.load('model.npz') as f:                                                                                                                                                                       
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]                                                                                                                                     
    # lasagne.layers.set_all_param_values(network, param_values) 
    return X_train, y_train, X_val, y_val
    


# In[10]:

# X_train, y_train, X_valid, y_valid = main(model='cnn', num_epochs=10)


# In[11]:

# X_train, y_train, X_valid, y_valid = load_StereoImages()
# print X_train.shape, y_train.shape, X_valid.shape, y_valid.shape


# In[12]:
print 'Loading network'
input_var = T.tensor4('inputs',dtype=theano.config.floatX)
network = build_stereo_cnn(input_var)
# network.print_summary()
with np.load('model_stereo_20k_finetuned.npz') as f:                                                                                                                                                                       
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]                                                                                                                                     
    lasagne.layers.set_all_param_values(network, param_values)  
test_prediction = lasagne.layers.get_output(network, deterministic=True)
pred_fn = theano.function([input_var], [test_prediction])
offset = np.load('offset.npy')
scale = np.load('scale.npy')
offset = np.rollaxis(offset,2)
scale = np.rollaxis(scale,2)

# l = imread('L.png',mode = 'RGB')
# r = imread('R.png',mode = 'RGB')

# inp = np.concatenate((l,r),-1)
# inp = np.rollaxis(inp,2)
# inp = (inp-offset)/scale
# out = pred_fn(inp.reshape((1,)+inp.shape))[0]
# print out.shape
# plt.imshow(out.reshape(200,200))
# plt.savefig('out.png')

# # In[13]:

# n = 0
# f, axarr = plt.subplots(4,3, sharey='col')
# f.set_figheight(25)
# f.set_figwidth(15)
# for Im_num in range(60,64):
    
#     a = X_valid[Im_num,0,...]
    
#     b = pred_fn(X_valid[Im_num:Im_num+1,...])
#     c = X_valid[Im_num:Im_num+1,...]
# #     c[0,1,...] = c[0,0,...]
#     b2 = pred_fn(c)
#     b = b[0]
#     b2 = b2[0]
    
    
#     axarr[n,0].imshow(a)
#     axarr[0,0].set_title('Left eye grayscale image')
#     axarr[n,1].imshow(y_valid[Im_num,0,...],cmap='gray')
#     axarr[0,1].set_title('DepthMap ground truth')
#     axarr[n,2].imshow(b[0,0,...],cmap='gray')
#     axarr[0,2].set_title('DepthMap prediction')
    
#     n +=1
# #     f.savefig('foo.png')
# f.show()


# # In[14]:

# # prefix_output = '/output/'
# prefix_output = './'

# # f.savefig('examples.png',cmap = '')
# f.savefig(prefix_output + 'examples_new.png')

# n = 0
# f, axarr = plt.subplots(7,3, sharey='col')
# f.set_figheight(25)
# f.set_figwidth(15)
# for Im_num in range(50,57):
    
#     a = X_valid[Im_num,0,...]
    
#     b = pred_fn(X_valid[Im_num:Im_num+1,...])
#     c = X_valid[Im_num:Im_num+1,...]
# #     c[0,1,...] = c[0,0,...]
#     b2 = pred_fn(c)
#     b = b[0]
#     b2 = b2[0]
    
    
#     axarr[n,0].imshow(a)
#     axarr[0,0].set_title('Left eye grayscale image')
#     axarr[n,1].imshow(y_valid[Im_num,0,...],cmap='viridis')
#     axarr[0,1].set_title('DepthMap ground truth')
#     axarr[n,2].imshow(b[0,0,...],cmap='viridis')
#     axarr[0,2].set_title('DepthMap prediction')
    
#     n +=1
# #     f.savefig('foo.png')
# f.show()


# # In[14]:

# # f.savefig('examples.png',cmap = '')
# f.savefig(prefix_output + 'examples_new2.png')
# # In[ ]:



