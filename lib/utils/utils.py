import tensorflow as tf
from tensorflow import keras

import numpy as np
import scipy.special as sps
import matplotlib.pyplot as plt
import glob
from aotools.functions import zernikeArray
from tqdm.notebook import tqdm
import time
import os
from keras.losses import MeanSquaredError

"""
This module provides utility functions for:
- Loading images and binary files.
- Creating custom datasets for training.
- Performing data augmentation.
- Drawing images in a grid format.
- Performing gradient descent optimization.
- Auxiliary functions for complex field manipulation.
"""

######################## LOADING IMAGES AND BINARY FILES ##########

def load_image_gray(fname, insize=None, outsize=None,ch=3, invert=False):
    """
    Loads an image file and converts it to grayscale.
    Args:
        fname (str): Path to the image file.
        insize (tuple, optional): Input size to resize the image. Defaults to None.
        outsize (tuple, optional): Output size to crop or pad the image. Defaults to None.
        ch (int, optional): Number of channels in the image. Defaults to 3.
        invert (bool, optional): Whether to invert the grayscale image. Defaults to False.
    Returns:
        tf.Tensor: Grayscale image tensor.
    """
    t = tf.io.decode_image(tf.io.read_file(fname), channels=ch, expand_animations = False)  
    #t = tf.image.decode_png(tf.io.read_file(fname),channels=0)
    t = tf.image.rgb_to_grayscale(t)
    t = tf.cast(t,dtype=tf.float32)/256.
    if insize is not None:
        t = tf.image.resize(t,insize[:2])
    if invert and tf.reduce_mean(t) > 0.5:
        t = 1 - t    
    if outsize:
        t = tf.image.resize_with_crop_or_pad(t,*outsize)
    return t

# Loads binary data, reshapes it according to the specified input size, and optionally resizes it.
def load_binary(fname, insize, outsize = None):
    t = tf.io.decode_raw(tf.io.read_file(fname),tf.float32)
    if len(insize)==3:
        t = tf.reshape(t,insize)
    elif len(insize)==2:
        t = tf.reshape(t,[*insize,-1])     
    if outsize:
        t = tf.image.resize(t,outsize)
    return t

# Loads an image or binary file based on its extension, resizing it if specified.
def load_any(f,insize=None,outsize=None):
    ff = tf.strings.substr(f,-3,3)
    extlist = tf.constant(['png','bin'], dtype=tf.string)  
    return tf.cond(ff == extlist[0],
                   lambda: load_image_gray(f, insize, outsize),
                   lambda: load_binary(f, insize, outsize))

    
###########################################
# Converts Cartesian coordinates (x, y) to polar coordinates (rho, phi).
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)

# Converts Cartesian coordinates (x, y) to polar coordinates (rho, phi) using TensorFlow.
def cart2pol_tf(x, y):
    rho = tf.math.sqrt(x**2 + y**2)
    phi = tf.math.atan2(y, x)
    return (rho, phi)

# Generates a Bessel beam with specified parameters.
def BesselBeam(nB, kr, R,PHI):
    return sps.jv(nB,kr*R)*np.exp(1j*PHI*nB)

# Normalizes a tensor. 
def norm(x):
    return x/tf.reduce_max(x,[0,1,2],keepdims = True)  
########################


    

###############################
def draw(Xin,size=10,split = 1,aspect = 1,frame = None, space = [.1,.1],savename=None,show=True):
    """
    Draws a grid of images from a tensor Xin.
    Args:
        Xin (tf.Tensor): input tensor of shape [batch, height, width, channels]
        size (int): size of the figure
        split (int): number of images per row
        aspect (float or str): aspect ratio of the images, 'square' for equal width and height
        frame (str or None): if 'all', adds colorbar to all images; if None, removes ticks
        space (list): space between images in the grid [horizontal, vertical]
        savename (str or None): if provided, saves the figure with this name
        show (bool): whether to display the figure
    Returns:
        fig (matplotlib.figure.Figure): the created figure
        ax (matplotlib.axes.Axes): the axes of the figure
    """
    dim = len(Xin.shape)
    if dim == 2:
        X = Xin[None,...,None]
    elif dim == 3:
        X = Xin[None,...]
    else:
        X = Xin
    Nch = X.shape[-1]
    Nb = X.shape[0]
    
    if aspect == 'square':
        aspect = X.shape[2]/X.shape[1]
    Ncol = int(np.ceil(Nch/split))
    Nrow = Nb * split
    Fig_X = X.shape[2]*(Ncol+space[0]*(Ncol-1))
    Fig_Y = X.shape[1]*(Nrow+space[1]*(Nrow-1))
    Fig_asp = Fig_Y/Fig_X * aspect
    fig, ax = plt.subplots(Nrow,Ncol,figsize=(size,Fig_asp*size))
    for i in range(Nb):
        for j in range(Nch):
            x, y = j%Ncol,i*split+j//Ncol
            if Nrow == 1:
                if Ncol == 1:
                    ax_h = ax
                else:
                    ax_h = ax[x]
            elif Ncol == 1:
                ax_h = ax[y]
            else:
                ax_h = ax[y,x]
            img_plot = ax_h.imshow(X[i,:,:,j], cmap = 'gray',interpolation = None,aspect=aspect)#,vmin=-pi, vmax=pi)
            if frame == 'all':
                fig.colorbar(img_plot, ax=ax_h, fraction=0.046, pad=0.04)
            elif frame == None:
                ax_h.set_xticks([])
                ax_h.set_yticks([])
    
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=space[0], hspace=space[1])
    if savename:
        plt.savefig(savename+'.svg', format='svg', bbox_inches='tight', pad_inches=0, transparent=True)

    if show:
        plt.show()
        plt.pause(0.01)
    #return fig, ax
    
def print_module_summary(module):
    """Prints a summary of a TensorFlow module, including its submodules and variables."""
    print(f"Module: {module.__class__.__name__}\n")
    print("Submodules:")
    for submodule in module.submodules:
        print(f"  {submodule.name} ({submodule.__class__.__name__})")
    print("\nVariables:")
    for var in module.variables:
        print(f"  {var.name}, shape={var.shape}, dtype={var.dtype}, trainable={var.trainable}")

    num_trainable_variables = sum(np.prod(variable.shape) for variable in module.trainable_variables)
    print(num_trainable_variables)

###############################
def GD(x, y, fn, lossfn, outnum, lr, itr = 1000, info = 'auto'):
    """
    Performs gradient descent optimization on a function.
    Args:
        x (tf.Tensor): input tensor of shape [batch, height, width, channels]
        y (tf.Tensor): target tensor of shape [batch, height, width, channels]
        fn (callable): function to optimize
        lossfn (callable): loss function to minimize
        outnum (int): number of outputs from the function
        lr (float): learning rate for the optimizer
        itr (int): number of iterations for optimization
        info (str or int): if 'auto', displays progress every 10 iterations; if int, displays every `info` iterations; if None, no display
    Returns:
        X (tf.Variable): optimized input tensor
        y_preds (tf.Tensor): predicted outputs from the function
        err (float): final error value after optimization
    """
    if info == 'auto':
        dispitr = itr//10
    elif info:
        dispitr = info
    else:
        dispitr = None
    #opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum= 0.)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    X = tf.Variable(x)
    time0 = time.time()    
    for i in tqdm(range(itr)):
        with tf.GradientTape(persistent=False) as t:
            outs = fn(X)
            if outnum == 1:
                y_preds = outs[0]
            else:
                y_preds = outs[:outnum]
            err = lossfn(y, y_preds)
            gradients = t.gradient(err, [X])
            opt.apply_gradients(zip(gradients, [X]))
            if dispitr and not i % dispitr:
                print('Error: {:.1e}\t| itr: {}\t| time: {:.3f}'.format(err.numpy(), i, time.time()-time0))

    return X, y_preds, err.numpy()

################################
# Prints colored text in the console.
def print_color(text, color_name = 'RED'):
    
    colors = {
        "RED": '\033[91m',
        "GREEN": '\033[92m',
        "BLUE": '\033[94m',
    }
    END = '\033[0m'  # End color formatting
    color_code = colors.get(color_name.upper(), "")  # Convert color_name to uppercase for case-insensitive matching
    colored_text = f"{color_code}{text}{END}"
    print(colored_text)
    
########################################

# Converts a complex field E to polar coordinates (amplitude and phase).
def cf_cart2pol(E):
    A = tf.math.abs(E)
    phi = tf.math.angle(E)
    return tf.concat([A,phi],-1)

# Calculates the amplitude of a complex field E.
def measure_amp(E):
    return tf.math.abs(E)

# Calculates the intensity of a complex field E.
def measure_int(E):
    return tf.math.pow(tf.math.abs(E),2,name = 'int')

# Converts a phase profile to a complex field with unit amplitude.
def phase2cf(phase):
    return tf.math.exp(tf.complex(0.,phase))

# Calculates the phase of a complex field E.
def cf2phase(E):
    return tf.math.angle(E)

# Converts amplitude and phase to a complex field.
def ampphase2cf(amp,phase):
    return tf.complex(amp,0.)*phase2cf(phase)

# Sets the amplitude of a complex field E to 1, keeping the phase.
def unityamp(E):
    return phase2cf(cf2phase(E))

################################################
# Upsampling and downsampling functions for 2D signals.

# Upsamples a 2D signal by a factor of `upfac` using padding.
def up_pad(sig, upfac):
    upsigp = sig[:,:,None,:,None,:]
    upsigp = tf.pad(upsigp, ((0, 0), (0, 0), (0, upfac-1), (0, 0), (0, upfac-1), (0, 0)))
    upsigp = tf.reshape(upsigp, shape=(-1, sig.shape[1]*upfac, sig.shape[2]*upfac, sig.shape[3]))
    return upsigp

# Upsamples a 2D signal by a factor of `upfac` using tiling.
def up_tile(sig, upfac):
    upsigp = sig[:,:,None,:,None,:]
    upsigp = tf.tile(upsigp, (1, 1, upfac, 1, upfac, 1))
    upsigp = tf.reshape(upsigp, shape=(-1, sig.shape[1]*upfac, sig.shape[2]*upfac, sig.shape[3]))
    return upsigp

# Downsamples a 2D signal by a factor of `downfac` by reshaping and averaging or selecting a single value.
def down_pad2D(sig, downfac, mode = 'single'):
    downsig = tf.reshape(sig, shape=(-1, sig.shape[1]//downfac,downfac, sig.shape[2]//downfac, downfac, sig.shape[3]))
    if mode == 'single':
        downsig = downsig[:,:,0,:,0,:]
    elif mode == 'mean':
        downsig = tf.reduce_mean(downsig,[2,4]) 
    return downsig

def down_pad2D_general(x, fac, mode = 'single', sigma = None, ks = None):
    """
    Downsamples a 2D signal by a factor of `fac` using various schemes.
    Args:
        x (tf.Tensor): Input tensor of shape [batch, height, width, channels].
        fac (int): Downsampling factor.
        mode (str): Downsampling mode:
            - 'single': Selects a single value from each downsampled block.
            - 'mean': Averages the values in each downsampled block.
            - 'conv': Uses a convolution with a Gaussian kernel.
        sigma (float, optional): Standard deviation for Gaussian kernel if mode is 'conv'. Defaults to None.
        ks (int, optional): Kernel size for Gaussian kernel if mode is 'conv'. Defaults to None.
    Returns:
        tf.Tensor: Downsampled tensor.
    """

    if mode == 'conv':
        out = down_pad2D_conv2d(x, fac, sigma=sigma, ks = ks)
        return out
    B, H, W, C = x.shape
    out = x
    if mode == 'mean':
        out  = tf.roll(out, [fac//2, fac//2], [1, 2])
    out  = tf.reshape(out, shape=(-1, H//fac,fac, W//fac, fac, C))
    if mode == 'single':
        out = out[:,:,0,:,0,:]
    elif mode == 'mean':
        out = tf.reduce_mean(out,[2,4]) 
    return out

# Generates a Gaussian kernel for 2D convolution.
def gaussian_kernel(size, sigma):
    ax = np.arange(-size // 2 + 1, size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel.astype(np.float32)

# Downsamples a 2D signal by a factor of `fac` using depthwise convolution with a Gaussian kernel.
def down_pad2D_conv2d(x, fac, sigma=None, ks=None):
    if sigma is None:
        sigma = fac / 2.0
    if ks is None:
        ks = fac * 2 + 1
    kernel_2d = gaussian_kernel(ks, sigma)
    kernel_tf = tf.convert_to_tensor(kernel_2d[:,:,None,None], dtype=tf.float32)
    
    in_channels = x.shape[-1]
    kernel_tf = tf.tile(kernel_tf, [1, 1, in_channels, 1])
    pad_size = ks // 2
    pad = [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]]
    out_re = tf.nn.depthwise_conv2d(tf.math.real(x), kernel_tf, strides=[1, fac, fac, 1], padding=pad)
    out_im = tf.nn.depthwise_conv2d(tf.math.imag(x), kernel_tf, strides=[1, fac, fac, 1], padding=pad)
    out = tf.complex(out_re, out_im)
    return out

# Creates a sparse tensor from sampled indices and values.
def create_tensor_from_sampled(Isampled, ind2D, original_shape):
    # Reshape ind2D to a 2D tensor of shape [num_samples, 2] for xy indices
    num_samples = tf.shape(ind2D)[1] * tf.shape(ind2D)[2]
    batch_indices = tf.zeros([num_samples, 1], dtype=tf.int64)
    xy_indices = tf.reshape(ind2D, [-1, 2])
    indices = tf.concat([batch_indices, xy_indices], axis=1)

    # Flatten Isampled to match the shape of indices
    flat_Isampled = tf.reshape(Isampled, [-1, original_shape[-1]])

    # Create the tensor using tf.scatter_nd
    sparse_tensor = tf.scatter_nd(indices, flat_Isampled, original_shape)

    return sparse_tensor


############## Metrics and Losses ##############

# Computes the logarithm base 10 of a tensor.
def log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

# Computes the Peak Signal-to-Noise Ratio (PSNR) between an image and its mean squared error (MSE).
def PSNR(x,mse):
    max_x = tf.reduce_max(x)
    err = 10*log10(max_x**2/mse)
    return err

# Computes PSNR between two the normalized images.
def PSNRnorm(x,y):
    max_x = tf.reduce_max(x,[0,1,2], keepdims = True)
    max_y = tf.reduce_max(y,[0,1,2], keepdims = True)
    mse_err = MeanSquaredError()(x/max_x,y/max_y)
    err = -10*log10(mse_err)
    return err