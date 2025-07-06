import tensorflow as tf
from tensorflow import keras

from keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate, UpSampling2D
from keras.layers import Flatten, Dense
from keras.models import Model
from keras import backend
from keras.activations import tanh

from tensorlayer.activation import hard_tanh

from keras.layers import ReLU, LeakyReLU, BatchNormalization, Lambda
from keras.layers import LayerNormalization

import numpy as np


#norm_fn = BatchNormalization(axis=3)
#norm_fn = LayerNormalization(axis = [1,2])
#class UNET:
#    def __init__(imsize,Nblocks,num_filter, Nin,Nout, Nresb, norm = 'batch', outtype = 'complex'):
#        self.norm = norm


def normalize(x, norm, axis=None):
    """
    Applies normalization to the input tensor x based on the specified norm type.
    Args:
        x (tf.Tensor): input tensor
        norm (str): type of normalization ('layer', 'batch', or None)
    Returns:
        tf.Tensor: normalized tensor
    """
    if norm == 'layer':
        ax = axis if axis is not None else [1, 2, 3]
        return LayerNormalization(axis = ax)(x,training=True)
    elif norm == 'batch':
        ax = axis if axis is not None else -1
        return BatchNormalization(axis=ax)(x,training=True)
    else:
        return x
def C2d(x, out_channels, fs = 3, bias = True):
    # Applies a 2D convolution to the input tensor x.
    return Conv2D(out_channels,kernel_size = (fs,fs), activation = None,
               padding='same', use_bias = bias)(x)

def C2d(x, out_channels, bias = True):
    # Applies a 2D convolution to the input tensor x.
    x = Conv2D(out_channels,kernel_size = (3, 3), activation = None,
               padding='same', use_bias = bias)(x)
    return x

def UpBlock(xin, out_channels, norm = True, post_conv=True, upsampling_mode='transpose'):
    """
    Upsampling block that applies a transposed convolution or upsampling followed by a convolution.
    Args:
        xin (tf.Tensor): input tensor
        out_channels (int): number of output channels
        norm (str): type of normalization ('batch', 'layer', or None)
        post_conv (bool): whether to apply a convolution after upsampling
        upsampling_mode (str): method for upsampling ('transpose', 'bilinear', or 'nearest')
        fs_convtr (int): kernel size for transposed convolution
    Returns:
        tf.Tensor: output tensor after upsampling and convolution
    """
    bias = norm
    x = xin
    if upsampling_mode == 'transpose':
        x = Conv2DTranspose(out_channels,kernel_size=(4,4),
                                    strides= (2,2), padding = 'same',
                                   use_bias = bias)(x)
    elif upsampling_mode == 'bilinear':
        x = UpSampling2D(2,interpolation='bilinear')(x)
        x = C2d(x,out_channels, bias)
    elif upsampling_mode == 'nearest':
        x = UpSampling2D(2,interpolation='nearest')(x)
        x = C2d(x,out_channels, bias)
    else:
        raise ValueError("Unknown upsampling mode!")

    if norm:
        x = LayerNormalization(axis = [1,2,3])(x,training=True)
    x = ReLU()(x)

    if post_conv:
        x =  C2d(x, out_channels, bias)
        if norm:
            x = LayerNormalization(axis = [1,2,3])(x,training=True)

        x = ReLU()(x)
    xout = x
    return xout


def DownBlock(xin, out_channels, norm = True, prep_conv=True):
    """
    Downsampling block that applies a convolution followed by downsampling 
    implemented with a strided convolution.
    Args:
        xin (tf.Tensor): input tensor
        out_channels (int): number of output channels
        fs (int): kernel size for the convolution
        norm (str): type of normalization ('batch', 'layer', or None)
        prep_conv (bool): whether to apply a convolution before downsampling
    Returns:
        tf.Tensor: output tensor after convolution and downsampling
    """
    bias = norm

    if prep_conv:
        x = C2d(xin,out_channels, bias)
        if norm:
            x = LayerNormalization(axis = [1,2,3])(x,training=True)
            x = LeakyReLU(0.2)(x)

            
    x = Conv2D(out_channels, (4, 4), strides=(2, 2), activation = None, padding='same',use_bias = bias)(x)
    if norm:
        x = LayerNormalization(axis = [1,2,3])(x,training=True)
    x = LeakyReLU(0.2)(x)      
    return x
def resblock(xin, n_kernels, fs = 3):
    """
    Residual block used at the bottleneck of the U-Net architecture.
    Applies two convolutional layers with ReLU activation and adds the input tensor to the output.
    Args:
        xin (tf.Tensor): input tensor
        n_kernels (int): number of output channels for the convolutional layers
        fs (int): kernel size for the convolutional layers
        norm (str): type of normalization ('batch', 'layer', or None)
    Returns:
        tf.Tensor: output tensor after the residual block
    """
    x = Conv2D(n_kernels, (fs, fs), activation = None, padding='same')(xin)
    x = LayerNormalization(axis = [1,2,3])(x,training=True)
    x = ReLU()(x)
    x = Conv2D(n_kernels, (fs, fs), activation = None, padding='same')(x)
    x = LayerNormalization(axis = [1,2,3])(x,training=True)
    x = ReLU()(x)
    x = xin + x
    return x

def __unet(imsize, num_blocks,num_filter, Nin,Nout, extra_ups, num_resb, norm = True, 
           fs_out = 3, out_type = 'complex'):
    """
    Constructs a U-Net model with specified parameters.
    Args:
        imsize (tuple): input image size (height, width)
        num_blocks (int): number of downsampling blocks
        num_filter (int): number of filters in the first convolutional layer
        Nin (int): number of input channels
        Nout (int): number of output channels
        extra_ups (int): number of additional upsampling blocks after the main U-Net structure
        num_resb (int): number of residual blocks at the bottleneck
        norm (str): type of normalization ('batch', 'layer', or None)
        fs_out (int): kernel size for the output convolution
        out_type (str): type of output ('scalar', 'complex', or 'phase')
    Returns:
        tf.keras.Model: U-Net model
    """
    xin = Input(shape=[*imsize, Nin])
    x = C2d(xin, num_filter)
    if norm:
        x = LayerNormalization(axis = [1,2,3])(x, training=True)
    x = LeakyReLU(0.2)(x)

    x_encoder = [x]
    for i in range(num_blocks):
        x = DownBlock(x, num_filter*(2**(i+1)), norm=norm, prep_conv=True)
        x_encoder.append(x)
    
    for i in range(num_resb):
        x = resblock(x,num_filter*(2**(num_blocks)), norm=norm)

    for i in range(num_blocks,0,-1):
        x = UpBlock(x, num_filter*(2**(i-1)), norm, post_conv=True, upsampling_mode='transpose')
        x = Concatenate()([x, x_encoder[i-1]])

    for i in range(extra_ups):
        x = UpBlock(x, num_filter, norm, post_conv=True, upsampling_mode='transpose')


    if out_type == 'scalar':
        xout = Conv2D(Nout,(fs_out,fs_out), activation = None, padding='same')(x)
    elif out_type == 'complex':             
        xre = Conv2D(Nout,(fs_out,fs_out), activation = None, padding='same')(x)
        xim = Conv2D(Nout,(fs_out,fs_out), activation = None, padding='same')(x)
        xout = tf.concat([xre,xim],-1)
    elif out_type == 'phase':
        x = Conv2D(Nout,(fs_out,fs_out), activation = None, padding='same')(x)
        x_out = hard_tanh(x) * 2 * np.pi
    else:
        raise ValueError('Wrong output type')
    
    return Model(xin, xout)

def UNET_new(imsize,num_blocks ,num_filter, Nin,Nout, extra_ups = 0, num_resb = 0, norm = 'batch',
         fs_out = 3, out_type = 'complex'):
    """
    Constructs a U-Net model with specified parameters.
    Args:
        imsize (tuple): input image size (height, width)
        num_blocks (int): number of downsampling blocks
        num_filter (int): number of filters in the first convolutional layer
        Nin (int): number of input channels
        Nout (int): number of output channels
        extra_ups (int): number of additional upsampling blocks after the main U-Net structure
        num_resb (int): number of residual blocks at the bottleneck
        norm (str): type of normalization ('batch', 'layer', or None)
        fs_out (int): kernel size for the output convolution
        out_type (str): type of output ('scalar', 'complex', or 'phase')
    Returns:
        tf.keras.Model: U-Net model
    
    Changes from the original __unet function:
    norm has been modified to be a string, so it can be 'batch', 'layer', or None
    output with 'complex' is now a complex tensor instead of a concatenated float tensor
    """
    bias = False if norm else True
    xin = Input(shape=[*imsize,Nin])
    x = C2d(xin,num_filter, bias=bias)
    x = normalize(x,norm)
    x = LeakyReLU(0.2)(x)

    x_encoder = [x]
    for i in range(num_blocks):
        x = DownBlock(x, num_filter*(2**(i+1)), norm=norm, prep_conv=True)
        x_encoder.append(x)
 
    for i in range(num_resb):
        x = resblock(x,num_filter*(2**(num_blocks)), norm=norm)

    for i in range(num_blocks,0,-1):
        x = UpBlock(x, num_filter*(2**(i-1)), norm, post_conv=True, upsampling_mode='transpose')
        x = Concatenate()([x, x_encoder[i-1]])

    for i in range(extra_ups):
        x = UpBlock(x, num_filter, norm, post_conv=True, upsampling_mode='transpose')
        

    if out_type == 'scalar':
        xout = Conv2D(Nout,(fs_out,fs_out), activation = None, padding='same')(x)
    elif out_type == 'complex':             
        xre = Conv2D(Nout,(fs_out,fs_out), activation = None, padding='same')(x)
        xim = Conv2D(Nout,(fs_out,fs_out), activation = None, padding='same')(x)
        xout = tf.complex(xre,xim)
    elif out_type == 'phase':
        x = Conv2D(Nout,(fs_out,fs_out), activation = None, padding='same')(x)
        x_out = hard_tanh(x) * 2 * np.pi
    else:
        raise ValueError('Wrong output type')
    return Model(xin, xout)


class Spatialcode(tf.keras.layers.Layer):
    """
    A custom Keras layer that adds a set of learnable spatial codes to the input tensor.
    The codes are concatenated to the input tensor along the last dimension.
    """
    def __init__(self, variable_shape, **kwargs):
        super(Spatialcode, self).__init__(**kwargs)
        self.variable_shape = variable_shape
    def build(self, input_shape):
        self.codes = self.add_weight(shape=self.variable_shape, initializer="random_normal", trainable=True, name="codes")
    def call(self, inputs):
        inputs_w_codes = tf.keras.layers.Concatenate(axis=-1)([inputs, self.codes[None,...]])
        return inputs_w_codes
    def get_codes(self):
        return self.codes