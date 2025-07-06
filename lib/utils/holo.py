import tensorflow as tf
import numpy as np
import scipy.special as sps
import matplotlib.pyplot as plt
import glob
from aotools.functions import zernikeArray
from tqdm.notebook import tqdm
import time
import os
import lib.utils.utils as ut

# This module provides functions for OAM multiplexing holography.

######################  HOLO
def create_sampling_vector(N, d, p, shift  = 'center'):
    """
    Creates a 1D sampling vector.
    Args:
        N (int): Length of the vector.
        d (int): Sampling interval.
        p (int): Number of pixels per sampling point.
        shift (str or int): Shift of the sampling pattern. If 'center', the pattern is centered.
                            If an integer, it specifies the number of positions to shift.
    Returns:
        np.ndarray: A 1D numpy array of length N with the sampling pattern.
    """
    pattern = np.array([1]*p + [0]*(d-p))
    vector = np.tile(pattern, N//d + 1)[:N]
    vector = np.roll(vector,-(p//2))
    if shift == 'center':
        vector = np.roll(vector,(N%d)//2)
    elif isinstance(shift, (int, np.integer)):
        vector = np.roll(vector, shift)
    else:
        raise ValueError("shift must be center or int")
    return vector

def create_sampling_matrix(N, ind):
    """
    Creates a 2D sampling matrix from the given dimensions and indices.
    """
    [Ny,Nx] = N
    [indy,indx] = ind
    S = np.zeros((Ny,Nx))
    #S[ind[:,None], ind[None,:]] = 1
    S[np.ix_(indy, indx)] = 1
    return S

def make_S_array(Nx, Ny, d, w = 1, shift = ['center','center']):
    """
    Creates a 2D sampling matrix and indices for the sampling array.
    Args:
        Nx (int): Number of columns in the sampling matrix.
        Ny (int): Number of rows in the sampling matrix.
        d (int): Sampling interval.
        w (int): Number of pixels per sampling point.
        shift (list): List of two elements specifying the shift for x and y dimensions.
    Returns:
        S (tf.Tensor): A 2D sampling matrix of shape (Ny, Nx).
        ind2D (tf.Tensor): A 2D tensor containing the indices of the sampling points.
    """
    Sx = create_sampling_vector(Nx, d, w, shift[0])
    Sy = create_sampling_vector(Ny, d, w, shift[1])
    indSx = np.where(np.array(Sx) == 1)[0]
    indSy = np.where(np.array(Sy) == 1)[0]
    S = create_sampling_matrix([Ny, Nx],[indSy, indSx])
    S = tf.cast(S,tf.complex64)
    ind2D = tf.stack(tf.meshgrid(indSy, indSx,indexing='ij'), -1)
    ind2D = tf.cast(ind2D,tf.int32)
    return S, ind2D

# Padding function
def offpadding(E_in,paddings):
    E = tf.pad(E_in, paddings, "CONSTANT")
    return E

# Propagates light field backward using FFT.
def backward_E(Eimg):
    Ny = Eimg.shape[1]
    Nx = Eimg.shape[2]
    E = tf.transpose(Eimg,[0,3,1,2])
    E = tf.signal.fftshift(tf.signal.fft2d(tf.signal.fftshift(E,[2,3])),[2,3])/np.sqrt(Ny*Nx)
    E = tf.transpose(E,[0,2,3,1])
    return E

# Propagates light field forward using FFT.
def forward_E(Eslm,padf = None):
    Ny = Eslm.shape[1]
    Nx = Eslm.shape[2]
    E = tf.transpose(Eslm,[0,3,1,2])
    E = tf.signal.fftshift(tf.signal.ifft2d(tf.signal.ifftshift(E,[2,3])),[2,3])*np.sqrt(Ny*Nx)
    E = tf.transpose(E,[0,2,3,1])
    return E


def GS_step(E_in, A_obj, A_holo, padf = None):
    """
    Performs one step of the Gerchberg-Saxton algorithm.
    Args:
        E_in (tf.Tensor): Input electric field.
        A_obj (tf.Tensor): Amplitude mask for the object.
        A_holo (tf.Tensor): Amplitude mask for the hologram.
        padf (None): Padding function, not used in this implementation.
    """
    E = A_obj * ut.unityamp(E_in)
    E_holo = backward_E(E)
    E_holo = A_holo * ut.unityamp(E_holo)
    E_out = forward_E(E_holo)
    return E_out

def GS(E_in, A_obj, A_holo, itr =100, loc = 'obj'):
    """
    Performs the Gerchberg-Saxton algorithm for phase retrieval. (confined)
    Args:
        E_in (tf.Tensor): Input electric field.
        A_obj (tf.Tensor): Amplitude mask for the object.
        A_holo (tf.Tensor): Amplitude mask for the hologram.
        itr (int): Number of iterations for the algorithm.
        loc (str): Location of the output field, either 'obj' or 'holo'.    
    Returns:
        E_out (tf.Tensor): Output electric field after the Gerchberg-Saxton iterations.
    """
    E_out = E_in
    for i in range(itr):
        E_out = GS_step(E_out, A_obj, A_holo)

    if loc == 'holo':
        E_out = backward_E(E_out)
        E_out = A_holo * ut.unityamp(E_out)
    return E_out

def GS_wpad_step(E_in, A_obj_keep, A_obj_replace, A_holo):
    """
    Performs one step of the Gerchberg-Saxton algorithm. (non-confined)
    Args:
        E_in (tf.Tensor): Input electric field.
        A_obj_keep (tf.Tensor): Amplitude mask for the object to keep.
        A_obj_replace (tf.Tensor): Amplitude mask for the object to replace.
        A_holo (tf.Tensor): Amplitude mask for the hologram.
    Returns:
        E_out (tf.Tensor): Output electric field after the Gerchberg-Saxton step
    """
    E = E_in * A_obj_keep + ut.unityamp(E_in) * A_obj_replace
    E_holo = backward_E(E)
    E_holo = A_holo * ut.unityamp(E_holo)
    E_out = forward_E(E_holo)
    return E_out

def GS_wpad(E_in, A_obj_keep, A_obj_replace, A_holo, itr =100, loc = 'obj'):
    """
    Performs the Gerchberg-Saxton algorithm for phase retrieval. (non-confined)
    Args:
        E_in (tf.Tensor): Input electric field.
        A_obj_keep (tf.Tensor): Amplitude mask for the object to keep.
        A_obj_replace (tf.Tensor): Amplitude mask for the object to replace.
        A_holo (tf.Tensor): Amplitude mask for the hologram.
        itr (int): Number of iterations for the algorithm.
        loc (str): Location of the output field, either 'obj' or 'holo'.
    Returns:
        E_out (tf.Tensor): Output electric field after the Gerchberg-Saxton iterations.
    """
    E_out = E_in
    for i in range(itr):
        E_out = GS_wpad_step(E_out,A_obj_keep,A_obj_replace, A_holo)

    if loc == 'holo':
        E_out = backward_E(E_out)
        E_out = A_holo * ut.unityamp(E_out)
    return E_out


def compute_zernike_basis(num_polynomials, field_size, dtype=tf.float32, wo_piston=False, aperture = 'circ'):
    """Computes a set of Zernike basis function with resolution field_res
    Args:
        num_polynomials: number of Zernike polynomials in this basis
        field_size: [height, width] in px, any list-like object
        wo_piston: if True, excludes the piston term (Zernike polynomial 0 and 1)
        aperture: 'circ' for circular aperture, 'square' for square aperture
    Returns:
        zernike: tensor of Zernike polynomials (height, width, num_polynomials)

    """
    # size the zernike basis to avoid circular masking
    if aperture == 'circ':
        zernike_diam = field_size
    elif aperture == 'square':
        zernike_diam = int(np.ceil(np.sqrt(field_size[0]**2 + field_size[1]**2)))
        
    # create zernike functions
    #print(zernike_diam)
    if not wo_piston:
        zernike = zernikeArray(num_polynomials, zernike_diam)
    else:  # 200427 - exclude pistorn term
        idxs = range(2, 2 + num_polynomials)
        zernike = zernikeArray(idxs, zernike_diam)
        
    # convert to tensor and create phase
    zernike = tf.constant(zernike, dtype=dtype)
    if aperture == 'square':
        #wrong fix later   zer = compute_zernike_basis(10, field_size = (suslm.Ny,suslm.Nx), aperture = 'square')
        dy = (zernike_diam - field_size[0])//2
        dx = (zernike_diam - field_size[1])//2
        #print(zernike_diam,field_size[0],field_size[1],dy,dx)
        zernike = zernike[:,dy:dy+field_size[0],dx:dx+field_size[1]]
    zernike = tf.transpose(zernike,[1,2,0])
    return zernike

def apply_NN(I, NNs, mode = 'mag', Nch = 1):
    """
    Applies a series of neural networks to an input image.
    Args:
        I (tf.Tensor): Input image tensor of shape (batch_size, height, width, channels).
        NNs (list): List of neural network layers to apply sequentially.
        mode (str): Mode of operation, either 'mag' for magnitude or 'int' for intensity.
        Nch (int): Number of channels in the output image.
    Returns:
        E_img (tf.Tensor): Output image tensor after applying the neural networks.
    """
    if mode == 'mag':
        out_NN = NNs[0](I**0.5)
    elif mode == 'int':
        out_NN = NNs[0](I)
    for i in range(len(NNs)-1):
        out_NN = NNs[i+1](out_NN)
    E_img = tf.complex(out_NN[:,:,:,:Nch],out_NN[:,:,:,Nch:])
    return E_img
    