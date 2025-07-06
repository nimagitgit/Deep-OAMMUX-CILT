import tensorflow as tf
import numpy as np
from PIL import Image
import glob
from tqdm.notebook import tqdm
#import time
import os
import lib.utils.utils as ut
from numpy import pi

class SLM:
    """
    Class for handling Spatial Light Modulators' (SLMs) operations, including hologram generation and phase modulation.
    The class constructs an SLM object with appropriate parameters such as pixel size, size, and modulation options.
    It supports off-axis modulation, padding and phase wrapping.
    The class can also handle OAM multiplexing holography with optional modulations.
    """

    def __init__(self, suslm, mask, ps= 8e-6, size= np.array([1920,1080]), 
                 mod = False, step= .2, angle= pi/2 - 1.2):
        """
        Initializes the SLM object with the given parameters.
        Args:
            suslm (object): The SLM object containing the spatial light modulator parameters.
            mask (tf.Tensor): The mask tensor for the SLM.
            ps (float): Pixel size in meters.
            size (np.array): Size of the SLM in pixels.
            mod (bool): Whether to apply off-axis modulation or not.
            step (float): Step size for off-axis modulation.
            angle (float): Angle for off-axis modulation in radians.
        """
        self.suslm = suslm
        self.ps = ps
        self.size = size
        self.Nx = size[0]
        self.Ny = size[1]
                 
        self.Px = (self.Nx - suslm.Nx)//2
        self.Py = (self.Ny - suslm.Ny)//2
        self.padding = tf.constant([[0,0],[self.Py, self.Py], [self.Px, self.Px],[0,0]])
        
        x = tf.cast(tf.linspace((-self.Nx//2+1)*ps,(self.Nx//2)*ps,self.Nx),tf.float32)
        y = tf.cast(tf.linspace((-self.Ny//2+1)*ps,(self.Ny//2)*ps,self.Ny),tf.float32)
        X,Y = tf.meshgrid(x, y)
        PHI,R = ut.cart2pol(X, Y)
        
        self.mask = mask
        self.mask_paded = tf.pad(mask, self.padding, "CONSTANT")
        self.mod = mod
        if mod: #to catch wrong usage
            self.angle = angle
            self.step = step/ps
            self.modul_padded = (X*tf.math.cos(self.angle) + Y*tf.math.sin(self.angle))*2*pi*self.step
            self.modul_padded = self.modul_padded[None,...,None]
            self.modul = self.select(self.modul_padded)

    def select(self, X):
        """
        Selects the hologram region from the full SLM region.
        """
        return X[:, self.Py:self.Py + self.suslm.Ny, self.Px:self.Px + self.suslm.Nx,:]

    def pad(self, holo, value = 128):
        """
        Pads the hologram to match the SLM size.
        Args:
            holo (tf.Tensor): The hologram tensor to be padded.
            value (int): The value to pad with, default is 128.
        Returns:
            tf.Tensor: Padded hologram tensor.
        """
        return tf.pad(holo, self.padding, constant_values= value)
    
    def holo_OAM_NN(self, oam, I, NNs, code):
        """
        Generates a hologram using a neural network for OAM multiplexing.
        Args:
            oam (object): The OAM object containing parameters for holography.
            I (tf.Tensor): Input intensity tensor.
            NNs (list): List of neural network layers to process the input.
            code (tf.Tensor): Phase code to be added to the hologram.
        Returns:
            tf.Tensor: Generated hologram tensor.
        """
        out_NN = NNs[0](I)
        for i in range(len(NNs) - 1):
            out_NN = NNs[i+1](out_NN)
        Eimg = tf.complex(out_NN[:,:,:,:oam.Nch], out_NN[:,:,:,oam.Nch:])
        E = ut.backward_E(Eimg)                             
        E = tf.reduce_sum(E, -1, keepdims= True)
        phi_slm = tf.math.angle(E)
        phi_slm += code[None,:,:,:]
        
        phi_slm += self.modul
        phi_slm = tf.math.mod(phi_slm, 2*pi)
        
        #phi_slm = tf.cast(255*mat2gray(phi_slm)*self.window,tf.uint8)

        return phi_slm[0,:,:,:]


    def holo_OAM_GS(self,oam,I,itr,code):
        """
        Generates a hologram using the Gerchberg-Saxton algorithm for OAM multiplexing.
        Args:
            oam (object): The OAM object containing parameters for holography.
            I (tf.Tensor): Input intensity tensor.
            itr (int): Number of iterations for the Gerchberg-Saxton algorithm. 
            code (tf.Tensor): Phase code to be added to the hologram.
        Returns:
            tf.Tensor: Generated hologram tensor.
        """
        Eimg = tf.complex(I**.5,0.)
        E = Eimg * oam.S[None,...,None]
        E = ut.GS(E,Aobj = E, Aholo = oam.H_mask, loc = 'holo',itr = itr)
        E *= ut.phase2cf(-oam.lcode)
        E = tf.reduce_sum(E,-1,keepdims=True)
        phi_slm = tf.math.angle(E)
        phi_slm += code[None,:,:,:]
        
        #phi_slm += self.modul
        
        print(oam.H_mask.shape,phi_slm.shape)
        
        
        phi_slm *= tf.math.abs(oam.H_mask)
        
        phi_slm = tf.math.mod(phi_slm, 2 * pi)
        #phi_slm = tf.cast(255*mat2gray(phi_slm)*self.window,tf.uint8)

        return phi_slm[0,:,:,:]

    
    def holo_GS(self,oam,I,itr):
        """
        Generates a hologram for sparse images using the Gerchberg-Saxton algorithm.
        Args:
            oam (object): The OAM object containing parameters for holography.
            I (tf.Tensor): Input intensity tensor.
            itr (int): Number of iterations for the Gerchberg-Saxton algorithm.
        Returns:
            tf.Tensor: Generated hologram tensor.
        """
        E = tf.complex(I**.5,0.)
        E = ut.GS(E,Aobj = E, Aholo = oam.H_mask, loc = 'holo',itr = itr)
                    
        phi_slm = tf.math.angle(E)
        phi_slm *= tf.math.abs(oam.H_mask)

        return phi_slm[0,:,:,:]
                 
    def wrap(self, phi, range = 2*pi):
        """
        Wraps the phase to a specified range.
        Args:
            phi (tf.Tensor): Input phase tensor.
            range (float): The range to wrap the phase to, default is 2*pi.
        Returns:    
            tf.Tensor: Wrapped phase tensor.
        """ 
        #png = ((tf.math.floormod(phi+pi,2*pi)) / (2 * pi)) * 255.0 * mask
        #png = tf.math.round((phi+pi)*256./2/pi)
        #gray = tf.math.round(tf.math.floormod(phi*256./2/pi+128.,256.))
        return tf.math.floormod(phi * range /2/pi + range/2, range)
    
    def phase2gray(self, phi_in, mask):
        """
        Converts phase to grayscale image.
        Args:
            phi_in (tf.Tensor): Input phase tensor.
            mask (tf.Tensor): Mask tensor to apply to the phase.
        Returns:
            tf.Tensor: Grayscale hologram tensor.
        """

        phi = phi_in
        if self.mod:
            phi += self.modul

        phi *= mask
        phi = self.pad(phi, value= 0.)
        holo = tf.math.round(self.wrap(phi, range = 256.))
        holo= tf.cast(holo,tf.uint8)
        return holo
    
    def get_phi(self, holo):
        """
        Extracts the phase from a grayscale hologram.
        Args:
            holo (tf.Tensor): Input hologram tensor.
        Returns:
            tf.Tensor: Phase tensor extracted from the hologram.
        """
        h = self.select(holo)
        phi = (tf.cast(h, tf.float32) - 128.)* 2 * pi / 256.
        #phi = tf.cast(phi,tf.float32)*2*pi-pi
        return phi
    

# Computes a grayscale hologram from a complex field or a phase profile and saves it as an image.
def makeholosave(x,slm, oam,name,path,shine = False, mod = False):
    mask = tf.math.abs(oam.H_mask)
    num = x.shape[-1]
    if x.dtype == tf.complex64:
        phi = ut.cf2phase(x)
    elif x.dtype == tf.float32:
        phi = x
       
    phi *= mask
    phi = phi.numpy()
    if shine:
        if num > 1:
            print('Wrong channel number')
            return
        phi = oam.shine_oam(phi,mask)
        num = oam.Nch

    holo = slm.phase2gray(phi, mask, mod)
    os.makedirs(path, exist_ok=True)
    for i in range(num):
        image = Image.fromarray(holo[0,:,:,i].numpy())
        if num == 1:
            image.save(path+ name + '.png')        
        else:
            image.save(path+ name + '_'+str(i)+ '.png')        
    
    return holo

# Computes a grayscale hologram from a complex field or a phase profile and saves it as a binary file.
def makeholosavebin(x,oam,name,path,shine = False, data = []):
    mask = tf.math.abs(oam.H_mask)
    SLMA = SLM(suslm = oam.suslm)
    num = x.shape[-1]
    if x.dtype == tf.complex64:
        phi = mask*ut.cf2phase(x)
    elif x.dtype == tf.float32:
        phi = mask*x
    phi = phi.numpy()
    if shine:
        if num > 1:
            print('Wrong channel number')
            return
        phi = oam.shine_oam(phi,mask)
        num = oam.Nch
        
    holo = SLMA.phase2gray(phi,1.)
    os.makedirs(path, exist_ok=True)
    with open(path+ name + '.npy',"wb") as f:
        np.save(f,holo)
        for item in data:
            np.save(f,item)
    return

# Computes a grayscale hologram from a complex field or a phase profile.
def makeholo(x, oam, shine = False):
    mask = tf.math.abs(oam.H_mask)
    SLMA = SLM(suslm = oam.suslm)
    num = x.shape[-1]
    if x.dtype == tf.complex64:
        phi = mask*ut.cf2phase(x)
    elif x.dtype == tf.float32:
        phi = mask*x
    phi = phi.numpy()
    if shine:
        if num > 1:
            print('Wrong channel number')
            return
        phi = oam.shine_oam(phi,mask)
        num = oam.Nch
    holo = SLMA.phase2gray(phi,1.)
    return holo