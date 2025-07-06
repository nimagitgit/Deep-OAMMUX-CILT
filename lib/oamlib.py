import tensorflow as tf
import numpy as np
import lib.utils.utils as ut
import lib.utils.holo as uth
import os
import lib
from PIL import Image
from typing import Any, List, Optional, Tuple
from numpy import pi

class Setup:
    """
    Class to define the holography setup for OAM (Orbital Angular Momentum) multiplexing.
    """    
    def __init__(self, Ns , L = 1, ps = 1, wl = .532e-6):
        """
        Initializes the setup with the given parameters.
        Args:
            Ns (tuple): number of pixels in the setup (Nx, Ny, Nz)
            L (float): length of the propagation distance in meters
            ps (float): pixel size in meters
            wl (float): wavelength in meters
        """
        self.Ns = Ns
        self.Nx,self.Ny,self.Nz = Ns

        self.ps = ps
        self.wavelength = wl
        self.L = L

        self._init_geomtery()

    def to_tensor(self,a):
        return tf.constant(a,dtype=tf.float32)[None,:]
    
    # Initialize geometry parameters for the setup.
    def _init_geomtery(self):     
        self.Wx = self.ps * self.Nx
        self.Wy = self.ps * self.Ny

        self.NQx = self.Wx**2/self.Nx/self.wavelength
        self.NQy = self.Wy**2/self.Ny/self.wavelength

        self.x = np.linspace((-self.Nx//2+1)*self.ps,((self.Nx)//2)*self.ps,self.Nx)
        self.y = np.linspace((-self.Ny//2+1)*self.ps,((self.Ny)//2)*self.ps,self.Ny)
        self.z = np.linspace(0.,self.L,self.Nz, endpoint=True)
        
        self.x_tf = self.to_tensor(self.x)
        self.y_tf = self.to_tensor(self.y)
        self.z_tf = self.to_tensor(self.z)

        self.X, self.Y = tf.meshgrid(self.x_tf,self.y_tf)
        self.R, self.PHI = ut.cart2pol_tf(self.X,self.Y)
        
################################################################################
class OAM:
    """
    Class for designing OAM-multiplexed computer generated holograms. This class provides 
    fundamental functions for forward/backward light propagation and decoding images. 
    Additionally, it includes methods for multiplexing/encoding images using OAM modes as 
    spatial codes. These methods include the conventional method using GS (Gerchberg-Saxton) 
    algorithm and propsed methods using U-Net and neural networks.
    """
    def __init__(self, suimg, suslm, 
                 offpp: Tuple[float, float], l_list: List[int], d: int,
                 shift = 'center', slmpadf: int = 0, RA = None,
                 up_func = ut.up_pad):
        """
        Initializes an OAM-multiplexed holography system.

        Args:
            suimg (Setup): Image plane setup object.
            suslm (Setup): SLM plane setup object.
            offpp (Tuple[float, float]): Fractional offset (y, x) between image and SLM padding.
            l_list (List[int]): List of topological charges (OAM channels).
            d (int): Sampling interval in the image plane in pixels.
            shift (str or list): Sampling array shift mode or amount.
            slmpadf (int): Factor for zero-padding at SLM plane (default is 0, no padding). 
            RA (float): Radius for circular hologram mask (optional).
            up_func (function): Function for upsampling or zero-padding at image plane when working in conjuction with zero-padding holograms.
        """

        self.suimg = suimg
        self.suslm = suslm
        self.Nx, self.Ny = suimg.Nx//d, suimg.Ny//d
        self.Nch = len(l_list)
        self.d = d
        self.dl = int(l_list[1]-l_list[0])
        self.l_list = l_list
        self.lcode = suslm.PHI[:,:,None]*l_list[None,None,:]
        self.lcode_E = ut.phase2cf(self.lcode)
        self.RA = RA
        self.shift = shift
        self.offpp = offpp
        if shift == 'center':
            self.shift = ['center','center']
        else:
            self.shift = shift
        self.up_slm = lambda x: up_func(x, 1 + slmpadf)    
        
        # Padding parameters which set the working region at image plane.
        Py = suslm.Ny-suimg.Ny
        Px = suslm.Nx-suimg.Nx
        self.Py0 = np.int32(Py*offpp[0])
        self.Py1 = Py-self.Py0
        self.Px0 = np.int32(Px*offpp[1])
        self.Px1 = Px-self.Px0
        self.imgpad =  tf.constant([[0,0],[self.Py0, self.Py1], [self.Px0, self.Px1],[0,0]])
        # Padding parameters at SLM plane.
        self.slmpadf = slmpadf
        self.slmpadx = int((slmpadf*suslm.Nx)//2)
        self.slmpady = int((slmpadf*suslm.Ny)//2)
        self.slmpad = [[0,0],[self.slmpady,self.slmpady],[self.slmpadx,self.slmpadx],[0,0]]
        # Creating 2D sampling array for MIMO multiplexing.
        self.S, self.ind2D = uth.make_S_array(suimg.Nx, suimg.Ny, d, w = 1, shift = self.shift)
        self.S4 = self.S[None,:,:,None]
        # Creating hologram mask for SLM plane.
        if RA:
            self.H_mask = tf.cast(suslm.R[None,:,:,None]<RA,tf.complex64)
            self.H_padmask = tf.pad(self.H_mask,self.slmpad)
        else:
            self.H_mask = tf.ones([1,suslm.Ny,suslm.Ny,1],tf.complex64)
            self.H_padmask = tf.pad(self.H_mask,self.slmpad)
        self.H_mask_abs = tf.math.abs(self.H_mask)
            


    ############################################################
    # Zero-pads image at the image plane.
    def imgpadding(self,Eimg):
        return tf.pad(Eimg, self.imgpad, "CONSTANT")
    # Selects the working image region at the image plane.
    def imgselect(self,x):
        return x[:, self.Py0:self.suslm.Ny-self.Py1, 
                 self.Px0:self.suslm.Nx-self.Px1, :]
    # Selects the working image region at the image plane with scaling factor s.
    def imgselectscaled(self,x,s):
        return x[:, s*self.Py0:s*self.suslm.Ny-s*self.Py1, 
                 s*self.Px0:s*self.suslm.Nx-s*self.Px1, :]
    # Selects the SLM region from the padded hologram.
    def slmselect(self,x):
        return x[:, self.slmpady:self.slmpady+self.suslm.Ny, 
                 self.slmpadx:self.slmpadx+self.suslm.Nx, :]
    # Selects the SLM region from the padded hologram with scaling factor s.
    def imgselectslmups(self,x):
        Ny = self.suslm.Ny
        Nx = self.suslm.Nx
        return x[:, Ny//2+self.Py0:Ny*3//2-self.Py1, 
                 Nx//2+self.Px0:Nx*3//2-self.Px1, :]
    ####################################################
    # Adds helial phase to the SLM phase.
    def shine_phase(self, phi_slm, mask=1.):
        phis = (phi_slm + self.lcode) * mask
        return phis
    # Adds helial phase to the SLM field.
    def shine_E(self, E_slm, mask=1.):
        Es = E_slm * self.lcode_E * mask
        return Es
    # Samples intenisity on the 2D sampling array and resizes it to the original image size.
    def postproc(self, I):
        I_out = tf.gather_nd(I, self.ind2D, batch_dims=1)
        I_out = tf.image.resize(I_out, [I.shape[1], I.shape[2]], method = 'nearest')
        return I_out
    # Samples intensity on the 2D sampling array. (Multiple versions)
    def samplev1(self, I):
        return tf.gather_nd(I, self.ind2D, batch_dims=1, name = 'sampled')
    def samplev2(self, I):
        b = I.shape[0]
        return tf.gather_nd(I, tf.tile(self.ind2D, [b, 1, 1, 1]), batch_dims = 1)
    def sample(self,x_in):
        x = tf.transpose(x_in, [1, 2, 0, 3])
        return tf.transpose(tf.gather_nd(x, self.ind2D), [2, 0, 1, 3])
    def sample_cost(self, I, ind):
        return tf.gather_nd(I, ind, batch_dims = 1, name = 'sampled')
    
    #################################################################
    def GS_sparse(self, I_in, itr):
        """
        Gerchberg-Saxton algorithm for phase retrieval with sparse sampling, 
        confined in working image region.
        Args:
            I_in (tf.Tensor): Sampled intensity profiles with shape (B, suimg.Ny, suimg.Nx, Nch).
            itr (int): Number of iterations for the GS algorithm.
        Returns:
            tf.Tensor: Output phase hologram with shape (B, suslm.Ny, suslm.Nx, 1).
        """
        mask = self.H_padmask
        I = I_in * tf.math.abs(self.S4)
        I = self.imgpadding(I)
        if self.slmpadf:
            I = self.up_slm(I)
        E = tf.complex(I**0.5, 0.)
        E = uth.GS(E, A_obj = E, A_holo = mask, itr = itr, loc = 'holo')
        phi = tf.math.abs(mask) * tf.math.angle(E)
        if self.slmpadf:
            phi = self.slmselect(phi)
        return phi
      
    def GS_sparse_imgpad(self, I_in, itr):
        """
        Gerchberg-Saxton algorithm for phase retrieval with sparse sampling,
        not confined in working image region.
        Args:
            I_in (tf.Tensor): Sampled intensity profiles with shape (B, suimg.Ny, suimg.Nx, Nch).
            itr (int): Number of iterations for the GS algorithm.
        Returns:
            tf.Tensor: Output phase hologram with shape (B, suslm.Ny, suslm.Nx, 1).
        """ 
        mask_slm = self.H_padmask
        mask_img = tf.ones_like(I_in)
        mask_img = self.imgpadding(mask_img)
        mask_img = tf.complex(1 - mask_img, 0.)
        
        I = I_in * tf.math.abs(self.S4)
        I = self.imgpadding(I)
        if self.slmpadf:
            I = self.up_slm(I)
        E = tf.complex(I**0.5,0.)
        
        E = uth.GS_imgpad(E, A_obj_keep = mask_img, A_obj_replace = E, A_holo = mask_slm, itr = itr, loc = 'holo')
        phi = tf.math.abs(mask_slm) * tf.math.angle(E)
        if self.slmpadf:
            phi = self.slmselect(phi)
        return phi
    #######################################################
    def encode_GS(self, I_in, itr, holotype = 'PH'):
        """
        Conventional method of designing OAM-multiplexed holograms using 
        the GS algorithm.
        Args:
            I_in (tf.Tensor): Sampled intensity profiles with shape (B, Ny, Nx, Nch).
            itr (int): Number of iterations for the GS algorithm.
            holotype (str): Hologram type ('PH': phase-only, else: complex amplitude).
        Returns:
            tf.Tensor: Output phase hologram with shape (B, suslm.Ny, suslm.Nx, 1).
        """
        E_img = tf.complex(I_in**.5, 0.)
        E_img = ut.up_pad(E_img,self.d)
        E_img = E_img * self.S4
        E_img = self.imgpadding(E_img)
        if self.slmpadf:
            E_img = self.up_slm(E_img)
        E_slm = uth.GS(E_img, A_obj = E_img, A_holo = self.H_padmask, loc = 'holo',itr = itr)
        if self.slmpadf:
            E_slm = self.slmselect(E_slm)
        E_slm *= ut.phase2cf(-self.lcode)
        E_slm = tf.reduce_sum(E_slm,-1,keepdims=True)
        if holotype == 'PH':
            E_slm = ut.unityamp(E_slm)
        E_slm = self.H_mask * E_slm
        return E_slm
    
    #########################################################
    def encode_E_sparse(self, E, holotype = 'PH', masked = True):
        """ Encodes the electric field E into a hologram using OAM modes.
        Args:
            E (tf.Tensor): Electric field with shape (B, Ny, Nx, Nch).
            holotype (str): Hologram type ('PH': phase-only, else: complex amplitude).
            masked (bool): Whether to apply the hologram mask.
        Returns:
            tf.Tensor: Output hologram with shape (B, suslm.Ny, suslm.Nx, 1).
        """
        E_img = ut.up_pad(E,self.d)
        E_img = self.imgpadding(E_img)
        if self.slmpadf:
            E_img = self.up_slm(E_img)
        E_slm = uth.backward_E(E_img)
        if self.slmpadf:
            E_slm = self.slmselect(E_slm)
            
        E_slm = E_slm * ut.phase2cf(-self.lcode)
        E_slm = tf.reduce_sum(E_slm, -1, keepdims = True)
        if holotype == 'PH':
            E_slm = ut.unityamp(E_slm)
        if masked:
            E_slm *= self.H_mask
        return E_slm
        
    def encode_NN_sparse(self, I_in, NNs, mode = 'mag', holotype = 'PH', masked = True):
        """
        Encodes the input intensity profiles I_in into a hologram using neural networks.
        The neural networks are applied to the input intensity profiles to 
        generate optimal auxiliary electric fields.
        Args:
            I_in (tf.Tensor): Sampled intensity profiles with shape (B, Ny, Nx, Nch).
            NNs (List[tf.keras.Model]): List of neural networks for each OAM channel.
            mode (str): Mode of the neural network ('mag' for magnitude, 'int' for intensity).
            holotype (str): Hologram type ('PH' for phase-only, else for complex amplitude).
            masked (bool): Whether to apply the hologram mask.
        Returns:
            tf.Tensor: Output hologram with shape (B, suslm.Ny, suslm.Nx, 1).
        """
        E_img = uth.apply_NN(I_in, NNs, mode, self.Nch)
        E_slm = self.encode_E_sparse(E_img, holotype = holotype, masked = masked)
        return E_slm

    #################################################################
    # Pads the electric field E at the SLM plane and propagates it to the image plane.
    def forward_E_pad(self, E_in):
        E = E_in
        if self.slmpadf:
            E = tf.pad(E,self.slmpad)
        E = uth.forward_E(E)
        return E
    # Pads and propagates the electric field E at the SLM plane, then down-samples and 
    # selects the field within the working region at the image plane.
    def forward(self, E_in):
        E = E_in
        if self.slmpadf:
            E = tf.pad(E, self.slmpad)
        E = uth.forward_E(E)
        if self.slmpadf:
            E = ut.down_pad2D(E, self.slmpadf + 1, mode = 'single')
        E = self.imgselect(E)
        return E

    def decode_E(self, E_mp, code = None):
        """
        Decodes the multiplexed electric field E_mp and returns demultiplexed fields in the 
        working region at the image plane assuming coherent measurement.
        Args:
            E_mp (tf.Tensor): Multiplexed electric field with shape (B, suslm.Ny, suslm.Nx, 1).
            code (tf.Tensor or None): OAM code for demultiplexing. If None, uses the default lcode.
        Returns:
            tf.Tensor: Demultiplexed electric fields with shape (B, suimg.Ny, suimg.Nx, Nch).
        """
        if code is None:
            E = E_mp * self.lcode_E
        else:
            E = E_mp * ut.phase2cf(code)
        E = self.forward(E)
        return E 

    def decode_E2I(self, E_mp, code = None, select = True, dsmode = 'single'):
        """
        Decodes the multiplexed electric field E_mp and returns demultiplexed intenisties in the 
        working region at the image plane with various incoherent measurement schemes.
        Args:
                E_mp (tf.Tensor): Multiplexed electric field with shape (B, suslm.Ny, suslm.Nx, 1).
                code (tf.Tensor or None): OAM code for demultiplexing. If None, uses the default lcode.
                select (bool): Whether to select the working region at the image plane.
                dsmode (str): Down-sampling mode correspnding to measurement scheme.
                'single' for single pixel measurement,
                'mean' for incoherent measurement with averaging,
                'conv' for incoherent measurement with convolution.
        Returns:
                tf.Tensor: Demultiplexed electric fields with shape (B, suimg.Ny, suimg.Nx, Nch).
        """
        if code is None:
            E = E_mp * self.lcode_E
        else:
            E = E_mp * ut.phase2cf(code)        
        E = E * ut.phase2cf(code)
        E = self.forward_E_pad(E)
        I = ut.measure_int(E)
        if self.slmpadf:
            I = ut.down_pad2D_general(I, self.slmpadf+1, mode = dsmode)
        if select:
            I = self.imgselect(I)
        return I
    
    def decode_E_propmodel(self, E_mp, propmodel, modulation, code = None):
        """
        Decodes the multiplexed electric field E_mp using a propagation model and modulation function.
        Args:
            E_mp (tf.Tensor): Multiplexed electric field with shape (B, suslm.Ny, suslm.Nx, 1).
            propmodel (PropagationModel): Propagation model function to apply to the modulated fields.
            modulation (Modulation): Modulation function to apply to the electric field.
            code (tf.Tensor or None): OAM code for demultiplexing. If None, uses the default lcode.
        Returns:
            tf.Tensor: Sampled demultiplexed electric fields with shape (B, Ny, Nx, Nch).
        """
        if code is None:
            code = self.lcode[None,...]

        phis_slm = modulation(E_mp, code)

        if not isinstance(phis_slm, list) and not isinstance(phis_slm, tuple):
            phis_slm = [phis_slm]
        
        phis_slm = [tf.transpose(phi, [3,1,2,0]) for phi in phis_slm]
        #phi_oamed_slm = tf.transpose(phi_oamed_slm,[3,1,2,0])
        E_img = propmodel(*phis_slm)
        E_img = tf.transpose(E_img,[3,1,2,0])
        return E_img 
    
    def field_to_img(self, E_in, sample = True):
        """
        Converts full electric field E_in to image intensity profile I.
        """
        E = E_in
        if sample:
            E = self.sample(E)
        I = ut.measure_int(E)
        return I
    def decode_img(self, E_mp, propmodel = None, modulation = None, code = None):
        """
        Decodes the multiplexed electric field E_mp and returns the image intensity profile I.
        Args:
            E_mp (tf.Tensor): Multiplexed electric field with shape (B, suslm.Ny, suslm.Nx, 1).
            propmodel (PropagationModel or None): Propagation model function to apply to the modulated fields.
            modulation (Modulation or None): Modulation function to apply to the electric field.
            code (tf.Tensor or None): OAM code for demultiplexing. If None, uses the default lcode.
        Returns:
            tf.Tensor: Image intensity profile with shape (B, Ny, Nx, Nch).
        """
        if propmodel:
            E = self.decode_E_propmodel(E_mp, propmodel, modulation, code = code)
            I = self.field_to_img(E, sample = False)
        else:
            E = self.decode_E(E_mp, code = code)
            I = self.field_to_img(E)
        return I
################################################################################
