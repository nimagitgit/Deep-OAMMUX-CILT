import tensorflow as tf

from keras.layers import LeakyReLU, Conv2D, BatchNormalization
from tensorflow_addons.layers import GroupNormalization

import numpy as np

import lib.utils.utils as ut
import lib.utils.holo as uth


class Target_process(tf.keras.layers.Layer):
    """
    Learnable vignetting layer that multiplies the decoded field at the image plane
    with a 2D scalar amplitude mask.
    """
    def __init__(self, size, **kwargs):
        super().__init__(**kwargs)
        self.size = size
        self.Amulti = tf.Variable(tf.ones([size[0], size[1]], dtype=tf.float32),name='Amulti') 
    def call(self, E_in):       
        E_out = (E_in) * tf.complex(self.Amulti[None, ...,None], 0.0)
        return E_out 
    
    
class Gauss_source(tf.keras.layers.Layer):
    """
    Superposition of Gaussian beams to approximate a learnable illumination source profile.
    """
    def __init__(self, suslm, n_gauss, init_sigma=(500.), init_amp=0.9, **kwargs):
        super().__init__(**kwargs)
        self.n_gauss = n_gauss
        self.init_sigma = init_sigma
        self.init_amp = init_amp
        self.center = tf.Variable(tf.zeros([n_gauss, 2], dtype=tf.float32), name='Center')
        self.std = tf.Variable([[sigma, sigma] for sigma in init_sigma], dtype=tf.float32, name='STD')       
        self.amp = tf.Variable(tf.ones([n_gauss], dtype=tf.float32)*init_amp/n_gauss, name='Amp')
        self.dc = tf.Variable([1.0*(1-init_amp)], dtype=tf.float32, name='DC')
        self.X, self.Y = suslm.X, suslm.Y
    def gauss(self, amp, cent, std):
        out = amp * tf.math.exp(-0.5 * (((self.X - cent[0]*1000) / std[0]/1000)**2 + ((self.Y - cent[1]*1000) / std[1]/1000)**2))
        return out
    def call(self, E_in):
        Asrc = self.dc
        for i in range(self.n_gauss):
            Asrc = Asrc + self.gauss(self.amp[i], self.center[i], self.std[i])
        E_out = E_in * tf.complex(Asrc[None, ..., None], 0.0)
        return E_out
class HD_source(tf.keras.layers.Layer):
    """
    High-definition (fully learnable) source profile defined as a pixel-wise 2D mask.
    """
    def __init__(self, suslm, **kwargs):
        super().__init__(**kwargs)
        self.Asrc = tf.Variable(tf.ones([suslm.Ny,suslm.Nx], dtype=tf.float32), name='Asrc')
    def call(self, E_in):
        E_out = E_in * tf.complex(self.Asrc[None, ..., None], 0.0)
        return E_out    
#MPL
class MLP(tf.keras.layers.Layer):
    """
    Phase correction model (CNN-based) with optional spatial latent codes.
    Approximates the nonlinear and spatially varying phase modulation of the SLM.
    """
    def __init__(self, in_shape, n_codes, n_layers=3, n_filters=32, fs = 1, norm= 'group', **kwargs):
        super().__init__(**kwargs)
        self.in_shape = in_shape
        self.norm = norm
        self.n_layers = n_layers
        self.fs = [fs]*n_layers if isinstance(fs, int) else fs
        self.n_filters = [n_filters]*n_layers if isinstance(n_filters, int) else n_filters 
        self.n_codes = [n_codes]*n_layers if isinstance(n_codes, int) else n_codes
        
        self.n_codes_tot = sum(n_codes)
        self.codes_idx = np.cumsum([0] + n_codes)
        self.codes = tf.Variable(tf.zeros([1,*in_shape,self.n_codes_tot], tf.float32)) 
        self.convs = [Conv2D(n_filters[i], (self.fs[i], self.fs[i]), activation=None, padding="same") for i in range(n_layers-1)]
        self.convs.append(Conv2D(1, (self.fs[-1], self.fs[-1]), activation=None, padding="same"))
        if norm == 'group':
            self.normfns = [GroupNormalization(groups=2) for _ in range(n_layers-1)]
        elif norm == 'batch':
            self.normfns = [BatchNormalization() for _ in range(n_layers-1)]   
            
    def call(self, phi_in):
        batch = tf.shape(phi_in)[0]
        x = phi_in
        for i in range(self.n_layers):
            if self.n_codes[i]:
                x = tf.concat([x,tf.tile(self.codes[...,self.codes_idx[i]:self.codes_idx[i+1]],[batch,1,1,1])],-1)
            x = self.convs[i](x)
            if i < self.n_layers - 1:
                if self.normfns[i] is not None:
                    x = self.normfns[i](x)
                x = LeakyReLU(0.2)(x)        
        phi_out = phi_in - x 
        return phi_out
    
    
class Zernike(tf.keras.layers.Layer):
    """
    Applies learnable Zernike polynomial phase aberrations.
    """
    def __init__(self, in_shape, n_zernike, **kwargs):
        super().__init__(**kwargs)
        self.in_shape = in_shape
        self.n_zernike = n_zernike
        self.zernike_coefs = tf.Variable(tf.zeros([self.n_zernike],tf.float32),name='zcoef')
        #coef_init = -tf.constant([0,0,0,-.35,.4,-1.39,-1.47,.94,-1.1,-.52,0,0,0,0,0],tf.float32)
        #self.zernike_coefs = tf.Variable(coef_init,name='zcoef')
        self.zernike_coefs = tf.Variable(tf.zeros([self.n_zernike],tf.float32),name='zcoef')
        self.zernike_basis = uth.compute_zernike_basis(n_zernike, field_size=1024, aperture='circ')
        
    def call(self, phases):
        phi_zernike = tf.reduce_sum(self.zernike_basis * self.zernike_coefs[None, None, :], -1)
        phase_out = phases + phi_zernike[None, ..., None]
        return phase_out
    
    
class PropagationModel(tf.keras.Model):
    """
    Parameterized propagation model for CITL calibration in a 1-SLM OAM holography system.
    Includes SLM model (MLP), Zernike aberrations, source modeling, and vignetting.
    """
    def __init__(self, oam, n_gauss = 3, n_zernike = 15,
                 n_layers_mpl = 3, n_codes_mpl = [1,0,0], n_filters_mpl = [16,16,16], fs_mpl = [3,1,1], **kwargs):
        super().__init__(**kwargs)
        self.oam = oam #OAM.from_config(oam1.get_config())
        self.n_zernike = n_zernike
        self.n_layers_mpl = n_layers_mpl
        self.n_codes_mpl = n_codes_mpl
        self.n_filters_mpl = n_filters_mpl
        self.fs_mpl = fs_mpl
        d = self.oam.d
        Ny_slm, Nx_slm = oam.suslm.Ny, oam.suslm.Nx
        Ny_img, Nx_img = oam.suimg.Ny, oam.suimg.Nx
        slm_shape = [Ny_slm,Nx_slm]
        img_raw_shape = [Ny_img,Nx_img]
        img_shape = [Ny_img//d,Nx_img//d]
        self.mlpNN = MLP(slm_shape, n_codes = n_codes_mpl, n_layers = n_layers_mpl, n_filters = n_filters_mpl, fs = fs_mpl, norm= 'group')
        self.zernike_mp = Zernike(1024,n_zernike)
        if n_gauss == -1:
            self.source = HD_source(oam.suslm)
        else:
            self.source = Gauss_source(oam.suslm, n_gauss = n_gauss, init_sigma = [.300,.500,.700], init_amp = 0.9)
        self.targetprocess = Target_process(img_shape)    
    def call(self, phi_in):
        phi_slm = self.mlpNN(phi_in)
        phi_slm = self.zernike_mp(phi_slm)
        E_slm = ut.phase2cf(phi_slm)*self.oam.H_mask
        E_slm = self.source(E_slm)
        E_img_raw = self.oam.forward(E_slm)
        E_img = self.oam.sample(E_img_raw)
        E_img = self.targetprocess(E_img)
        E_img_out = E_img
        return E_img_out

        
class PropagationModel_2SLM(tf.keras.Model):
    """
    Extended parameterized propagation model for CITL calibration and 
    for two SLMs: one for image modulation and one for OAM multiplexing.
    Supports distinct MLP and Zernike models for each SLM.
    """
    def __init__(self, oam, n_gauss = 3, n_zernike = 25,
                 n_layers_mpl = 3, n_codes_mpl = [1,0,0], 
                 n_filters_mpl = [16,16,16], fs_mpl = [3,1,1], **kwargs):
        super().__init__(**kwargs)
        self.oam = oam #OAM.from_config(oam1.get_config())
        #self.modulation = modulation
        self.n_zernike = n_zernike
        self.n_layers_mpl = n_layers_mpl
        self.n_codes_mpl = n_codes_mpl
        self.n_filters_mpl = n_filters_mpl
        self.fs_mpl = fs_mpl
        d = self.oam.d
        Ny_slm, Nx_slm = oam.suslm.Ny, oam.suslm.Nx
        Ny_img, Nx_img = oam.suimg.Ny, oam.suimg.Nx
        slm_shape = [Ny_slm,Nx_slm]
        img_raw_shape = [Ny_img,Nx_img]
        img_shape = [Ny_img//d,Nx_img//d]
        self.mlpNN_mp = MLP(slm_shape, n_codes = n_codes_mpl, n_layers = n_layers_mpl, n_filters = n_filters_mpl, fs = fs_mpl, norm= 'group')
        self.mlpNN_oam = MLP(slm_shape, n_codes = [1,0,0], n_layers = n_layers_mpl, n_filters = [8,8,8], fs = [5,3,1], norm= 'group')

        self.zernike_mp = Zernike(1024,n_zernike)
        self.zernike_oam = Zernike(1024,n_zernike)
        
        if n_gauss == -1:
            self.source = HD_source(oam.suslm)
        else:
            self.source = Gauss_source(oam.suslm, n_gauss = n_gauss, init_sigma = [.300,.500,.700], init_amp = 0.9)
        self.targetprocess = Target_process(img_shape)    
        #self.targetprocess_oam = Target_process(slm_shape, add = True)    

    def call(self, phi_mp, phi_oam):
        phi_slm_oam = phi_oam #+ self.phi_z_oamslm
        phi_slm_oam = self.zernike_oam(phi_slm_oam)
        phi_slm_oam = self.mlpNN_oam(phi_slm_oam)
        
        E_slm_oam = ut.phase2cf(phi_slm_oam)
        #E_oam = self.modulatation.unmod_oam(E_slm_oam)
        E_oam = E_slm_oam
        E_slm_oam = self.source(E_oam)
        #E_slm_oam = self.targetprocess_oam(E_slm_oam)
        
        phi_slm_mp = self.mlpNN_mp(phi_mp)
        phi_slm_mp = self.zernike_mp(phi_slm_mp)
        E_slm_mp = ut.phase2cf(phi_slm_mp)
        E_slm = E_slm_mp * E_slm_oam *self.oam.H_mask
        
        E_img_raw = self.oam.forward(E_slm)  
        E_img = self.oam.sample(E_img_raw)
        E_img = self.targetprocess(E_img)
        E_img_out = E_img
        return E_img_out
                