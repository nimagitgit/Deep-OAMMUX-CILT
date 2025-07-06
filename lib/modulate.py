
import tensorflow as tf
import lib.utils.utils as ut
import lib.utils.slm as utslm
from numpy import pi

class Modulation_1SLM(tf.keras.Model):
    """
    Implements modulation for a single-SLM OAM-multiplexed holography system.
    This class computes the phase profiles to be displayed on 
    the phase-only SLM, for a given OAM-multiplexed hologram 
    and the decoding key. It provides phase wrapping 
    and conversion to gray-level holograms for physical SLM display.
    """
    def __init__(self, oam, **kwargs):
        super().__init__(**kwargs)
        self.oam = oam
        self.slm_mp = utslm.SLM(suslm = oam.suslm, mask= oam.H_mask_abs)

    def call(self, E_mp, code):
        """
        Compute the modulated wrapped phase for SLM input.
        Args:
            E_mp (tf.Tensor): Complex field of shape [B, H, W, 1].
            code (tf.Tensor): OAM phase code of shape [H, W, Nch].
        Returns:
            tf.Tensor: Wrapped phase profiles, shape [B, H, W, Nch].
        """
        phi_mp = ut.cf2phase(E_mp) 

        phi_oamed_slm = (phi_mp + code) * self.oam.H_mask_abs
        phi_oamed_slm = self.slm_mp.wrap(phi_oamed_slm, range = 2*pi)

        return phi_oamed_slm
    
    def decode(self, X, code):
        """
        decode and convert phase or complex field to SLM-ready gray-level hologram.
        Args:
            X (tf.Tensor): Either a phase pattern (tf.float32) or a complex field (tf.complex64).
            code (tf.Tensor): OAM decoding code (phase mask), shape [H, W, Nch].
        Returns:
            tf.Tensor: SLM gray-level image, shape [B, H, W, 1].
        """
        if X.dtype is tf.float32:
            phi = X
        elif X.dtype is tf.complex64:
            phi = ut.cf2phase(X)
        else:
            raise ValueError("phase (tf.float32) or field (tf.complex64)")
        phi = (phi + code) * self.oam.H_mask_abs
        holo = self.slm_mp.phase2gray(phi, 1.)
        return holo

class Modulation_2SLM(tf.keras.Model):#not completed still
    """
    Implements modulation for a two-SLM OAM-multiplexed holography system.
    This class computes the phase profiles to be displayed on 
    the phase-only SLMs, for a given OAM-multiplexed hologram 
    and the decoding key. The first SLM displays mutiplexed holograms; 
    the second SLM displays OAM keys
    This class provides phase wrapping and conversion to gray-level holograms for physical SLM display.
    """
    def __init__(self, oam, **kwargs):
        super().__init__(**kwargs)
        self.oam = oam
        self.slm_mp = utslm.SLM(suslm = oam.suslm, mask= oam.H_mask_abs)
        self.slm_oam = utslm.SLM(suslm = oam.suslm, mask= oam.H_mask_abs, mod= True)

    def call(self, E_mp, code):
        """
        Decode and converts multiplexed hologram and compute the modulated wrapped phase for SLM input.
        Args:
            E_mp (tf.Tensor): Complex field of shape [B, H, W, 1].
            code (tf.Tensor): OAM phase code of shape [H, W, Nch].
        Returns:
            tuple: Wrapped phase profiles for multiplexed hologram and OAM code 
            with shapes [B, H, W, 1] and [H, W, Nch] respectively.
        """

        phi_mp_slm = ut.cf2phase(E_mp) * self.oam.H_mask_abs
        phi_code_slm = code

        #phi_code_slm = self.slms_oam.wrap(code, range = 2*pi)
        #not complete add blazed grating interaction
        return phi_mp_slm, phi_code_slm
    
    def decode(self, X, code):
        """
        Decode and convert phase or complex field to SLM-ready gray-level hologram.
        Args:
            X (tf.Tensor): Either a phase pattern (tf.float32) or a complex field (tf.complex64).
            code (tf.Tensor): OAM decoding code (phase mask), shape [H, W, Nch].
        Returns:
            tuple: SLM gray-level images for multiplexed hologram and OAM code,
            shapes [B, H, W, 1] and [H, W, Nch] respectively.
        """

        if X.dtype is tf.float32:
            phi_mp = X * self.oam.H_mask_abs
        elif X.dtype is tf.complex64:
            phi_mp = ut.cf2phase(X) * self.oam.H_mask_abs
        else:
            raise ValueError("phase (tf.float32) or field (tf.complex64)")

        holo_code = self.slm_oam.phase2gray(code, self.oam.H_mask_abs) #not correct?
        holo_mp = self.slm_mp.phase2gray(phi_mp, self.oam.H_mask_abs)

        return holo_mp, holo_code  