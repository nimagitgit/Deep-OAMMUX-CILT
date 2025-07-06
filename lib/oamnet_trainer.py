import os
import tensorflow as tf
from tensorflow import keras

from keras.layers import Input
from keras.models import Model
#######################
import numpy as np
import cv2
import json
#import time
#######################
from math import pi
from tqdm.notebook import tqdm
#######################
import lib.utils.utils as ut
import lib.utils.slm as utslm
import lib.oamlib as ol
import lib.loss as loss
import lib.NNmodel as NNmodel

import lib.proplib as proplib

from time import time
import lib.CITL as CITL

def create_oamnet(oam: ol.OAM, propmodel= None, modulation= None, 
                  vignetting= False, zernike= True, net= None):
    """
    Construct a autoencoder including a U-net model as encoder and optical simulation as decoder.
    The autoencoder is used for unsupervised training of the U-net for desining 
    OAM multiplexed holograms. If parameterized propagation model (propmodel) is provided, 
    the U-net is initialized to account for vignetting and Zernike phase aberrations.
    Args:
        oam (ol.OAM): OAM object containing OAM multiplexing parameters.
        propmodel (proplib.PropModel, optional): Propagation model learned via CITL.
        modulation (Modulation, optional): Modulation scheme used in holography.
        vignetting (bool, optional): Whether to account for vignetting in the model.
        zernike (bool, optional): Whether to account for Zernike phase aberrations.
        net (keras.Model, optional): Predefined neural network model. 
                                    If None, a new U-net is created.
    Returns:
        keras.Model: Autoencoder model that maps input sampled intensities to reconstructed output.
    """
    I_multi = 1.
    phi_zernike = 0.
    if propmodel:
        if vignetting:
            I_multi = propmodel.targetprocess.Amulti[None,...,None] ** 2
            #ut.draw(I_multi, size= 2)
        if zernike:
            phi = tf.zeros([1, oam.suslm.Ny, oam.suslm.Nx, 1], tf.float32)
            phi_zernike = propmodel.zernike_mp(phi)
            #ut.draw(phi_zernike, size= 2)
    
    if net is None:
        net = NNmodel.__unet((oam.Ny, oam.Nx), num_blocks= 4, num_filter= 32,
                             Nin= oam.Nch, Nout= oam.Nch, extra_ups= 0, num_resb= 0,
                             norm= True, fs_out= 3, out_type= 'complex')
    I_in = Input(shape= (oam.Ny, oam.Nx, oam.Nch))
    I = I_in / I_multi
    E_mp = oam.encode_NN_sparse(I, [net], 'mag', masked= False)
    E_mp /= ut.phase2cf(phi_zernike)

    I_out = oam.decode_img(E_mp, propmodel= propmodel, modulation= modulation)

    autoencoder = Model(I_in,I_out)
    return autoencoder

def save(net, path, name, oam):
    """
    Saves the trained U-Net model and its metadata.
    Args:
        net (keras.Model): Trained neural network model.
        path (str): Directory to save the model.
        name (str): Base filename for the model.
        oam (ol.OAM): OAM object for metadata.
    """
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, f"{name}_dl{oam.dl}_N{oam.Nch}")
    net.save(save_path)

    metadata = {"dl": oam.dl, "Nch": oam.Nch, "d": oam.d}
    with open(save_path + "_meta.json", "w") as f:
        json.dump(metadata, f, indent=4)

def train(net, dataset, oam, propmodel, modulation, 
          schedule = [(1, 0.1)],
          save_dir = './', name = 'NNprop',
          verbose = False):
    """
    Trains the U-Net using an optical autoencoder (OAMcoder) where:
    - The encoder is the neural network (U-Net).
    - The decoder is the OAM encoding and optical propagation pipeline.

    Args:
        net (keras.Model): U-Net model to be trained.
        dataset (tf.data.Dataset): Training dataset containing sampled target images.
        oam (ol.OAM): OAM object to encode/decode images using optical methods.
        propmodel (proplib.PropModel): Propagation model for decoding.
        modulation (function): SLM modulation model used in the decoding step.
        schedule (List[Tuple[int, float]]): Training schedule [(epochs, learning_rate)].
        save_dir (str): Directory to save the trained model.
        name (str): Base name for saving model files.
        verbose (bool): If True, print model summary.
    """
    # Freeze propagation model layers (learned via CITL)
    for layer in propmodel.layers:
        layer.trainable = False
        
    # Create optical autoencoder system for unsupervised training
    OAMcoder = create_oamnet(oam, propmodel, modulation, False, True, net= net)
    
    if verbose:
        OAMcoder.summary()  

    # Compile with custom correlation loss
    opt = tf.keras.optimizers.Adam(learning_rate= .1)
    OAMcoder.compile(optimizer= opt, loss= loss.CorrlossSep())

    for round, lr in schedule:
        opt.learning_rate = lr
        OAMcoder_history = OAMcoder.fit(dataset, epochs = round)

    if save_dir:
        save(net, save_dir, name, oam)
