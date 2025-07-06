from time import time
from tqdm.notebook import tqdm

import tensorflow as tf
import lib.loss as loss
import lib.utils.utils as ut

# Solvers for designing OAM-multiplexed phase holograms


def GD_solver(I_true, oam, itr= 100, lr= 1e3, opt= 'SGD', propmodel= None, modulation= None):
    """
    Gradient descent solver for OAM-multiplexed phase holograms.
    Args:
        I_true: Target intensity image.
        oam: OAM object containing the necessary parameters.
        itr: Number of iterations for optimization.
        lr: Learning rate for the optimizer.
        opt: Optimizer type ('SGD' or 'adam').
        propmodel: Propagation model (optional).
        modulation: Modulation type (optional).
    Returns:
        E_mp: Encoded complex field.
        I_sim: Simulated intensity image.
        err: Final error value.
        terr: List of error values over iterations.
    """
    loss_fn = loss.CorrlossSep()
    if opt == 'SGD':
        opt = tf.keras.optimizers.SGD(learning_rate=lr)
    elif opt == 'adam':  
        opt = tf.keras.optimizers.Adam(learning_rate=lr)

    E_img = tf.complex(tf.random.uniform([1,oam.Ny, oam.Nx, oam.Nch]),0.)
    E_img = tf.Variable(E_img)
    start = time()
    pbar = tqdm(range(itr))
    terr = []
    for i in pbar:
        with tf.GradientTape(persistent=False) as tape:
            E_mp = oam.encode_E_sparse(E_img, masked = False)
            I_sim = oam.decode_img(E_mp, propmodel, modulation)
            err = loss_fn(I_true, I_sim)
            gradients = tape.gradient(err, [E_img])
            opt.apply_gradients(zip(gradients, [E_img]))
        terr.append([time()-start,err.numpy()])
        pbar.set_postfix(err=f'{err*100:.1f}%')
    return E_mp, I_sim, err,terr


def NN_solver(I_true, oam, net, propmodel= None, modulation= None, vignetting= False, zernike= True):
    """
    Neural network solver for OAM-multiplexed phase holograms.
    Args:
        I_true: Target intensity image.
        oam: OAM object containing the necessary parameters.
        net: Neural network model for encoding.
        propmodel: Propagation model (optional).
        modulation: Modulation type (optional).
        vignetting: Whether to apply vignetting correction.
        zernike: Whether to apply Zernike polynomial correction.
    Returns:
        E_mp: Encoded complex field.
        I_sim: Simulated intensity image.
        err: Final error value.
        tim: Time taken for the simulation.
    """
    I_multi = 1.
    phi_zernike = 0.
    if propmodel:
        if vignetting:
            I_multi = propmodel.targetprocess.Amulti[None,...,None] ** 2
        if zernike:
            phi = tf.zeros([1, oam.suslm.Ny, oam.suslm.Nx, 1], tf.float32)
            phi_zernike = propmodel.zernike_mp(phi)
            
    I = I_true / I_multi
    start = time()
    E_mp = oam.encode_NN_sparse(I, [net], 'mag', masked= False)
    E_mp /= ut.phase2cf(phi_zernike)
    tim = time() - start
    I_sim = oam.decode_img(E_mp, propmodel= propmodel, modulation= modulation)
    
    err = loss.CorrlossSep()(I_sim,I_true).numpy()
    return E_mp, I_sim, err, tim

def GS_solver(I_true, oam, propmodel= None, itr= 200):
    """
    Conventional Phase Hologram (Conv. PH) solver for OAM-multiplexed phase holograms.
    Args:
        I_true: Target intensity image.
        oam: OAM object containing the necessary parameters.
        propmodel: Propagation model (optional).
        itr: Number of iterations for optimization.
    Returns:
        E_mp_z: Encoded complex field with Zernike correction.
        I_sim: Simulated intensity image.
        err: Final error value.
        tim: Time taken for the simulation.
    """
    I_multi = 1.
    phi_zernike = 0.
    if propmodel:
        I_multi = propmodel.targetprocess.Amulti[None,...,None] ** 2
        phi = tf.zeros([1, oam.suslm.Ny, oam.suslm.Nx, 1], tf.float32)
        phi_zernike = propmodel.zernike_mp(phi)
            
    loss_fn = loss.CorrlossSep()
    start = time()
    I = I_true / I_multi
    E_mp = oam.encode_GS(I, itr= itr)
    E_mp_z = E_mp / ut.phase2cf(phi_zernike)
    tim = time() - start
    I_sim = oam.decode_img(E_mp)   
    I_sim *= I_multi 
    err = loss_fn(I_true, I_sim)
    
    return E_mp_z, I_sim, err, tim