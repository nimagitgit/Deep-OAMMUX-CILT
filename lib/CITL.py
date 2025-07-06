
import numpy as np
from time import time
from tqdm.notebook import tqdm
import os

import tensorflow as tf
from keras.optimizers import Adam
from keras.losses import MeanSquaredError

import lib.utils.utils as ut
from lib.camera_pp import CameraProcess
import lib.oamlib as ol
import lib.loss as loss
import lib.solvers as solvers

class ExperimentClass:
    """
    This class performs experimental measurements during Camera-in-the-loop (CITL) training.
    It captures images from the camera after displaying a hologram on the SLM, processes the captured images.
    """

    def __init__(self, oam, device_manager, modulation, holos_grid, dh, prom, win_size = 3, slmpause = 300):
        """
        Initializes the ExperimentClass with the given parameters.
        Args:
            oam (ol.OAM): OAM object containing OAM multiplexing parameters.
            device_manager (DeviceManager): Device manager for handling SLM and camera.
            modulation (Modulation): Modulation scheme used in holography.
            holos_grid (np.ndarray): Grid hologram for camera calibration.
            dh (float): height parameter for grid detection used in scipy.signal.find_peaks
            prom (float): prominence parameter for grid detection used in scipy.signal.find_peaks
            win_size (int): Size of the window for extracting sparse samples.
            slmpause (int): Pause duration after displaying the SLM hologram.
        """

        self.oam = oam
        self.Nx = oam.Nx
        self.Ny = oam.Ny
        self.dh = dh
        self.prom = prom
        self.device_manager = device_manager
        self.holos_grid = holos_grid
        self.slmpause = slmpause
        self.win_size = win_size
        self.slm_mp = modulation.slm_mp
        self.modulation = modulation

    def run(self, holo, grid, slmpause = None, win_size = None, norm = False, mean = True):
        """
        Runs the experiment by displaying a hologram on the SLM and capturing images from the camera.
        Args:
            holo (np.ndarray): Hologram to be displayed on the SLM.
            grid (np.ndarray): Grid hologram for camera calibration.
            slmpause (int, optional): Pause duration after displaying the SLM hologram.
            win_size (int, optional): Size of the window for extracting sparse samples.
            norm (bool, optional): Whether to normalize the extracted intensities based on measured grid's intensities.
            mean (bool, optional): Whether to average intensities over the window.
        Returns:
            tuple: Extracted sparse (sampled) intensities and raw (full) captured images.
        """
        pause = slmpause if slmpause else self.slmpause
        win = win_size if win_size else self.win_size
        #I_raw = self.device_manager.show_capture(holo, slmpause = pause) #OLD version single SLM
        #I_grid = self.device_manager.show_capture(grid, slmpause = pause) #OLD version single SLM
        
        I_raw = self.device_manager.show_capture_multi(holo, slmpause = pause) 
        I_grid = self.device_manager.show_capture_multi(grid, slmpause = pause)

        camproc = CameraProcess(I_grid[0,:,:,0], self.Nx, self.Ny)
        camproc.findgrid_raster(dh=self.dh, prom=self.prom)
        #camproc.findgrid_conv()

        if not (camproc.pos.shape[1] == self.Nx):
            print('\033[94m' + 'wrong grid' + '\033[0m')
            print(camproc.pos.shape)
            #raise exception("Wrong Grid")
            return None
        
        I_sparse = camproc.extract(I_raw[0,...], win_size = win, norm = norm, mean = mean)
        
        print(f'measured and processed : {camproc.dx:.2f}, {camproc.dy:.2f}, {camproc.pos.shape}')
        return I_sparse, I_raw

    def run_single_slm(self, holos, slmpause = None, win_size = None, norm = False, mean = True):
        """
        Runs the experiment by displaying a sequence of decoded holograms on the single SLM and capturing images from the camera.
        Args:
            holos (np.ndarray): Sequence of holograms to be displayed on the SLM including a grid hologram for calibration.
            slmpause (int, optional): Pause duration after displaying the SLM hologram.
            win_size (int, optional): Size of the window for extracting sparse samples.
            norm (bool, optional): Whether to normalize the extracted intensities based on measured grid's intensities.
            mean (bool, optional): Whether to average intensities over the window.
        Returns:
            tuple: Extracted sparse (sampled) intensities and raw (full) captured images.
        """
        pause = slmpause if slmpause else self.slmpause
        win = win_size if win_size else self.win_size
        I_raw = self.device_manager.show_capture(holos, slmpause = pause)

        camproc = CameraProcess(I_raw[0, :, :, 0], self.Nx, self.Ny)
        camproc.findgrid_raster(dh=self.dh, prom=self.prom)
        #camproc.findgrid_conv()

        if not (camproc.pos.shape[1] == self.Nx):
            print('\033[94m' + 'wrong grid' + '\033[0m')
            print(camproc.pos.shape)
            #except "Wrong Grid"
            return None
        
        I_sparse = camproc.extract(I_raw[0,...], win_size = win, norm = norm, mean = mean)
        
        print(f'measured and processed : {camproc.dx:.2f}, {camproc.dy:.2f}, {camproc.pos.shape}')
        return I_sparse, I_raw
         
        
    def decode_exp(self, X_mp, code, win_size = None):
        """
        Decodes and measures images from a multiplexed hologram thorugh experimental measurement by a .
        Args:
            X_mp (np.ndarray): Multiplexed hologram (phase or complex amplitude field).
            code (np.ndarray): OAM codes used for decoding.
            win_size (int, optional): Size of the window for extracting sparse samples.
        Returns:
            tuple: Extracted sparse (sampled) intensities and raw (full) captured images
        """
        holo_code = self.modulation.decode(X_mp, code)

        I_sparse, I_raw = self.run(holo_code, self.holos_grid, win_size = win_size)
        I_sparse = tf.convert_to_tensor(I_sparse[None,...],tf.float32)
        I_raw = tf.convert_to_tensor(I_raw[None,...],tf.float32)
        
        return I_sparse, I_raw
        
    def sim_exp_save(self, I_true, sim_fun, fname, fullsave = False, win_size = 3, verbose = False):
        """
        Multiplexes the true images into a hologram, simulates/reconstructs decoded images, and 
        performs experimental measurements. Finally, it saves all results to a file.
        Args:
            I_true (np.ndarray): True images to be multiplexed.
            sim_fun (function): Multiplexing and simulation function that takes true images as input 
                                and returns OAM-multiplexed hologran and simulated decoded images.
            fname (str): Filename to save the results.
            fullsave (bool, optional): Whether to save the raw experimental images.
            win_size (int, optional): Size of the window for extracting sparse samples.
            verbose (bool, optional): Whether to print detailed information about the simulation and experimental results.
        """
        start = time()
        E_mp, I_sim, err_sim, tim = sim_fun(I_true)
        code = self.oam.lcode[None,...]
        timer = time()-start
        I_exp, I_exp_raw = self.decode_exp(E_mp, code = code, win_size = win_size)
        err_exp = loss.CorrlossSep()(I_exp,I_true).numpy()
        
        if verbose:
            print(f"sim:{err_sim*100:.1f}\texp:{err_exp*100:.1f}\ttime:{timer:.3f}")
            ut.draw(I_sim)
            ut.draw(I_exp)

        save_dict = {"I_true": I_true, "I_sim": I_sim, "I_exp": I_exp, 
                     "err_sim": err_sim, "err_exp": err_exp, 
                     "t": tim}
        if fullsave:
            save_dict.update({"I_exp_raw": I_exp_raw})
        np.savez(fname, **save_dict)

        return
    

class TrainingRoutine:
    """
    This class implements a training routine for learning a parameterized propagation model
    using Camera-in-the-loop (CITL) measurements.
    It optimizes the model parameters to minimize the difference between measured and simulated intensities.
    """
    def __init__(self,experiment, n_split = 1, 
                 avelen = 20, save_dir = '', save_freq = None):
        """
        Initializes the TrainingRoutine with the given parameters.
        Args:
            experiment (ExperimentClass): ExperimentClass object for performing experimental measurements.
            n_split (int): Number of splits along the channel dimention for processing training data.
            avelen (int): Length of the averaging window for calculating PSNR as a metric.
            save_dir (str): Directory to save the trained model and results.
            save_freq (int, optional): Frequency of saving the model during training.
        """
        self.opt = Adam(learning_rate=0.01)
        self.loss_fn = MeanSquaredError()

        self.experiment = experiment
        self.oam = experiment.oam
        self.slm_mp = experiment.slm_mp
        self.modulation = experiment.modulation
        self.n_split = n_split
        self.avelen = avelen
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.itr = 0
        self.psnr_list = []
        self.counter = 0

    def adjust_learning_rate(self, lr, counter = None):
        """
        Adjusts the learning rate of the optimizer based on the current iteration count.
        """
        if isinstance(lr, (float, int)):
            self.opt.learning_rate = lr
        elif counter in lr:
            self.opt.learning_rate = lr[counter]
            
    def process_batch(self, phis, A_exp, model):
        """
        Processes a batch of phase holograms and measured intensities to compute the error
        between the measured and simulated intensities, and applies the gradients to the model.
        Args:
            phis (list or tuple): List or tuple of phase holograms to be processed.
            A_exp (tf.Tensor): Measured intensities from the experiment.
            model (tf.keras.Model): Model to be trained.
        Returns:
            tuple: Error between measured and simulated intensities, and simulated intensities.
        """
        with tf.GradientTape() as tape:
            E_img_out = model(*phis)        
            A_sim = ut.measure_amp(E_img_out)
            A_sim = tf.transpose(A_sim,[3,1,2,0])
            err = self.loss_fn(A_exp, A_sim)
        gradients = tape.gradient(err, model.trainable_variables)
        self.opt.apply_gradients(zip(gradients, model.trainable_variables))
        return err, A_sim
    
    def metric(self,A_exp,err):
        """
        Computes the PSNR (Peak Signal-to-Noise Ratio) metric between the measured and simulated intensities.
        Args:
            A_exp (tf.Tensor): Measured intensities from the experiment.
            err (tf.Tensor): Error between measured and simulated intensities.
        Returns:
            float: Average PSNR value over the last 'avelen' iterations.
        """
        psnr = ut.PSNR(A_exp,err)
        self.psnr_list.append(psnr)
        return np.mean(self.psnr_list[-self.avelen:])
    
    def train(self, propmodel, holoiter, maxitr, lr, draw= False):
        """
        Trains the propagation model using Camera-in-the-loop (CITL) measurements.
        Args:
            propmodel (tf.keras.Model): Propagation model to be trained.
            holoiter (iterable): Iterable object that yields multiplexed holograms and true images.
            maxitr (int): Maximum number of iterations for training.
            lr (float or dict): Learning rate for the optimizer, can be a constant or a dictionary with iteration-specific values.
            draw (bool): Whether to visualize the results during training.
        """
        counter = 0
        code = self.oam.lcode[None,...]
        while counter < maxitr:
            print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")

            self.adjust_learning_rate(lr,counter)
            itertime = time()
            holo_mp, I_true = next(holoiter)

            phi_mp = self.slm_mp.get_phi(holo_mp)
            I_exp, _ = self.experiment.decode_exp(phi_mp, code)
            A_exp = I_exp ** 0.5

            phis = self.modulation(phi_mp, code)
            phis = phis if isinstance(phis, list) or isinstance(phis, tuple) else [phis]

            phis_parts = []
            for phi in phis:
                phi = tf.transpose(phi,[3,1,2,0]) 
                if phi.shape[0] == 1:
                    phi = tf.tile(phi, [self.n_split,1,1,1])    
                phi_parts = tf.split(phi, num_or_size_splits= self.n_split, axis=0)
                phis_parts.append(phi_parts)
            
            A_exp_parts = tf.split(A_exp, num_or_size_splits= self.n_split, axis=3)

            err_parts = []
            for *phis_part, A_exp_part in zip(*phis_parts, A_exp_parts):
                err, A_sim = self.process_batch(phis_part, A_exp_part, propmodel)
                err_parts.append(err.numpy())
            err = np.mean(err_parts)

            #print(f'iter: {counter} \t sim calcualted  t={time()-itertime:.2f}s')       
            if draw or counter%40 == 0 :
                print("I_true:")
                ut.draw(I_true[...,:], size= 6, frame= None)
                print("Measured intensity:")
                ut.draw(A_exp[...,0:]**2, size= 6, frame= None)
                print("Predicted intensity:")
                ut.draw(A_sim[...,0:]**2, size= 6, frame= None)
                #print("Vignetting:")
                #ut.draw(propmodel.targetprocess.Amulti, size=2, frame = 'all')
            
            maxI = tf.reduce_max(I_exp)
            psnr_ave = self.metric(A_exp,err)
            ut.print_color(f'err= {err:.3f}\tpsnr = {psnr_ave:.3f}\titer:{counter}\tImax={maxI:.0f}', 'RED')

            #if counter % save_every_n == 1:
            #    save_path = os.path.join(save_dir, f"model_itr={counter}_err={psnr_ave:.4f}")   
                #print(f'done\tt={time()-itertime:.2f}s')
            counter += 1
      

def load_or_create_samples(Nch, dl, image_size, name, data_path, sample_path = "./Itrue", regen = False):
    """
    Loads or creates samples of true images for training.
    Args:
        Nch (int): Number of channels.
        dl (int): Downsampling factor.
        image_size (tuple): Size of the images (height, width).
        name (str): Name of the sample set.
        data_path (str): Path to the directory containing the data.
        sample_path (str, optional): Path to save or load the samples. Defaults to "./Itrue".
        regen (bool, optional): Whether to regenerate the samples if they already exist. Defaults to False. 
    Returns:
        np.ndarray: Array of true images.
    """
    sample_dir = os.path.join(sample_path, f"L{dl}_N{Nch}")
    os.makedirs(sample_dir, exist_ok=True)

    sample_file = os.path.join(sample_dir, "I_true_" +  name + ".npz")

    if os.path.isfile(sample_file) and regen is False:
        print(f"Loading samples:\n{sample_file}")

        samples = np.load(sample_file)
        I_true = samples["I_true"]
        
    else:
        print(f"Creating samples:\n{sample_file}")

        sample_ds = ut.customdataset([data_path], Nch, 5, 
                                     insize = image_size, outsize=None,
                                     sf=True, aug=False, Batch=1, pair = False)

        I_true =  list(sample_ds)
        np.savez(sample_file, I_true = I_true)

    return I_true


def measure_samples(Is, oam, path, experimenter: ExperimentClass, fun, win):
    """
    Designs OAM-multiplexed hologram. Then it obtains decoded/reconstructed images 
    by simulating the hologram and performing experimental measurements and saves the results.
    Args:
        Is (np.ndarray): Array of true images to be measured.
        oam (ol.OAM): OAM object containing OAM multiplexing parameters.
        path (str): Path to save the results.
        experimenter (ExperimentClass): ExperimentClass object for performing experimental measurements.
        fun (function): Designs OAM-multiplexed hologram and simulates/reconstructs decoded images.
        win (int): Size of the window for extracting sparse samples.
    """    
    os.makedirs(path, exist_ok=True)
    for i, I_true_raw  in enumerate(tqdm(Is)):
        I_true = oam.sample(I_true_raw)
        name = os.path.join(path, f'test_{i}.npz')
        experimenter.sim_exp_save(I_true, fun, name, win_size= win, verbose= True)
        
def measure_gd(Is, oam, path, experimenter, win, propmodel= None, modulation= None, itr= 100, lr= 1e3):
    """
    Designs OAM-multiplexed hologram using Gradient Descent (GD) solver. Then it obtains decoded/reconstructed images
    by simulating the hologram and performing experimental measurements and saves the results.
    Args:
        Is (np.ndarray): Array of true images to be measured.
        oam (ol.OAM): OAM object containing OAM multiplexing parameters.
        path (str): Path to save the results.
        experimenter (ExperimentClass): ExperimentClass object for performing experimental measurements.
        win (int): Size of the window for extracting sparse samples.
        propmodel (tf.keras.Model, optional): Propagation model learned via CITL. Defaults to None.
        modulation (Modulation, optional): Modulation scheme used in holography. Defaults to None.
        itr (int, optional): Number of iterations for GD solver. Defaults to 100.
        lr (float, optional): Learning rate for GD solver. Defaults to 1e3.
    """
    GD_fun = lambda I: solvers.GD_solver(I, oam, itr= itr, lr= lr, propmodel= propmodel, modulation= modulation)
    measure_samples(Is, oam, path, experimenter, GD_fun, win)

def measure_nn(Is, oam, path, net, experimenter, win, propmodel= None, modulation= None):
    """
    Designs OAM-multiplexed hologram using Neural Network (NN) solver. Then it obtains decoded/reconstructed images
    by simulating the hologram and performing experimental measurements and saves the results.
    Args:
        Is (np.ndarray): Array of true images to be measured.
        oam (ol.OAM): OAM object containing OAM multiplexing parameters.
        path (str): Path to save the results.
        net (tf.keras.Model): Neural network model for solving the hologram design.
        experimenter (ExperimentClass): ExperimentClass object for performing experimental measurements.
        win (int): Size of the window for extracting sparse samples.
        propmodel (tf.keras.Model, optional): Propagation model learned via CITL. Defaults to None.
        modulation (Modulation, optional): Modulation scheme used in holography. Defaults to None.
    """
    NN_fun = lambda I: solvers.NN_solver(I, oam, net, propmodel, modulation)
    measure_samples(Is, oam, path, experimenter, NN_fun, win)

def measure_gs(Is, oam, path, experimenter, win, propmodel= None, itr= 200):
    """
    Designs OAM-multiplexed hologram using Conventional method using the GS algorithm. Then it obtains decoded/reconstructed images
    by simulating the hologram and performing experimental measurements and saves the results.
    Args:
        Is (np.ndarray): Array of true images to be measured.
        oam (ol.OAM): OAM object containing OAM multiplexing parameters.
        path (str): Path to save the results.
        experimenter (ExperimentClass): ExperimentClass object for performing experimental measurements.
        win (int): Size of the window for extracting sparse samples.
        propmodel (tf.keras.Model, optional): Propagation model learned via CITL. Defaults to None.
        itr (int, optional): Number of iterations for the GS method. Defaults to 200.
    """
    GS_fun = lambda I: solvers.GS_solver(I, oam, propmodel, itr)
    measure_samples(Is, oam, path, experimenter, GS_fun, win)