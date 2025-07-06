import os
import tensorflow as tf
#######################
import lib.utils.utils as ut
import lib.proplib as proplib
import lib.CITL as CITL

def load(filepath):
    """
    Load a trained propagation model from a specified file path.
    """
    propmodel = tf.keras.models.load_model('filepath')
    return propmodel

def save(model, path, oam, experimenter):
    """
    Save the trained propagation model to a specified directory.
    """
    os.makedirs(path, exist_ok=True)

    save_path = os.path.join(path, f"Prop_L{oam.dl}N{oam.Nch} win{experimenter.win_size}")
    model.save(save_path)
    #tf.keras.models.save_model(propmodel, 'model',save_format="tf")


def train(setup, oam, experimenter, 
          dataset, lrs, itr_max, 
          save_dir = './prop_models', verbose = True):
    """
    Train a propagation model using the CITL training routine.
    Args:
        setup (str): "1SLM" or "2SLM" for the type of propagation model and holography setup.
        oam (OAM): An instance of the OAM class containing parameters for OAM multiplexing.
        experimenter (ExperimentClass): ExperimentClass object for performing experimental measurements.
        dataset (tf.data.Dataset): A TensorFlow dataset containing holograms for CITL training.
        lrs (list): List of learning rates for training.
        itr_max (int): Maximum number of iterations for training.
        save_dir (str): Directory to save the trained model.
        verbose (bool): If True, prints module summaries and training progress.
    Returns:
        propmodel (PropagationModel): The trained propagation model.
    """
    trainer = CITL.TrainingRoutine(experimenter, n_split = 1, 
                                avelen = 20, save_dir = '', save_freq = None)

    if setup == "1SLM":
        propmodel = proplib.PropagationModel(oam)
    elif setup == "2SLM":
        propmodel = proplib.PropagationModel_2SLM(oam)
    if verbose:
        ut.print_module_summary(propmodel)

    holoiter = iter(dataset)
    trainer.train(propmodel, holoiter, itr_max, lrs, draw= verbose)

    if save_dir:
        save(propmodel, save_dir, oam, experimenter)

    return propmodel

