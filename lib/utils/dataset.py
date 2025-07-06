import tensorflow as tf
#from tensorflow.data import AUTOTUNE
import numpy as np
import glob
import os
from . import utils as ut
from .dataset_gen import DatasetGenerator

# Module for creating TensorFlow datasets from binary files or local directories.
# It also contrains a function to create a dataset from a generator of synthetic images (circles and lines).


def binarydataset(num, patharr, n_items = 1, types = None, suffle = True, AT = 'auto', batch = 1):
    """
    Creates a TensorFlow dataset from binary files in the specified paths.
    Args:
        num (int): number of samples to take from the dataset
        patharr (list): list of paths where binary files are located
        n_items (int): number of items in each binary file
        types (list): list of data types for each item, if None, defaults to [tf.float32]*n_items
        suffle (bool): whether to shuffle the dataset
        AT (str or int): number of parallel calls for mapping, 'auto' uses tf.data.AUTOTUNE
        batch (int): batch size for the dataset, if None, no batching is applied
    Returns:
        ds (tf.data.Dataset): a TensorFlow dataset containing the binary data
    """

    #tds = binarydataset(10,['./oam_DS/'], n_items = 2, types = [tf.uint8,tf.float32], suffle = False, batch = None)
    AUTOTUNE = tf.data.AUTOTUNE

    typearr = [tf.float32]*n_items if types is None else types
    if AT == 'auto':
        AT = AUTOTUNE
    all_files = []
    for path in patharr:
        bin_files = glob.glob(path + '*.npy')
        all_files.extend(bin_files)
    ds = tf.data.Dataset.list_files(all_files,shuffle=suffle)
    ds = ds.repeat()
    ds = ds.take(num)
    ds = ds.map(map_func= lambda x : tf.numpy_function(ut.load_npy, [x,n_items], typearr),num_parallel_calls=AT)
    if batch:
        ds = ds.batch(batch)
    ds = ds.prefetch(buffer_size=AT)
    return ds

def process_dataset(dataset, Nch, num, batch = 1, suffle = False, augment = True, pair = True, AT = tf.data.AUTOTUNE):
    """
    Processes a TensorFlow dataset by batching, shuffling, augmenting, and optionally pairing the data.
    Args:
        dataset (tf.data.Dataset): input dataset to be processed
        Nch (int): number of channels
        num (int): number of samples to take from the dataset
        batch (int): batch size for the dataset, if None, no batching is applied
        suffle (bool): whether to shuffle the dataset
        augment (bool): whether to apply data augmentation
        pair (bool): if True, returns pairs of images (input, target)
        AT (str or int): number of parallel calls for mapping, 'auto' uses tf.data.AUTOTUNE
    Returns:
        ds (tf.data.Dataset): a processed TensorFlow dataset
    """
    ds = dataset
    ds = ds.batch(Nch, drop_remainder=True)
    ds = ds.map(map_func= lambda x: tf.squeeze(tf.transpose(x,[3,1,2,0]),0))
    if augment:
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.2),
            ])
        ds = ds.map(lambda x: data_augmentation(x, training= True),num_parallel_calls=AT)  #check training= True if it is correct?
    if pair:
        ds = ds.map(lambda x: (x,x))
    ds = ds.repeat()
    ds = ds.take(num)
    
    ds = ds.batch(batch)
    ds = ds.prefetch(buffer_size=AT)
    return ds

def dataset_fromfiles(paths, exts, Nch, num, insize, outsize=None, batch = 1, 
                  suffle = True, augment = True, pair = True, AT = tf.data.AUTOTUNE):
    """
    Creates a TensorFlow dataset from files in the specified paths with given extensions.
    Args:
        paths (list): list of paths where files are located
        exts (list): list of file extensions to look for
        Nch (int): number of channels
        num (int): number of samples to take from the dataset
        insize (list): input size of the images [height, width]
        outsize (list or None): output size of the images [height, width], if None, no resizing is applied
        batch (int): batch size for the dataset, if None, no batching is applied
        suffle (bool): whether to shuffle the dataset
        augment (bool): whether to apply data augmentation
        pair (bool): if True, returns pairs of images (input, target)
        AT (str or int): number of parallel calls for mapping, 'auto' uses tf.data.AUTOTUNE
    Returns:
        ds (tf.data.Dataset): a TensorFlow dataset containing the images
    """
    
    all_files = []
    for path in paths:
        for ext in exts:
            files = glob.glob(os.path.join(path, '*.' + ext))
            all_files.extend(files)  
    ds = tf.data.Dataset.list_files(all_files, shuffle = suffle)
    
    ds = ds.map(map_func = lambda x : ut.load_any(x, insize, outsize)[...,0:1], num_parallel_calls = AT)

    ds = process_dataset(dataset = ds, Nch = Nch, num = num, batch = batch, suffle = False, 
                         augment = augment, pair = pair, AT = AT)

    return ds

def dataset_fromlocals(size, nums, Nch):
    """
    Creates training and validation datasets from local directories.
    Args:
        size (list): size of the images [height, width]
        nums (list): number of training and validation samples [num_train, num_valid]
        Nch (int): number of channels 
    Returns:
        train_ds (tf.data.Dataset): training dataset
        valid_ds (tf.data.Dataset): validation dataset
    """
    # Example paths, adjust according to your local setup
    trainpathbin = 'C:/Users/nima_/DS/DIV2K_train_HR_bin/'
    validpath = 'C:/Users/nima_/My Drive/DS/valid/'
    circpath = 'C:/Users/nima_/DS/newcirc_(40,30)_240x900/'

    saved_size = [900,240,1]
    [num_train,num_valid] = nums
    train_ds = dataset_fromfiles([trainpathbin,circpath], ['bin','png'], Nch, num_train,
                             insize = saved_size, outsize = size, batch = 1, augment = True)
    valid_ds = dataset_fromfiles([validpath], ['bin','png'], Nch, num_valid,
                             insize = size, outsize = None, batch = 1, suffle = False, augment = False)
    return train_ds, valid_ds

def dataset_fromlocals_test(size, num, Nch):
    """
    Creates a test dataset from local directories.
    """
    testpath = 'C:/Users/nima_/My Drive/DS/DIV2K_valid_HR/'
    test_ds = dataset_fromfiles(paths = [testpath], exts = ['png'], num = num, Nch = Nch, 
                                insize = size, outsize = None, suffle = False, augment = False, batch = 1)
    return test_ds

def create_circline_ds(path, size, num, weight):
    """
    Creates a dataset of synthetic images (circles and lines) and saves it to the specified path.
    Args:
        path (str): directory where the dataset will be saved
        size (list): size of the images [height, width]
        num (int): number of images to generate
        weight (float): weight for the circle generation in the random mix
    """
    ds_generator = DatasetGenerator(size)
    ds_generator.dataset_create_save(path = path, N = num, weight = weight)


def dataset_circline_fromgen(size, num, Nch =1, weight = 0.5, batch = 1, suffle = True, augment = True, 
                                 pair = True, AT = tf.data.AUTOTUNE):
    """
    Creates a TensorFlow dataset from a generator of synthetic images (circles and lines).
    Args:
        size (list): size of the images [height, width]
        num (int): number of samples to take from the dataset
        Nch (int): number of channels
        weight (float): weight for the circle generation in the random mix
        batch (int): batch size for the dataset, if None, no batching is applied
        suffle (bool): whether to shuffle the dataset
        augment (bool): whether to apply data augmentation
        pair (bool): if True, returns pairs of images (input, target)
        AT (str or int): number of parallel calls for mapping, 'auto' uses tf.data.AUTOTUNE
    Returns:
        ds (tf.data.Dataset): a TensorFlow dataset containing the generated images
    """
    ds_generator = DatasetGenerator(size)
    ds = ds_generator.dataset_from_generator(weight = weight)
    ds = ds.map(lambda x: x[:,:,None])
    ds = process_dataset(dataset = ds, num = num, Nch = Nch, batch = batch, suffle = True,
                         augment = augment, pair = pair, AT = AT)
    return ds



########################
def customdataset(patharr, Nch, num, insize=None, outsize=None, Batch = 1, 
                  sf = True, aug = True, AT = 'auto', pair = True):
    data_augmentation = tf.keras.Sequential([
      tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
    ])
    if AT == 'auto':
        AT = tf.data.experimental.AUTOTUNE

    all_files = []
    for path in patharr:
        png_files = glob.glob(path + '*.png')
        bin_files = glob.glob(path + '*.bin')
        all_files.extend(png_files + bin_files)  
    dsarr = [tf.data.Dataset.list_files(all_files,shuffle=sf) for path in patharr]
    ds = dsarr[0]
    for d in dsarr[1:]:
        ds.concatenate(d)
    
    
    ds = ds.map(map_func= lambda x : ut.load_any(x,insize,outsize)[...,0:1],num_parallel_calls=AT)
    ds = ds.batch(Nch,drop_remainder=True)
    ds = ds.map(map_func= lambda x: tf.squeeze(tf.transpose(x,[3,1,2,0]),0))
    if aug:
        ds = ds.map(lambda x: data_augmentation(x, training=True),num_parallel_calls=AT)
    if pair:
        ds = ds.map(lambda x: (x,x))
    ds = ds.repeat()
    ds = ds.take(num)
    ds = ds.batch(Batch)
    ds = ds.prefetch(buffer_size=AT)
    return ds

#########
def binarydataset(num, patharr, n_items = 1, types = None, sf = True, AT = 'auto', Batch = 1):
    #tds = binarydataset(10,['./oam_DS/'], n_items = 2, types = [tf.uint8,tf.float32], sf = False, Batch = None)

    typearr = [tf.float32]*n_items if types is None else types
    if AT == 'auto':
        AT = tf.data.AUTOTUNE
    all_files = []
    for path in patharr:
        bin_files = glob.glob(path + '*.npy')
        all_files.extend(bin_files)
    ds = tf.data.Dataset.list_files(all_files,shuffle=sf)
    ds = ds.repeat()
    ds = ds.take(num)
    ds = ds.map(map_func= lambda x : tf.numpy_function(load_npy, [x,n_items], typearr),num_parallel_calls=AT)
    if Batch:
        ds = ds.batch(Batch)
    ds = ds.prefetch(buffer_size=AT)
    return ds

def load_npy(path, n_items):
    out = [] 
    with open(path, 'rb') as f:
        for i in range(n_items):
            out.append(np.load(f))
    return out
