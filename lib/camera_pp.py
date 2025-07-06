#This library provides functions to process camera images, grid detection (various methods), and data extraction.

import numpy as np
import scipy as sp
from numpy import unravel_index
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import find_peaks
from scipy.optimize import minimize
import lib.utils.utils as ut
import cv2



def loadfiles(filepaths, blur= False, blursize= 4):
    """
    Loads images from specified file paths and applies optional Gaussian blur.
    Args:
        filepaths (list): List of file paths to the images.
        blur (bool): If True, applies Gaussian blur to the images.
        blursize (int): Size of the Gaussian kernel to be used for blurring.
    Returns:
        np.ndarray: A 3D numpy array containing the loaded images, with shape (H, W, C).
    """
    imgs = []
    for i, filepath in enumerate(filepaths):
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if blur:
            img = cv2.GaussianBlur(img, (blursize, blursize), 0)
        imgs.append(img)
    imgs = np.stack(imgs,-1)
    return imgs


class CameraProcess:
    """
    A class to process camera images for grid detection and extraction.
    It includes methods for preprocessing images, finding grid patterns, and extracting sparse data.
    """
    def __init__(self,image,Nx,Ny):
        """
        Initializes the CameraProcess with an grid image and grid dimensions.
        Args:
            image (np.ndarray): The input image to process.
            Nx (int): Number of grid points in the x-direction.
            Ny (int): Number of grid points in the y-direction.
        """
        if len(image.shape) == 3:
            self.img = image[:,:,0]
        else:
            self.img = image
        self.img = np.array(self.img,np.float64)/255.
        
        self.img = self.preproc(self.img)

        self.int = np.ones_like(self.img)
        
        self.Nimg_x = self.img.shape[1]
        self.Nimg_y = self.img.shape[0]
        self.Ng_x = Nx
        self.Ng_y = Ny
        self.corners = [[]]
        self.positions = [[]]

        
    def preproc(self,image):
        """
        Preprocesses the input image by applying a threshold to create a binary image.
        Args:
            image (np.ndarray): The input image to preprocess.
        Returns:
            np.ndarray: The preprocessed binary image.
        """
        thresh_value = .05
        #image = np.flip(image,0)
        #image = np.flip(image,1)
        #_,self.img = cv2.threshold(self.img ,thresh_value, 1., cv2.THRESH_BINARY)

        return image
    
    def findgrid_raster(self, dh = 0.2, prom = .2):
        """
        Finds the grid pattern in the image using a raster method.
        This method uses the Fourier Transform to identify the grid spacing and then applies peak detection
        to find the grid lines in both horizontal and vertical directions.
        Args:
            dh (float): The minimum height of peaks to be considered significant.
            prom (float): The minimum prominence of peaks to be considered significant.
        """
        #Spacing
        fft_img = np.fft.fft2(self.img)
        fft_img = fft_img[:self.Nimg_y//2,:self.Nimg_x//2,]
        fft_img[:20,:] = fft_img[:,:20] = 0.
        freqy, freqx = unravel_index(fft_img.argmax(), fft_img.shape)
        dx = self.Nimg_x / freqx
        dy = self.Nimg_y / freqy
        
        horz = np.sum(self.img,0)
        horz /= horz.max() 
        vert = np.sum(self.img,1)
        vert /= vert.max() 
        
        cx, hx = find_peaks(horz, height= dh, distance=dx * 0.6, prominence= prom)
        cy, hy = find_peaks(vert, height= dh, distance=dx * 0.6, prominence= prom)

        hx = hx['peak_heights']
        hy = hy['peak_heights']
        
        Hx, Hy = np.meshgrid(hx,hy)
        
        self.int = np.sqrt(Hx * Hy)
        
        self.dx = (cx[-1] - cx[0])/(self.Ng_x-1)
        self.dy = (cy[-1] - cy[0])/(self.Ng_y-1)

        self.topleft_x = cx[0]
        self.topleft_y = cy[0]
        
        self.botright_x = cx[-1]
        self.botright_y = cy[-1]
        
        grid_x, grid_y = np.meshgrid(cx,cy)
        
        self.pos = np.stack([grid_x,grid_y],-1)

    def gaussian_filter(self,size=3, sigma=1.0):
        """
        Creates a Gaussian filter of specified size and standard deviation.
        Args:
            size (int): The size of the filter (must be odd).
            sigma (float): The standard deviation of the Gaussian distribution.
        Returns:
            np.ndarray: The Gaussian filter.
        """
        filter_range = np.linspace(-(size//2), size//2, size)
        x, y = np.meshgrid(filter_range, filter_range)
        gaussian = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
        gaussian /= np.sum(gaussian)
        return gaussian       
        
    def findgrid_conv(self, fsize = 3):
        """
        Finds the grid pattern in the image using a convolution method.
        This method applies a Gaussian filter to the image and uses convolution to find the grid points.
        Args:
            fsize (int): The size of the Gaussian filter to be used for convolution.
        """

        hdx = int(self.dx/2)
        hdy = int(self.dy/2)
        df = fsize // 2
        filter_ = self.gaussian_filter(size= fsize, sigma=1.0)
        
        pos = np.zeros([self.Ng_y,self.Ng_x,2])    
        self.int = np.zeros([self.Ng_y,self.Ng_x])    

        for i in range(self.Ng_y):
            for j in range(self.Ng_x):
                x = self.pos[i,j,0]
                y = self.pos[i,j,1]

                window = self.img[y-hdy:y+hdy,x-hdx:x+hdx]
                conv = cv2.filter2D(window, -1, filter_,anchor = [0,0])
                _, _, _, max_loc = cv2.minMaxLoc(conv)

                pos[i,j,:] = [x-hdx + max_loc[0]+df,y-hdy + max_loc[1] + df]
                self.int[i,j] = np.sum(window)
                
        self.pos = pos      
        
    def intmap(self,w = None, blur = None):
        """
        Computes the intensity map of the grid points in the image.
        Args:
            w (int): The width of the window to compute the intensity. If None, uses the grid spacing.
            blur (bool): If True, applies Gaussian blur to the image before computing intensity.
        Returns:
            np.ndarray: The intensity map of the grid points.
        """
        if not w:
            hdx = int(np.round(self.dx/2))
            hdy = int(np.round(self.dy/2))
            wx = int(self.dx)
            wy = int(self.dy)
        else:
            hdx = w//2
            hdy = w//2
            wx = wy = w   
        intensity = np.zeros([self.Ng_y,self.Ng_x])  
        img = self.img
        if blur:
            img = cv2.GaussianBlur(img, (201, 201), 0)
        for i in range(self.Ng_y):
            for j in range(self.Ng_x):
                x = self.pos[i,j,0]
                y = self.pos[i,j,1]

                window = img[y-hdy:y-hdy+wy,x-hdx:x-hdx+wx]
                intensity[i,j] = np.sum(window)
                
        return intensity
                
    def findgrid_minmse(self):
        """
        Finds the grid pattern in the image using a minimization approach.
        This method optimizes the grid parameters by minimizing the mean squared error between the detected grid points
        and the expected grid positions.
        """
        def model(params, coords, ind_grid):
            row_vector = np.array(params[:2])
            col_vector = np.array(params[2:4])
            start_point = np.array(params[4:])

            residuals = []
            grid = ind_grid[:,:,0:1] * row_vector[None,None,:] + ind_grid[:,:,1:] * col_vector[None,None,:]
            grid += start_point
            
            diff = coords - grid
            error = np.sum(diff[:,:,0]**2 + diff[:,:,1]**2)

            return error        
    

        
        indx_x, indx_y = np.meshgrid(np.arange(self.Ng_x),np.arange(self.Ng_y))
        
        indx_grid = np.stack([indx_x,indx_y],-1)
        coords = self.pos
        #coords = coords[~np.isnan(coords).any(axis=1)]
        initial_guess = [float(self.dx), 0.0, 0.0, float(self.dy), float(self.topleft_x), float(self.topleft_y)]
        gridfit = minimize(model, initial_guess, args=(coords, indx_grid))
        
        optimized_params = gridfit.x
        self.row_vector = np.array(optimized_params[:2])
        self.col_vector = np.array(optimized_params[2:4])
        self.start_point = np.array(optimized_params[4:])
    

        self.pos = np.zeros((self.Ng_y, self.Ng_x, 2),dtype = np.float64)
        self.pos = indx_grid[:,:,0:1] * self.row_vector[None,None,:] + indx_grid[:,:,1:] * self.col_vector[None,None,:]
        self.pos += self.start_point

        
    def findgrid_manual(self,off_row,off_col,off_start):
        """
        Finds the grid pattern in the image using manual offsets for row, column, and starting point.
        This method allows for manual adjustment of the grid parameters based on user-defined offsets.
        Args:
            off_row (np.ndarray): Offset for the row vector.
            off_col (np.ndarray): Offset for the column vector.
            off_start (np.ndarray): Offset for the starting point of the grid.
        """

        indx_x, indx_y = np.meshgrid(np.arange(self.Ng_x),np.arange(self.Ng_y))
        indx_grid = np.stack([indx_x,indx_y],-1)
        
        self.pos = np.zeros((self.Ng_y, self.Ng_x, 2))
        self.pos = indx_grid[:,:,0:1] * (self.row_vector[None,None,:]+off_row) 
        self.pos += indx_grid[:,:,1:] * (self.col_vector[None,None,:]+off_col)
        self.pos += self.start_point + off_start

        
    def extract(self,image, win_size = 3, norm = False, mean = False):
        """
        Extracts a grid of sparse data from the image based on the detected grid positions.
        This method extracts patches of the image centered around the grid points and returns them in a structured format.
        Args:
            image (np.ndarray): The input image from which to extract data.
            win_size (int): The size of the window to extract around each grid point.
            norm (bool): If True, normalizes the extracted patches by their intensity.
            mean (bool): If True, returns the mean of the extracted patches instead of the patches themselves.
        Returns:
            np.ndarray: The extracted grid of patches or their means, reshaped to fit the grid structure.
        """

        image = self.preproc(image)#*self.img       
        def extract_cell(img, center, win_size, mean = False):

            half_size = win_size // 2
            start_x = int(np.round(center[0])) - half_size
            start_y = int(np.round(center[1])) - half_size
            end_x = start_x + win_size
            end_y = start_y + win_size
            img = img[start_y:end_y, start_x:end_x,:]
            if mean:
                return np.mean(img,axis=(0,1), keepdims = True)
            else:
                return img
            
        Nimg = image.shape[-1]
        if mean:
            out_size = 1
        else:
            out_size = win_size
        ext = np.zeros([self.Ng_y,self.Ng_x,out_size,out_size,Nimg])

        for i in range(self.Ng_y):
            for j in range(self.Ng_x):
                cell = extract_cell(image, self.pos[i, j], win_size, mean)
                if norm:
                    ext[i,j,...] = cell/self.int[i,j]
                else:
                    ext[i,j,...] = cell
        ext = ext.swapaxes(1, 2).reshape(self.Ng_y*out_size,self.Ng_x*out_size,Nimg)

        return ext
    def showcorners(self):
        """
        Displays the detected grid corners on the image for manual grid adjustment.
        """
        w = 300
        TL = self.pos[0,0]
        BR = self.pos[-1,-1]

        fig, ax = plt.subplots(figsize=(10,10))
        img_plot = plt.imshow(self.img, cmap='gray', interpolation=None, aspect=1)
        plt.scatter(self.pos[:,:,0],self.pos[:,:,1],s = 1)
        plt.xlim([TL[0] - 10,TL[0] + w])
        plt.ylim([TL[1] - 10,TL[1] + w])
        plt.show()
        
        fig, ax = plt.subplots(figsize=(10,10))
        img_plot = plt.imshow(self.img, cmap='gray', interpolation=None, aspect=1)
        plt.scatter(self.pos[:,:,0],self.pos[:,:,1],s = 1)
        plt.xlim([BR[0] - w,BR[0] + 10])
        plt.ylim([TL[1] - 10,TL[1] + w])
        plt.show()
        
        fig, ax = plt.subplots(figsize=(10,10))
        img_plot = plt.imshow(self.img, cmap='gray', interpolation=None, aspect=1)
        plt.scatter(self.pos[:,:,0],self.pos[:,:,1],s = 1)
        plt.xlim([BR[0] - w,BR[0] + 10])
        plt.ylim([BR[1] - w,BR[1] + 10])
        plt.show()
        
        
        