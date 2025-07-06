import os
import tensorflow as tf
import numpy as np
import random
import shutil
from tqdm import tqdm

# Dataset generator for synthetic images of circles and lines

class CircleGenerator:
    """
    Generates synthetic images with random circles.
    The circles can vary in radius, angle, aspect ratio, position, and intensity.
    """
    def __init__(self, dims, params):
        """
        Initializes the CircleGenerator with the given parameters.
        Args:
            dims (tuple): dimensions of the output image (Ny, Nx)
            params (dict): parameters for circle generation
            - 'N': number of circles
            - 'N_del': variability in the number of circles
            - 'radius': average radius of circles
            - 'radius_del': variability in the radius of circles
            - 'angle_max': maximum angle for circle rotation
            - 'aspect_max': maximum aspect ratio for circles
            - 'value_min': minimum intensity value for circles
        """
        self.Ny, self.Nx = dims
        self.N = params['N']
        self.N_del = params['N_del']
        self.radius = params['radius']
        self.radius_del = params['radius_del']
        
        self.angle_max = params['angle_max']
        self.aspect_max = params['aspect_max']
        self.value_min = params['value_min']
        self.eps = 1e-6

        x = tf.range(self.Nx, dtype = tf.float64)
        y = tf.range(self.Ny, dtype = tf.float64)
        self.X, self.Y = tf.meshgrid(x, y)

    @staticmethod
    def cart2pol(a, b):
        rho = tf.math.sqrt(a**2 + b**2)
        phi = tf.math.atan2(b, a)
        return rho, phi

    def generate(self):
        """
        Generates a synthetic image with random circles according to the specified parameters.
        Returns:
            out (tf.Tensor): a tensor of shape (Ny, Nx) containing the generated circles
        """
        out = tf.zeros((self.Ny, self.Nx), dtype=tf.float32)
        N_circ = int(self.N + random.randint(-self.N_del//2, self.N_del//2))
        radiuses = self.radius + np.random.uniform(-self.radius_del/2, self.radius_del/2, size = N_circ)
        angles = np.random.uniform(-self.angle_max, self.angle_max, size = N_circ)
        aspects = np.random.uniform(1, self.aspect_max, size = N_circ)
        centers_x = np.random.uniform(0, self.Nx, size = N_circ)
        centers_y = np.random.uniform(0, self.Ny, size = N_circ)
        intensities = np.random.uniform(self.value_min, 1, size = N_circ)
        
        for i in range(N_circ):
            X = self.X - centers_x[i]
            Y = self.Y - centers_y[i]

            Xrot = (X * tf.math.cos(angles[i]) - Y * tf.math.sin(angles[i]))/aspects[i]
            Yrot = X * tf.math.sin(angles[i]) + Y * tf.math.cos(angles[i])
            R, _ = self.cart2pol(Xrot,Yrot)
            circs = tf.cast(R < radiuses[i], tf.float32) * intensities[i]
            out = tf.maximum(out, circs)
        return out

class LineGenerator:
    """
    Generates synthetic images with random lines.
    The lines can vary in thickness, angle, position, and intensity.
    """
    def __init__(self, dims, params):
        """
        Initializes the LineGenerator with the given parameters.
        Args:
            dims (tuple): dimensions of the output image (Ny, Nx)
            params (dict): parameters for line generation
            - 'N': number of lines
            - 'N_del': variability in the number of lines
            - 'thickness': thickness of the lines
            - 'value_min': minimum intensity value for lines
        """
        self.Ny, self.Nx = dims
        self.N = params['N']
        self.N_del = params['N_del']
        
        self.thickness = params['thickness']
        self.value_min = params['value_min']

        x = tf.range(self.Nx, dtype = tf.float64)
        y = tf.range(self.Ny, dtype = tf.float64)
        self.X, self.Y = tf.meshgrid(x, y)

    @staticmethod
    def cart2pol(a, b):
        rho = tf.math.sqrt(a**2 + b**2)
        phi = tf.math.atan2(b, a)
        return rho, phi

    def generate(self):
        """
        Generates a synthetic image with random lines according to the specified parameters.
        Returns:
            out (tf.Tensor): a tensor of shape (Ny, Nx) containing the generated lines
        """
        out = tf.zeros((self.Ny, self.Nx), dtype=tf.float32)
        N_line = int(self.N + random.randint(-self.N_del//2, self.N_del//2))
        angles = np.random.uniform(0, np.pi, size = N_line)
        centers_x = np.random.uniform(0, self.Nx, size = N_line)
        centers_y = np.random.uniform(0, self.Ny, size = N_line)
        intensities = np.random.uniform(self.value_min, 1, size = N_line)
        
        for i in range(N_line):
            X = self.X - centers_x[i]
            Y = self.Y - centers_y[i]

            Xrot = X * tf.math.cos(angles[i]) - Y * tf.math.sin(angles[i])
            Yrot = X * tf.math.sin(angles[i]) + Y * tf.math.cos(angles[i])
            lines = tf.logical_and(Yrot >= 0, Yrot < self.thickness)
            lines = tf.cast(lines, tf.float32) * intensities[i]
            out = tf.maximum(out, lines)
        return out

class DatasetGenerator:
    """
    Generates a dataset of synthetic images containing random circles and lines.
    The dataset can be saved to disk or returned as a TensorFlow dataset.
    """
    def __init__(self, size, params_circ = None, params_line = None):
        """
        Initializes the DatasetGenerator with the specified image size and parameters for circles and lines.
        Args:
            size (tuple): dimensions of the output images (Ny, Nx)
            params_circ (dict, optional): parameters for circle generation
            params_line (dict, optional): parameters for line generation
        """
        self.size = size
        self.Ny, self.Nx = size
        if params_circ:
            self.params_circ = params_circ    
        else:
            self.params_circ = {
                'N':30, 'N_del':5, 'radius':10, 'radius_del':4,
                'angle_max': np.pi, 'aspect_max':5, 'value_min':0.5
                }
        if params_line:
            self.params_line = params_line
        else:
            self.params_line = {
                'N':30, 'N_del':5, 'thickness':2,
                'angle_max': np.pi, 'value_min':0.5
                }
        self.circle_gen = CircleGenerator([self.Ny, self.Nx], self.params_circ)
        self.line_gen = LineGenerator([self.Ny, self.Nx], self.params_line)

    def randmix_gen(self, weight):
        """
        Generates a random mix of circles and lines based on the specified weight.
        """
        while True:
            generator = random.choices(
            [self.circle_gen.generate, self.line_gen.generate], 
            weights=[weight,1-weight])[0]
            yield generator()
    def dataset_create_save(self, path , N, weight):
        """
        Creates a dataset of synthetic images and saves them to the specified path.
        Args:
            path (str): directory where the dataset will be saved
            N (int): number of images to generate
            weight (float): weight for the circle generation in the random mix
        """
        try:
            shutil.rmtree(path)
        except Exception as e:
            pass
        os.makedirs(path,exist_ok=True)
        dataset = tf.data.Dataset.from_generator(lambda : self.randmix_gen(weight), tf.float32)
        iterator = iter(dataset)
        for i in tqdm(range(N)):
            sample = next(iterator)
            fname = os.path.join(path, f'{i:03}.bin')
            with open(fname, 'wb') as f:
                sample.numpy().tofile(f)
    def dataset_from_generator(self,weight, pair = False):
        """
        Creates a TensorFlow dataset from the random mix generator.
        Args:
            weight (float): weight for the circle generation in the random mix
            pair (bool): if True, returns pairs of images (input, target)
        Returns:
            dataset (tf.data.Dataset): a TensorFlow dataset containing the generated images
        """
        dataset = tf.data.Dataset.from_generator(lambda : self.randmix_gen(weight), output_shapes= self.size, output_types= tf.float32)
        print(dataset)
        if pair:
            dataset = dataset.map(lambda x: (x,x))
        return dataset