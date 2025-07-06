import tensorflow as tf
from tensorflow import keras
from keras.losses import MeanSquaredError, MeanAbsoluteError
from keras import backend
from skimage.metrics import structural_similarity as ssim
###################################################
class Corrloss(tf.keras.losses.Loss):
    """
    Cross correlation loss function equal to one minus 
    normalized cross correlation between two tensors
    """
    def __init__(self):
        super(Corrloss, self).__init__()

    def call(self, y_true, y_pred):
        """
        Computes the normalized cross correlation loss between two tensors.
        Args:
            y_true (tf.Tensor): ground truth tensor [batch, W, H, C]
            y_pred (tf.Tensor): predicted tensor [batch, W, H, C]
        Returns:
            loss (tf.Tensor): normalized cross correlation loss
        """
        a = tf.reduce_sum(y_true*y_pred,[1,2,3])
        b = tf.reduce_sum(y_true**2,[1,2,3])
        c = tf.reduce_sum(y_pred**2,[1,2,3])
        loss = tf.reduce_mean(1-(a)/(tf.sqrt(b*c)))
        return loss

###################################################
class CorrlossSep(tf.keras.losses.Loss):
    """
    Cross correlation loss function equal to one minus
    normalized cross correlation between two tensors, averaged over channels.
    """
    def __init__(self):
        super(CorrlossSep, self).__init__()

    def call(self, y_true, y_pred):
        """
        Computes the normalized cross correlation loss averaged over the last dimension (channel).
        Args:
            y_true (tf.Tensor): ground truth tensor [batch, W, H, C]
            y_pred (tf.Tensor): predicted tensor [batch, W, H, C]
        Returns:
            loss (tf.Tensor): normalized cross correlation loss averaged over channels
        """
        eps =0.
        a = tf.reduce_sum(y_true*y_pred,[1,2])
        b = tf.reduce_sum(y_true**2,[1,2])
        c = tf.reduce_sum(y_pred**2,[1,2])
        loss = tf.reduce_mean(1-(a)/(eps + tf.sqrt(b*c)))
        return loss
###################################################
class msemae(tf.keras.losses.Loss):
    """
    Combined Mean Squared Error (MSE) and Mean Absolute Error (MAE) loss function.
    """
    def __init__(self, factor=0.1):
        super(msemae, self).__init__()
        self.factor = factor
        self.mse = tf.keras.losses.MeanSquaredError()
        self.mae = tf.keras.losses.MeanAbsoluteError()
    def call(self, y_true, y_pred):
        """
        Computes the combined MSE and MAE loss.
        Args:
            y_true (tf.Tensor): ground truth tensor [batch, W, H, C]
            y_pred (tf.Tensor): predicted tensor [batch, W, H, C]
        Return:
            loss (tf.Tensor): combined MSE and MAE loss
        """
        loss = self.mse(y_true, y_pred) + self.factor*self.mae(y_true, y_pred)
        return loss
###################################################
def rabs(y_t, y_p):
    """
    Relative Absolute Error (RAE) loss function in percentage.
    Args:
        y_t (tf.Tensor): ground truth tensor [batch, W, H, C]
        y_p (tf.Tensor): predicted tensor [batch, W, H, C]
    Returns:
        loss (tf.Tensor): relative absolute error in percentage
    """
    return backend.mean(backend.abs(y_p - y_t), axis=-1)/(
        tf.keras.backend.max(y_t) - tf.keras.backend.min(y_t))*100
      
def log10(x):
    """
    Computes the base-10 logarithm of a tensor.
    Args:
        x (tf.Tensor): input tensor
    Returns:
        log10 (tf.Tensor): base-10 logarithm of the input tensor
    """
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def PSNR(x,y):
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) between two tensors.
    Args:
        x (tf.Tensor): ground truth tensor [batch, W, H, C]
        y (tf.Tensor): predicted tensor [batch, W, H, C]
    Returns:
        err (tf.Tensor): PSNR value in decibels (dB)
    """
    mse = tf.keras.losses.MeanSquaredError()
    xn = x/tf.reduce_max(x)
    yn = y/tf.reduce_max(y)
    err = 10*log10(1/mse(xn,yn))
    return err
###################################################
def PSNRSep(x, y):
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) between two tensors, averaged over channels.
    Args:
        x (tf.Tensor): ground truth tensor [batch, W, H, C]
        y (tf.Tensor): predicted tensor [batch, W, H, C]
    Returns:
        avg_psnr (tf.Tensor): average PSNR value across channels in decibels (dB)
    """
    mse = tf.keras.losses.MeanSquaredError()
    xn = x / tf.reduce_max(x, axis=[1, 2, 3], keepdims=True)
    yn = y / tf.reduce_max(y, axis=[1, 2, 3], keepdims=True)
    psnr_list = []
    for channel in range(x.shape[-1]):
        mse_val = mse(xn[..., channel], yn[..., channel])
        psnr = 10 * log10(1 / mse_val)
        psnr_list.append(psnr)
    avg_psnr = tf.reduce_mean(psnr_list)
    return avg_psnr

##########################################################
def SSIM(x,y):
    """
    Computes the Structural Similarity Index (SSIM) between two tensors with values normalized to [0, 1].
    Args:
        x (tf.Tensor): ground truth tensor [W, H]
        y (tf.Tensor): predicted tensor [W, H]
    Returns:
        err (float): SSIM value between the two tensors
    """
    xn = x / tf.reduce_max(x)
    yn = y / tf.reduce_max(y)
    err = ssim(xn.numpy(),yn.numpy(),data_range=1)
    return err