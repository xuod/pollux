from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K

from . import layers

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy
from IPython.display import clear_output, set_matplotlib_formats
from mpl_toolkits.axes_grid1 import make_axes_locatable

def build_vanilla_vae(encoder, decoder, full_cov=False, coeff_KL=1.0):
    input_vae = Input(shape=encoder.input.shape[1:])
    output_encoder = encoder(input_vae)

    z, Dkl = layers.SampleMultivariateGaussian(full_cov=full_cov, add_KL=True, return_KL=True, coeff_KL=coeff_KL)(output_encoder)

    vae = Model(input_vae, decoder(z))
    vae_utils = Model(input_vae, [*encoder(input_vae), z, Dkl, decoder(z)])

    return vae, vae_utils


def vae_rec_loss(x, x_decoded_mean):
    xent_loss = K.sum(K.binary_crossentropy(x, x_decoded_mean), axis=[1,2,3])
    # kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1) # KL divergence: from input image to latent variable
    # return xent_loss + kl_loss
    return K.mean(xent_loss)
#     return 64*64*10 * K.mean(K.binary_crossentropy(x, x_decoded_mean))


class VAEHistory(Callback):
    def __init__(self, xval_sub, vae_utils, latent_dim, plot_bands=[5,6,7]):
        self.xval_sub = xval_sub
        self.latent_dim = latent_dim
        self.plot_bands = plot_bands
        self.vae_utils = vae_utils
        
        self.counter = 0
        
        self.epoch = 0
        self.loss = []
        self.val_loss = []
        self.D_KL = []
        
        self.zz = np.linspace(-5,5,1000)
        self.gauss = scipy.stats.norm.pdf(self.zz)
                
        self.colors = mpl.cm.jet(np.linspace(0,1,latent_dim))
        
        
    def mask_outliers(points_plop, thresh=3.5):
        """
        Returns a boolean array with True if points are outliers and False 
        otherwise.

        Parameters:
        -----------
            points : An numobservations by numdimensions array of observations
            thresh : The modified z-score to use as a threshold. Observations with
                a modified z-score (based on the median absolute deviation) greater
                than this value will be classified as outliers.

        Returns:
        --------
            mask : A numobservations-length boolean array.

        References:
        ----------
            Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
            Handle Outliers", The ASQC Basic References in Quality Control:
            Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
        """
        points = np.array(points_plop)

        if len(points.shape) == 1:
            points = points[:,None]
        median = np.median(points, axis=0)
        diff = np.sum((points - median)**2, axis=-1)
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)

        modified_z_score = 0.6745 * diff / med_abs_deviation

        #return modified_z_score > thresh

        return np.ma.masked_array(data=points, mask=modified_z_score > thresh)

    def on_epoch_end(self, epoch, logs={}):
        self.epoch += 1
        
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
                
        mu, sigma, z, dkl, out = self.vae_utils.predict(self.xval_sub)
        self.D_KL.append(np.mean(dkl))

        clear_output(wait=True)
        
        fig, axes = plt.subplots(1, 7, figsize=(7*4,4))
        
#         axes = np.ravel(axes)
        
        ax = axes[0]
        ax.plot(self.loss, c='b', label='train')
        ax.plot(self.val_loss, c='r', label='val')
        ax.legend()
        ax.set_xlabel('epoch')
        ax.set_title('Training/validation losses')
        
#         def all_positive(list):
#             return all([x>0. for x in list])
#         if all_positive(self.loss) and all_positive(self.val_loss):
#             ax.set_yscale('log')
        
        ax = axes[1]
        ax.plot(self.D_KL)
        ax.set_xlabel('epoch')
        ax.set_title('D_KL')
        #ax.set_ylim(np.min(self.D_KL), self.D_KL[0]*1.05)

        for dim in range(self.latent_dim):
            c = self.colors[dim]
            _ = axes[2].hist(mu[:,dim], bins=50, histtype='step', color=c, label=str(dim), density=True)
            _ = axes[3].hist(sigma[:,dim], bins=50, log=False, histtype='step', color=c, label=str(dim), density=True)
            _ = axes[4].hist(z[:,dim], bins=50, histtype='step', color=c, label=str(dim), density=True)
        
        axes[4].plot(self.zz, self.gauss, c='k')
        axes[4].set_xlim(-5,5)
        #axes[3].set_xscale('log')
        
        axes[2].set_title('mu_y(X)')
        axes[3].set_title('sigma_y(X)')
        axes[4].set_title('z~N(0,1)')
        
        idx = np.random.randint(0,len(self.xval_sub))#, size=1)
        ax = axes[5]
        im = ax.imshow(self.xval_sub[idx][:,:,self.plot_bands])
        if isinstance(self.plot_bands, int):
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im, cax=cax)
        ax.set_title('input')
        ax.axis('off')
        
        ax = axes[6]
        im = ax.imshow(out[idx][:,:,self.plot_bands])
        if isinstance(self.plot_bands, int):
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im, cax=cax)
        ax.set_title('output')
        ax.axis('off')
        
        plt.tight_layout()
        plt.show()

