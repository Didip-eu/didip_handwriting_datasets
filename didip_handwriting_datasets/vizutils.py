from skimage.filters.rank import mean
import numpy as np
import matplotlib.pyplot as plt


# 2D-mask functions, to be used in combination with either the BB image or the polygon/background image,
# as a possible value for the 'channel_func' constructor's parameter.
# Just shown as examples;
def bbox_alpha_mask(img_chw: np.ndarray, mask_hw: np.ndarray, channel_dim: int=0, transparency=.4) -> np.ndarray:
    """ Create a transparency mask that keeps polygon clear, but background faded.
    Meant to be applied to a bounding box image."""
    #img = img_chw.transpose(2,0,1) if channel_dim == 2 else img_chw
    mask = np.full( mask_hw.shape, transparency ) * np.logical_not( mask_hw) + mask_hw 
    return mask

def bbox_blurry_mask(img_chw: np.ndarray, mask_hw: np.ndarray, channel_dim: int=0,kernel_size=10) -> np.ndarray:
    """ Create a mean-filtered version of the entire BB.
    Meant to be applied to a polygon/background image."""
    return mean( np.average( img_chw, axis=0 ), np.full((kernel_size,kernel_size),1))

def bbox_gray_mask(img_chw: np.ndarray, mask_hw: np.ndarray, channel_dim: int=0) -> np.ndarray:
    """ Create a gray version of the entire BB."""
    return np.average(img_chw, axis=0)

def show_img_three_plus_one(img_chw):
    """ Combine first 3 channels with the 4th one """
    img_array = (img_chw[:3].numpy()/255).transpose(1,2,0)
    mask_array = img_chw[3].numpy()
    mask_array = np.stack( [ mask_array, mask_array, mask_array ]).transpose(1,2,0)
    plt.imshow( img_array * mask_array )
    plt.show()

