import skimage as ski
import numpy as np
import matplotlib.pyplot as plt


# 2D-channel functions, to be used in combination with either the BB image or the polygon/background image,
# as a possible value for the 'channel_func' constructor's paramete.
#
# Channels are 
# + created and stored on-disk as numpy arrays with image types (int, 0-255)
# + loaded as numpy arrays (0-255)
# Conversion to tensor (0-1) is done by the transform for both.

def bbox_alpha_mask(img_hwc: np.ndarray, mask_hw: np.ndarray, transparency=.4) -> np.ndarray:
    """ Create a transparency mask that keeps polygon clear, but background faded.
    Meant to be applied to a bounding box image.

    Args:
        img_hwc (np.ndarray): Ignored here, but required by the interface.
        mask_hw (np.ndarray): boolean array.
    Output:
        np.ndarray: (H,W) channel, with int values in the range 0-255 out of the polygons and 255 in
            the polygon.
    """
    mask = (np.full( mask_hw.shape, transparency ) * np.logical_not( mask_hw) + mask_hw)
    return (mask*255).astype('uint8')

def bbox_blurry_mask(img_hwc: np.ndarray, mask_hw: np.ndarray, kernel_size=10) -> np.ndarray:
    """ Create a mean-filtered version of the entire BB.
    Meant to be added to a tensor representing polygon+background image.

    Args:
        img_hwc (np.ndarray): array (H,W,C)-uint8, as created from a PIL image.
        mask_hw (np.ndarray): ignored here, but required by the interface.
    Output:
        np.ndarray: (H,W) channel, with float values in the range 0-1.
    """
    # note: skimage functions typically return 0-1 floats, no matter the input
    out = ski.util.img_as_ubyte( ski.filters.rank.mean( ski.color.rgb2gray(img_hwc), np.full((kernel_size,kernel_size),1)))
    return out

def bbox_gray_mask(img_hwc: np.ndarray, mask_hw: np.ndarray) -> np.ndarray:
    """ Create a gray version of the entire BB.
    Meant to be added to a tensor representing polygon+background image.
    
    Args:
        img_hwc (np.ndarray): array (H,W,C)-uint8, as created from a PIL image.
        mask_hw (np.ndarray): ignored here, but required by the interface.
    Output:
        np.ndarray: (H,W) channel, with float values in the range 0-1.
    """
    print("Into gray_mask_func()", img_hwc.dtype)
    out = ski.util.img_as_ubyte( ski.color.rgb2gray(img_hwc))
    print("gray_mask()", out.dtype, np.max(out))
    return out

def show_img_three_plus_one(img_chw):
    """ Combine first 3 channels with the 4th one 
    
    Args:
        img_chw (np.ndarray): (C,H,W) array, typically obtained from a tensor. 
    """
    img_array = (img_chw[:3].numpy()).transpose(1,2,0)
    mask_array = img_chw[3].numpy()
    mask_array = np.stack( [ mask_array, mask_array, mask_array ]).transpose(1,2,0)
    plt.imshow( img_array * mask_array )
    plt.show()

