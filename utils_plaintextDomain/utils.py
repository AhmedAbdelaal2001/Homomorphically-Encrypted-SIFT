import skimage.io as io
import numpy as np

from skimage.color import rgb2gray,rgb2hsv
import matplotlib.pyplot as plt
from skimage.util import random_noise

from skimage.exposure import histogram
from matplotlib.pyplot import bar

def show_images(images,titles=None):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()

def convolve2D(image, kernel, padding=0, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.pad(image, ((padding, padding), (padding, padding)), mode='constant', constant_values=0)
    else:
        imagePadded = image

    # Iterate through image
    for y in range(imagePadded.shape[1]):
        # Exit Convolution
        if y > imagePadded.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(imagePadded.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > imagePadded.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output

def generate_gaussian_kernel(sigma):
    """
    Generates a Gaussian kernel given the standard deviation (sigma) and the shape of the image.
    The kernel size is automatically adjusted to ensure it's smaller than the image.
    """
    # Generate Gaussian kernel
    kernel_size = int(2 * np.ceil(3 * sigma) + 1)
    x = np.linspace(-np.ceil(3 * sigma), np.ceil(3 * sigma), kernel_size)
    y = np.linspace(-np.ceil(3 * sigma), np.ceil(3 * sigma), kernel_size)
    x, y = np.meshgrid(x, y)
    gaussian_kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    gaussian_kernel /= gaussian_kernel.sum()
    
    return gaussian_kernel

def pad_to_match(filter1, filter2):
    """ Pad the smaller filter to match the size of the larger one """
    if filter1.shape == filter2.shape:
        return filter1, filter2

    # Identify the larger and smaller filter
    larger_filter, smaller_filter = (filter1, filter2) if filter1.size > filter2.size else (filter2, filter1)

    # Calculate the padding required
    pad_y = (larger_filter.shape[0] - smaller_filter.shape[0] + 1) // 2
    pad_x = (larger_filter.shape[1] - smaller_filter.shape[1] + 1) // 2

    # Apply symmetric padding
    smaller_filter_padded = np.pad(smaller_filter, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant', constant_values=0)

    # Return the filters with matched sizes
    return (smaller_filter_padded, larger_filter) if filter1.size < filter2.size else (larger_filter, smaller_filter_padded)

import numpy as np

def remove_zero_rows_columns(tensor):
    """
    Remove rows and columns from a numpy array that contain only zeros.

    :param arr: Input numpy array.
    :return: Numpy array with zero-only rows and columns removed.
    """
    # Remove rows and columns that are all zeros
    non_zero_rows = ~np.all(tensor == 0, axis=1)
    non_zero_cols = ~np.all(tensor == 0, axis=0)
    result = tensor[non_zero_rows][:, non_zero_cols]

    return result

def replicate_border(image, top, bottom, left, right):
    """
    Manually replicate borders for an image.
    :param image: Input image (numpy array).
    :param top, bottom, left, right: Number of pixels to add on each side.
    :return: Image with replicated borders.
    """
    height, width = image.shape[:2]

    # Top border
    top_border = np.repeat(image[np.newaxis, 0, :], top, axis=0)

    # Bottom border
    bottom_border = np.repeat(image[np.newaxis, -1, :], bottom, axis=0)

    # Combine top and bottom borders with the original image
    bordered_image = np.vstack([top_border, image, bottom_border])

    # Update height after adding top and bottom borders
    new_height = bordered_image.shape[0]

    # Left border
    left_border = np.repeat(bordered_image[:, np.newaxis, 0], left, axis=1)

    # Right border
    right_border = np.repeat(bordered_image[:, np.newaxis, -1], right, axis=1)

    # Combine left and right borders
    bordered_image = np.hstack([left_border, bordered_image, right_border])

    return bordered_image
