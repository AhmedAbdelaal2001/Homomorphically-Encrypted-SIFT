import sys
sys.path.append('../utils_encryptedDomain')

from numpy import all, any, array, arctan2, cos, sin, exp, dot, log, logical_and, roll, sqrt, stack, trace, unravel_index, pi, deg2rad, rad2deg, where, zeros, floor, full, nan, isnan, round, float32
from numpy.linalg import det, lstsq, norm
from cv2 import resize, GaussianBlur, subtract, KeyPoint, INTER_LINEAR, INTER_NEAREST
from functools import cmp_to_key
import logging
from feature_extractors.SIFT import *
from utils_encryptedDomain.cryptosystem import *
from utils_encryptedDomain.homomorphic_operations import *

logger = logging.getLogger(__name__)
float_tolerance = 1e-7

def generateGaussianKernels(sigma, num_intervals):
    """Generate list of gaussian kernels at which to blur the input image. Default values of sigma, intervals, and octaves follow section 3 of Lowe's paper.
    """
    print('Generating Gaussian Kernels...')
    num_images_per_octave = num_intervals + 3
    k = 2 ** (1. / num_intervals)
    gaussian_kernels = []  # scale of gaussian blur necessary to go from one blur scale to the next within an octave

    for image_index in range(0, num_images_per_octave):
        gaussian_kernels.append(remove_zero_rows_columns((100 * generate_gaussian_kernel(sigma)).astype(np.int64)))
        sigma *= k
    return gaussian_kernels

def generateEncryptedGaussianImages(encryptedImage, num_octaves, gaussian_kernels):
    """Generate scale-space pyramid of Gaussian images
    """
    print('Generating Gaussian images...')
    gaussian_images = []

    for octave_index in range(num_octaves):
        print(f"Octave {octave_index + 1} Running:")
        gaussian_images_in_octave = []
        for kernel in gaussian_kernels:
            pad_size = max(kernel.shape) // 2
            gaussian_images_in_octave.append(encryptedConvolve2D(encryptedImage, kernel, padding=pad_size))
        print(f"Octave {octave_index + 1} Done!!!")
        print("-----------------------------------------------------------------------")
        gaussian_images.append(gaussian_images_in_octave)
        encryptedImage = zoom(encryptedImage, 0.5, order=0)
    return array(gaussian_images, dtype=object)

def generateEncryptedDoGImages(gaussian_images):
    """Generate Difference-of-Gaussians image pyramid
    """
    print('Generating Difference-of-Gaussian images...')
    dog_images = []
    octaveIndex = 1
    for gaussian_images_in_octave in gaussian_images:
        print(f"Octave {octaveIndex} Running:")
        dog_images_in_octave = []
        for first_image, second_image in zip(gaussian_images_in_octave, gaussian_images_in_octave[1:]):
            dog_images_in_octave.append(tensor_homomorphicSubtraction(second_image, first_image))  # ordinary subtraction will not work because the images are unsigned integers
        print(f"Octave {octaveIndex} Done!!!")
        print("-----------------------------------------------------------------------")
        dog_images.append(dog_images_in_octave)
    return array(dog_images, dtype=object)

"""
def encrypted_generateDoGImagesUsingFilters(encryptedImage, num_octaves, gaussian_kernels):
    #Generate scale-space pyramid of DoG images directly
    print('Generating DoG images using filters...')
    dog_images = []
    dog_filters = generateDoGFilters(gaussian_kernels)
    for i in range(len(dog_filters)):
        dog_filters[i] = (100 * dog_filters[i]).astype(np.int64)

    for octave in range(num_octaves):
        dog_images_in_octave = []
        for filter in dog_filters:
            pad_size = filter.shape[0] // 2
            dog_image = encryptedConvolve2D(encryptedImage, filter, padding=pad_size)
            print("convolution done")
            dog_images_in_octave.append(dog_image)
        dog_images.append(dog_images_in_octave)

        # Prepare the next octave
        octave_base = dog_images_in_octave[-3]
        image = zoom(octave_base, 0.5, order=0)
        print(octave)

    return np.array(dog_images, dtype=object)
"""