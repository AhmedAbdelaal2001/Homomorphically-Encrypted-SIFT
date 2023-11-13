import sys
sys.path.append('../utils_plaintextDomain')

from numpy import all, any, array, arctan2, cos, sin, exp, dot, log, logical_and, roll, sqrt, stack, trace, unravel_index, pi, deg2rad, rad2deg, where, zeros, floor, full, nan, isnan, round, float32
from numpy.linalg import det, lstsq, norm
from cv2 import resize, GaussianBlur, subtract, KeyPoint, INTER_LINEAR, INTER_NEAREST
from functools import cmp_to_key
import logging
from utils_plaintextDomain.utils import *
from scipy.ndimage import zoom

logger = logging.getLogger(__name__)
float_tolerance = 1e-7

def generateBaseImage(image, sigma, assumed_blur):
    """Generate base image from input image by upsampling by 2 in both directions and blurring
    """
    print('Generating base image...')
    image = resize(image, (0, 0), fx=2, fy=2, interpolation=INTER_LINEAR)
    sigma_diff = sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))
    return GaussianBlur(image, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)

def computeNumberOfOctaves(image_shape):
    """Compute number of octaves in image pyramid as function of base image shape (OpenCV default)
    """
    return int(round(log(min(image_shape)) / log(2) - 1))

def generateGaussianKernels(sigma, num_intervals):
    """Generate list of gaussian kernels at which to blur the input image. Default values of sigma, intervals, and octaves follow section 3 of Lowe's paper.
    """
    print('Generating scales...')
    num_images_per_octave = num_intervals + 3
    k = 2 ** (1. / num_intervals)
    gaussian_kernels = zeros(num_images_per_octave)  # scale of gaussian blur necessary to go from one blur scale to the next within an octave
    gaussian_kernels[0] = sigma

    for image_index in range(1, num_images_per_octave):
        sigma_previous = (k ** (image_index - 1)) * sigma
        sigma_total = k * sigma_previous
        gaussian_kernels[image_index] = sqrt(sigma_total ** 2 - sigma_previous ** 2)
    return gaussian_kernels

def generateGaussianImages(image, num_octaves, gaussian_kernels):
    """Generate scale-space pyramid of Gaussian images
    """
    print('Generating Gaussian images...')
    gaussian_images = []
    gaussian_kernels_filters = []
    for sigma in gaussian_kernels:
        gaussian_kernels_filters.append(generate_gaussian_kernel(sigma))

    for octave in range(num_octaves):
        gaussian_images_in_octave = [image]  # Start with a copy to avoid modifying the original
        for filter in gaussian_kernels_filters[1:]:  # Skip the first kernel since the first image is already at that scale
            pad_size = filter.shape[0] // 2
            gaussian_images_in_octave.append(convolve2D(image, filter, padding=pad_size).astype(np.float32))
        gaussian_images.append(gaussian_images_in_octave)
        
        # Prepare the next octave
        octave_base = gaussian_images_in_octave[-3]
        image = resize(octave_base, (int(octave_base.shape[1] / 2), int(octave_base.shape[0] / 2)), interpolation=INTER_NEAREST)

    return array(gaussian_images, dtype=object)

def generateDoGFilters(gaussian_kernels):
    """ Generate DoG filters from a list of Gaussian kernels """
    dog_filters = []
    for first_sigma, second_sigma in zip(gaussian_kernels, gaussian_kernels[1:]):
        first_filter = generate_gaussian_kernel(first_sigma)
        second_filter = generate_gaussian_kernel(second_sigma)
        first_filter, second_filter = pad_to_match(first_filter, second_filter)
        dog_filter = second_filter - first_filter
        dog_filters.append(dog_filter)
    return dog_filters

def generateDoGImages(gaussian_images):
    """Generate Difference-of-Gaussians image pyramid
    """
    print('Generating Difference-of-Gaussian images...')
    dog_images = []

    for gaussian_images_in_octave in gaussian_images:
        dog_images_in_octave = []
        for first_image, second_image in zip(gaussian_images_in_octave, gaussian_images_in_octave[1:]):
            dog_images_in_octave.append(subtract(second_image, first_image))  # ordinary subtraction will not work because the images are unsigned integers
        dog_images.append(dog_images_in_octave)
    return array(dog_images, dtype=object)

def generateDoGImagesUsingFilters(image, num_octaves, gaussian_kernels):
    """Generate scale-space pyramid of DoG images directly
    """
    print('Generating DoG images using filters...')
    dog_images = []
    dog_filters = generateDoGFilters(gaussian_kernels)

    for octave in range(num_octaves):
        dog_images_in_octave = []
        for filter in dog_filters:
            pad_size = filter.shape[0] // 2
            dog_image = convolve2D(image, filter, padding=pad_size).astype(np.float32)
            dog_images_in_octave.append(dog_image)
        dog_images.append(dog_images_in_octave)

        # Prepare the next octave
        octave_base = dog_images_in_octave[-3]
        image = zoom(octave_base, 0.5, order=0)

    return np.array(dog_images, dtype=object)