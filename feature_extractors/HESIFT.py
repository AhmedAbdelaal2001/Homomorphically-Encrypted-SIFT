import sys
sys.path.append('../utils_encryptedDomain')

from numpy import all, any, array, arctan2, cos, sin, exp, dot, log, logical_and, roll, sqrt, stack, trace, unravel_index, pi, deg2rad, rad2deg, where, zeros, floor, full, nan, isnan, round, float32
from numpy.linalg import det, lstsq, norm
from cv2 import resize, GaussianBlur, subtract, KeyPoint, INTER_LINEAR, INTER_NEAREST
from functools import cmp_to_key
import logging
from SIFT import *
from utils_encryptedDomain.cryptosystem import *
from utils_encryptedDomain.homomorphic_operations import *

logger = logging.getLogger(__name__)
float_tolerance = 1e-7

def encrypted_generateDoGImagesUsingFilters(encryptedImage, num_octaves, gaussian_kernels):
    """Generate scale-space pyramid of DoG images directly
    """
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