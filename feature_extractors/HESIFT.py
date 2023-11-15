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




def isEncryptedPixelAnExtremum(first_subimage, second_subimage, third_subimage, threshold):
    """Return True if the center element of the 3x3x3 input array is strictly greater than or less than all its neighbors, False otherwise
    """
    center_pixel_value = second_subimage[1, 1]

    if center_pixel_value > threshold:
            local_maxima = tensor_homomorphicComparator(center_pixel_value,first_subimage) and \
                   tensor_homomorphicComparator(center_pixel_value,third_subimage) and \
                   tensor_homomorphicComparator(center_pixel_value,second_subimage[0, :]) and \
                   tensor_homomorphicComparator(center_pixel_value,second_subimage[2, :]) and \
                   homomorphicComparator(center_pixel_value,second_subimage[1, 0]) and \
                   homomorphicComparator(center_pixel_value,second_subimage[1, 2])
            if not local_maxima:
                return tensor_homomorphicComparator(center_pixel_value,first_subimage,False) and \
                   tensor_homomorphicComparator(center_pixel_value,third_subimage,False) and \
                   tensor_homomorphicComparator(center_pixel_value,second_subimage[0, :],False) and \
                   tensor_homomorphicComparator(center_pixel_value,second_subimage[2, :],False) and \
                   homomorphicComparator(center_pixel_value,second_subimage[1, 0],False) and \
                   homomorphicComparator(center_pixel_value,second_subimage[1, 2],False)
            return local_maxima

    return False


def computeDecryptedGradientAtCenterPixel(pixel_array):
    """Approximate gradient at center pixel [1, 1, 1] of 3x3x3 array using central difference formula of order O(h^2), where h is the step size
    """
    # With step size h, the central difference formula of order O(h^2) for f'(x) is (f(x + h) - f(x - h)) / (2 * h)
    # Here h = 1, so the formula simplifies to f'(x) = (f(x + 1) - f(x - 1)) / 2
    # NOTE: x corresponds to second array axis, y corresponds to first array axis, and s (scale) corresponds to third array axis
    dx = homomorphicSubtraction(pixel_array[1, 1, 2] , pixel_array[1, 1, 0])
    dy = homomorphicSubtraction(pixel_array[1, 2, 1] , pixel_array[1, 0, 1])
    ds = homomorphicSubtraction(pixel_array[2, 1, 1] , pixel_array[0, 1, 1])
    return 0.5*decryptImage(array([dx, dy, ds]))


def computeDecryptedHessianAtCenterPixel(pixel_array):
    """Approximate Hessian at center pixel [1, 1, 1] of 3x3x3 array using central difference formula of order O(h^2), where h is the step size
    """
    # With step size h, the central difference formula of order O(h^2) for f''(x) is (f(x + h) - 2 * f(x) + f(x - h)) / (h ^ 2)
    # Here h = 1, so the formula simplifies to f''(x) = f(x + 1) - 2 * f(x) + f(x - 1)
    # With step size h, the central difference formula of order O(h^2) for (d^2) f(x, y) / (dx dy) = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ^ 2)
    # Here h = 1, so the formula simplifies to (d^2) f(x, y) / (dx dy) = (f(x + 1, y + 1) - f(x + 1, y - 1) - f(x - 1, y + 1) + f(x - 1, y - 1)) / 4
    # NOTE: x corresponds to second array axis, y corresponds to first array axis, and s (scale) corresponds to third array axis
    center_pixel_value = pixel_array[1, 1, 1]
    
    
    dxx = decrypt(homomorphicAddition(homomorphicSubtraction(pixel_array[1, 1, 2],homomorphicScalarMultiplication(center_pixel_value,2)),pixel_array[1, 1, 0]))
    dyy = decrypt(homomorphicAddition(homomorphicSubtraction (pixel_array[1, 2, 1],homomorphicScalarMultiplication(center_pixel_value,2)), pixel_array[1, 0, 1]))
    dss = decrypt(homomorphicAddition(homomorphicSubtraction (pixel_array[2, 1, 1],homomorphicScalarMultiplication(center_pixel_value,2)),pixel_array[0, 1, 1]))

    dxy = 0.25*decrypt(homomorphicAddition(homomorphicSubtraction(homomorphicSubtraction(pixel_array[1, 2, 2],pixel_array[1, 2, 0]),pixel_array[1, 0, 2]),pixel_array[1, 0, 0]))
    dxs = 0.25*decrypt(homomorphicAddition(homomorphicSubtraction(homomorphicSubtraction(pixel_array[2, 1, 2],pixel_array[2, 1, 0]),pixel_array[0, 1, 2]),pixel_array[0, 1, 0]))
    dys = 0.25*decrypt(homomorphicAddition(homomorphicSubtraction(homomorphicSubtraction(pixel_array[2, 2, 1],pixel_array[2, 0, 1]),pixel_array[0, 2, 1]), pixel_array[0, 0, 1]))

    return array([[dxx, dxy, dxs], 
                  [dxy, dyy, dys],
                  [dxs, dys, dss]])





def localizeDecryptedExtremumViaQuadraticFit(i, j, image_index, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width, eigenvalue_ratio=10, num_attempts_until_convergence=5):
    """Iteratively refine pixel positions of scale-space extrema via quadratic fit around each extremum's neighbors
    """
    logger.debug('Localizing scale-space extrema...')
    extremum_is_outside_image = False
    image_shape = dog_images_in_octave[0].shape
    for attempt_index in range(num_attempts_until_convergence):
        first_image, second_image, third_image = dog_images_in_octave[image_index-1:image_index+2]
        pixel_cube = stack([first_image[i-1:i+2, j-1:j+2],
                            second_image[i-1:i+2, j-1:j+2],
                            third_image[i-1:i+2, j-1:j+2]])
        
        gradient = computeDecryptedGradientAtCenterPixel(pixel_cube)
        hessian = computeDecryptedHessianAtCenterPixel(pixel_cube)

        extremum_update = -lstsq(hessian, gradient, rcond=None)[0]

        if abs(extremum_update[0]) < 0.5 and abs(extremum_update[1]) < 0.5 and abs(extremum_update[2]) < 0.5:
            break
        j += int(round(extremum_update[0]))
        i += int(round(extremum_update[1]))
        image_index += int(round(extremum_update[2]))
        # make sure the new pixel_cube will lie entirely within the image
        if i < image_border_width or i >= image_shape[0] - image_border_width or j < image_border_width or j >= image_shape[1] - image_border_width or image_index < 1 or image_index > num_intervals:
            extremum_is_outside_image = True
            break
    if extremum_is_outside_image:
       
        return None
    if attempt_index >= num_attempts_until_convergence - 1:
       
        return None
    
    functionValueAtUpdatedExtremum = decrypt(pixel_cube[1, 1, 1]) + 0.5 * dot(gradient, extremum_update)

    if abs(functionValueAtUpdatedExtremum) * num_intervals >= contrast_threshold:
        xy_hessian = hessian[:2, :2]
        xy_hessian_trace = trace(xy_hessian)
        xy_hessian_det = det(xy_hessian)
        if xy_hessian_det > 0 and eigenvalue_ratio * (xy_hessian_trace ** 2) < ((eigenvalue_ratio + 1) ** 2) * xy_hessian_det:
            # Contrast check passed -- construct and return OpenCV KeyPoint object
            keypoint = KeyPoint()
            keypoint.pt = ((j + extremum_update[0]) * (2 ** octave_index), (i + extremum_update[1]) * (2 ** octave_index))
            keypoint.octave = octave_index + image_index * (2 ** 8) + int(round((extremum_update[2] + 0.5) * 255)) * (2 ** 16)
            keypoint.size = sigma * (2 ** ((image_index + extremum_update[2]) / float32(num_intervals))) * (2 ** (octave_index + 1))  # octave_index + 1 because the input image was doubled
            keypoint.response = abs(functionValueAtUpdatedExtremum)
            return keypoint, image_index
    return None


def computeDecryptedKeypointsWithOrientations(keypoint, octave_index, gaussian_image, radius_factor=3, num_bins=36, peak_ratio=0.8, scale_factor=1.5):
    """Compute orientations for each keypoint
    """
    logger.debug('Computing keypoint orientations...')
    keypoints_with_orientations = []
    image_shape = gaussian_image.shape

    scale = scale_factor * keypoint.size / float32(2 ** (octave_index + 1))  # compare with keypoint.size computation in localizeExtremumViaQuadraticFit()
    radius = int(round(radius_factor * scale))
    weight_factor = -0.5 / (scale ** 2)
    raw_histogram = zeros(num_bins)
    smooth_histogram = zeros(num_bins)

    for i in range(-radius, radius + 1):
        region_y = int(round(keypoint.pt[1] / float32(2 ** octave_index))) + i
        if region_y > 0 and region_y < image_shape[0] - 1:
            for j in range(-radius, radius + 1):
                region_x = int(round(keypoint.pt[0] / float32(2 ** octave_index))) + j
                if region_x > 0 and region_x < image_shape[1] - 1:

                    dx = decrypt(homomorphicSubtraction(gaussian_image[region_y, region_x + 1],gaussian_image[region_y, region_x - 1]))
                    dy = decrypt(homomorphicSubtraction(gaussian_image[region_y - 1, region_x],gaussian_image[region_y + 1, region_x]))
                    
                    #Awel khazoo2
                    gradient_magnitude = sqrt(dx * dx + dy * dy)
                    #Tany khazoo2
                    gradient_orientation = rad2deg(arctan2(dy, dx))

                    weight = exp(weight_factor * (i ** 2 + j ** 2))  # constant in front of exponential can be dropped because we will find peaks later
                    histogram_index = int(round(gradient_orientation * num_bins / 360.))
                    raw_histogram[histogram_index % num_bins] += weight * gradient_magnitude

    for n in range(num_bins):
        smooth_histogram[n] = (6 * raw_histogram[n] + 4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % num_bins]) + raw_histogram[n - 2] + raw_histogram[(n + 2) % num_bins]) / 16.
    orientation_max = max(smooth_histogram)
    orientation_peaks = where(logical_and(smooth_histogram > roll(smooth_histogram, 1), smooth_histogram > roll(smooth_histogram, -1)))[0]
    for peak_index in orientation_peaks:
        peak_value = smooth_histogram[peak_index]
        if peak_value >= peak_ratio * orientation_max:
            # Quadratic peak interpolation
            # The interpolation update is given by equation (6.30) in https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
            left_value = smooth_histogram[(peak_index - 1) % num_bins]
            right_value = smooth_histogram[(peak_index + 1) % num_bins]
            interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (left_value - 2 * peak_value + right_value)) % num_bins
            orientation = 360. - interpolated_peak_index * 360. / num_bins
            if abs(orientation - 360.) < float_tolerance:
                orientation = 0
            new_keypoint = KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
            keypoints_with_orientations.append(new_keypoint)
    return keypoints_with_orientations


def findDecryptedScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width, contrast_threshold=0.04):
    """Find pixel positions of all scale-space extrema in the image pyramid
    """
    logger.debug('Finding scale-space extrema...')
    threshold = floor(0.5 * contrast_threshold / num_intervals * 255)  # from OpenCV implementation
    keypoints = []

    for octave_index, dog_images_in_octave in enumerate(dog_images):
        for image_index, (first_image, second_image, third_image) in enumerate(zip(dog_images_in_octave, dog_images_in_octave[1:], dog_images_in_octave[2:])):
            # (i, j) is the center of the 3x3 array
            for i in range(image_border_width, first_image.shape[0] - image_border_width):
                for j in range(image_border_width, first_image.shape[1] - image_border_width):
                    if isEncryptedPixelAnExtremum(first_image[i-1:i+2, j-1:j+2], second_image[i-1:i+2, j-1:j+2], third_image[i-1:i+2, j-1:j+2], threshold):
                       
                        localization_result = localizeDecryptedExtremumViaQuadraticFit(i, j, image_index + 1, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width)
                        if localization_result is not None:
                            keypoint, localized_image_index = localization_result

                            keypoints_with_orientations = computeDecryptedKeypointsWithOrientations(keypoint, octave_index, gaussian_images[octave_index][localized_image_index])
                            for keypoint_with_orientation in keypoints_with_orientations:
                                keypoints.append(keypoint_with_orientation)
    return keypoints