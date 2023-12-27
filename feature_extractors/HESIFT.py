import sys
sys.path.append('../utils_encryptedDomain')

from numpy import all, any, array, arctan2, cos, sin, exp, dot, log, logical_and, roll, sqrt, stack, trace, unravel_index, pi, deg2rad, rad2deg, where, zeros, floor, full, nan, isnan, round, float32
from numpy.linalg import det, lstsq, norm
from cv2 import resize, GaussianBlur, subtract, KeyPoint, INTER_LINEAR, INTER_NEAREST
from functools import cmp_to_key
import logging
from utils_encryptedDomain.cryptosystem import *
from utils_encryptedDomain.homomorphic_operations import *
from scipy.ndimage import zoom

logger = logging.getLogger(__name__)
float_tolerance = 1e-7

def prepareInitialHESIFTImage(input_image: np.ndarray, initial_sigma: float, initial_blur: float) -> np.ndarray:
    """
    Prepare the initial image for SIFT processing by upsampling and applying Gaussian blur.
    
    Args:
    input_image (np.ndarray): The image to be processed.
    initial_sigma (float): The standard deviation for the Gaussian kernel.
    initial_blur (float): The assumed initial blur in the image.

    Returns:
    np.ndarray: The processed image after upsampling and blurring.
    """

    # Log the start of the base image generation process
    print('Preparing initial HESIFT image...')

    # Upsample the input image by a factor of 2 both horizontally and vertically
    # INTER_LINEAR interpolation is used for resizing
    upsampled_image = resize(input_image, (0, 0), fx=2, fy=2, interpolation=INTER_LINEAR)

    # Calculate the net effective sigma needed for Gaussian blurring
    # It's computed based on the desired sigma and the assumed initial blur
    effective_sigma = sqrt(max((initial_sigma ** 2) - ((2 * initial_blur) ** 2), 0.01))

    # Apply Gaussian blur to the upsampled image with the calculated effective sigma
    # This step adds the necessary blur to make the image scale-invariant
    blurred_image = GaussianBlur(upsampled_image, (0, 0), sigmaX=effective_sigma, sigmaY=effective_sigma)

    return blurred_image

def computeNumberOfOctaves(image_shape: Tuple[int, ...]) -> int:
    """Compute number of octaves in image pyramid as function of base image shape (OpenCV default)
    """
    return int(round(log(min(image_shape)) / log(2) - 1))


def createGaussianBlurKernels(initial_sigma: float, intervals_per_octave: int) -> List[np.ndarray]:
    """
    Create a series of Gaussian blur kernels for different scales.

    Args:
    initial_sigma (float): The initial standard deviation for the Gaussian blur.
    intervals_per_octave (int): The number of intervals per octave in the scale space.

    Returns:
    List[np.ndarray]: A list of Gaussian kernels for different blur levels.
    """

    # Log the start of Gaussian kernel generation
    print('Creating Gaussian Blur Kernels...')

    # Determine the total number of images per octave including the initial three
    total_images_per_octave = intervals_per_octave + 3

    # Calculate the constant multiplicative factor k
    k_multiplier = 2 ** (1.0 / intervals_per_octave)

    # Initialize an empty list to store the Gaussian kernels
    blur_kernels = []

    # Iterate over each image index in the octave
    for index in range(total_images_per_octave):
        # Generate a Gaussian kernel and scale it up by 100
        # Convert the result to integer for compatibility with homomorphic operations
        scaled_kernel = remove_zero_rows_columns((100 * generate_gaussian_kernel(initial_sigma)).astype(np.int64))

        # Append the scaled kernel to the list
        blur_kernels.append(scaled_kernel)

        # Update the sigma value for the next kernel
        initial_sigma *= k_multiplier

    return blur_kernels


def createScaleSpacePyramid(encrypted_img: np.ndarray, octave_count: int, blur_kernels: List[np.ndarray]) -> np.ndarray:
    """
    Generate a scale-space pyramid of Gaussian-blurred images in an encrypted domain.

    Args:
    encrypted_img (np.ndarray): The encrypted image to be processed.
    octave_count (int): The number of octaves to generate in the scale space.
    blur_kernels (List[np.ndarray]): List of Gaussian kernels for blurring.

    Returns:
    np.ndarray: A scale-space pyramid of Gaussian-blurred images.
    """

    # Log the start of the Gaussian image generation
    print('Creating Scale-Space Pyramid...')

    # Initialize an empty list to store the Gaussian images for each octave
    scale_space_images = []

    # Iterate over each octave
    for octave in range(octave_count):
        print(f"Processing Octave {octave + 1}:")

        # Initialize an empty list to store Gaussian images for the current octave
        images_in_current_octave = []

        # Iterate over each Gaussian kernel
        for kernel in blur_kernels:
            # Calculate padding size for the convolution
            padding_size = max(kernel.shape) // 2

            # Apply encrypted convolution and add the result to the octave's image list
            blurred_image = encryptedConvolve2D(encrypted_img, kernel, padding=padding_size)
            images_in_current_octave.append(blurred_image)
            print("Convolution Done!")

        # Add the octave's Gaussian images to the scale space list
        print(f"Octave {octave + 1} Completed!")
        print("-----------------------------------------------------------------------")
        scale_space_images.append(images_in_current_octave)

        # Downsample the image by a factor of 2 for the next octave
        encrypted_img = zoom(encrypted_img, 0.5, order=0)

    # Convert the list of images to a NumPy array for easier handling
    return array(scale_space_images, dtype=object)


def createEncryptedDoGPyramid(blurred_images: List[List[np.ndarray]]) -> np.ndarray:
    """
    Create a Difference-of-Gaussians (DoG) image pyramid from Gaussian-blurred images in an encrypted domain.

    Args:
    blurred_images (List[List[np.ndarray]]): A list of lists containing Gaussian-blurred images at different scales.

    Returns:
    np.ndarray: A pyramid of DoG images.
    """

    # Log the start of DoG image generation
    print('Creating Encrypted DoG Image Pyramid...')

    # Initialize an empty list to store the DoG images for all octaves
    dog_image_pyramid = []

    # Iterate over each octave in the blurred images
    for images_in_octave in blurred_images:
        # Initialize an empty list to store the DoG images for the current octave
        dog_images_current_octave = []

        # Iterate over each pair of adjacent Gaussian-blurred images
        for first_blurred_image, second_blurred_image in zip(images_in_octave, images_in_octave[1:]):
            # Calculate the DoG image by subtracting the first image from the second
            # Use homomorphic subtraction to handle encrypted images
            dog_image = tensor_homomorphicSubtraction(second_blurred_image, first_blurred_image)
            dog_images_current_octave.append(dog_image)

        # Add the octave's DoG images to the pyramid
        dog_image_pyramid.append(dog_images_current_octave)

    # Convert the list of DoG images to a NumPy array for easier handling
    return array(dog_image_pyramid, dtype=object)


def checkEncryptedPixelExtremum(central_pixel_tensor: np.ndarray, neighbor_pixel_tensor: np.ndarray) -> int:
    """
    Check whether a given pixel in an encrypted image is an extremum (maximum or minimum) compared to its neighbors.

    Args:
    central_pixel_tensor (np.ndarray): The encrypted value of the central pixel.
    neighbor_pixel_tensor (np.ndarray): The encrypted values of the neighboring pixels.

    Returns:
    int: > 0 if the central pixel is an extremum, 0 otherwise.
    """

    # Compare the central pixel to its neighbors using homomorphic operations
    # This will determine if the central pixel is a local maximum or minimum
    is_local_max, is_local_min = tensor_homomorphicComparator(central_pixel_tensor, neighbor_pixel_tensor)

    # Return True if the pixel is either a local maximum or minimum
    return is_local_max + is_local_min



def calculateDecryptedImageGradient(center_pixel_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate the gradient at the center pixel of a 3x3x3 array in an encrypted image using the central difference formula.

    Args:
    center_pixel_matrix (np.ndarray): A 3x3x3 array representing a portion of an encrypted image.

    Returns:
    np.ndarray: The decrypted gradient vector at the center pixel.
    """

    # Calculate the gradient in the x-direction (horizontal)
    # Using central difference: (f(x + 1) - f(x - 1)) / 2
    gradient_x = homomorphicSubtraction(center_pixel_matrix[1, 1, 2], center_pixel_matrix[1, 1, 0])

    # Calculate the gradient in the y-direction (vertical)
    gradient_y = homomorphicSubtraction(center_pixel_matrix[1, 2, 1], center_pixel_matrix[1, 0, 1])

    # Calculate the gradient in the s-direction (scale)
    gradient_scale = homomorphicSubtraction(center_pixel_matrix[2, 1, 1], center_pixel_matrix[0, 1, 1])

    # Decrypt the gradient vector and scale it down
    decrypted_gradient = decryptImage(array([gradient_x, gradient_y, gradient_scale])).astype(np.float32) / 200

    return decrypted_gradient



def calculateDecryptedHessianMatrix(encrypted_pixel_block: np.ndarray) -> np.ndarray:
    """
    Calculate the Hessian matrix at the center pixel of a 3x3x3 array in an encrypted image using the central difference formula.

    Args:
    encrypted_pixel_block (np.ndarray): A 3x3x3 array representing a portion of an encrypted image.

    Returns:
    np.ndarray: The decrypted Hessian matrix at the center pixel.
    """

    # Define the center pixel value for reference in calculations
    center_pixel = encrypted_pixel_block[1, 1, 1]
    
    # Calculate second-order derivatives using the central difference formula
    # For x, y, and scale (s) directions
    second_derivative_xx = decrypt(
        homomorphicAddition(
            homomorphicSubtraction(encrypted_pixel_block[1, 1, 2], 
                                   homomorphicScalarMultiplication(center_pixel, 2)),
            encrypted_pixel_block[1, 1, 0])
    ).astype(np.float32) / 100

    second_derivative_yy = decrypt(
        homomorphicAddition(
            homomorphicSubtraction(encrypted_pixel_block[1, 2, 1], 
                                   homomorphicScalarMultiplication(center_pixel, 2)),
            encrypted_pixel_block[1, 0, 1])
    ).astype(np.float32) / 100

    second_derivative_ss = decrypt(
        homomorphicAddition(
            homomorphicSubtraction(encrypted_pixel_block[2, 1, 1], 
                                   homomorphicScalarMultiplication(center_pixel, 2)),
            encrypted_pixel_block[0, 1, 1])
    ).astype(np.float32) / 100

    # Calculate mixed derivatives (cross-terms)
    mixed_derivative_xy = decrypt(
        homomorphicAddition(
            homomorphicSubtraction(
                homomorphicSubtraction(encrypted_pixel_block[1, 2, 2], encrypted_pixel_block[1, 2, 0]),
                encrypted_pixel_block[1, 0, 2]),
            encrypted_pixel_block[1, 0, 0])
    ).astype(np.float32) / 400

    mixed_derivative_xs = decrypt(
        homomorphicAddition(
            homomorphicSubtraction(
                homomorphicSubtraction(encrypted_pixel_block[2, 1, 2], encrypted_pixel_block[2, 1, 0]),
                encrypted_pixel_block[0, 1, 2]),
            encrypted_pixel_block[0, 1, 0])
    ).astype(np.float32) / 400

    mixed_derivative_ys = decrypt(
        homomorphicAddition(
            homomorphicSubtraction(
                homomorphicSubtraction(encrypted_pixel_block[2, 2, 1], encrypted_pixel_block[2, 0, 1]),
                encrypted_pixel_block[0, 2, 1]),
            encrypted_pixel_block[0, 0, 1])
    ).astype(np.float32) / 400

    # Construct and return the Hessian matrix
    return array([[second_derivative_xx, mixed_derivative_xy, mixed_derivative_xs], 
                  [mixed_derivative_xy, second_derivative_yy, mixed_derivative_ys],
                  [mixed_derivative_xs, mixed_derivative_ys, second_derivative_ss]])






def refineExtremumPositionInScaleSpace(row: int, col: int, img_idx: int, octave_lvl: int, intervals: int, 
                                       dog_octave_imgs: List[np.ndarray], blur_sigma: float, threshold: float,
                                       border_width: int, eigen_ratio: Optional[float] = 10, 
                                       max_attempts: Optional[int] = 5) -> Tuple[KeyPoint, int]:
    """
    Iteratively refine the position of scale-space extrema using a quadratic fitting approach.

    Args:
    row, col (int): Initial row and column indices of the extremum.
    img_idx (int): Index of the image in the octave.
    octave_lvl (int): Current octave level.
    intervals (int): Number of intervals in the scale space.
    dog_octave_imgs (List[np.ndarray]): DoG images in the current octave.
    blur_sigma (float): Sigma value used for blurring.
    threshold (float): Contrast threshold for feature rejection.
    border_width (int): Width of the image border to avoid.
    eigen_ratio (float): Ratio for edge response elimination.
    max_attempts (int): Maximum number of attempts for convergence.

    Returns:
    tuple or None: Returns a tuple containing the refined KeyPoint object and its image index if successful, else None.
    """

    print('Refining extremum position in scale-space...')

    # Flag to check if extremum goes outside the image boundary
    is_extremum_outside = False
    img_dimensions = dog_octave_imgs[0].shape

    # Iteratively refine the extremum position
    for attempt in range(max_attempts):
        # Extract the images around the current index
        prev_img, current_img, next_img = dog_octave_imgs[img_idx-1:img_idx+2]

        # Construct a 3x3x3 pixel block around the extremum
        pixel_block = stack([prev_img[row-1:row+2, col-1:col+2],
                             current_img[row-1:row+2, col-1:col+2],
                             next_img[row-1:row+2, col-1:col+2]])

        # Compute gradient and Hessian at the central pixel
        gradient_vector = calculateDecryptedImageGradient(pixel_block)
        hessian_matrix = calculateDecryptedHessianMatrix(pixel_block)

        # Calculate extremum update using least squares solution
        extremum_shift = -lstsq(hessian_matrix, gradient_vector, rcond=None)[0]

        # Check if update is small enough to stop
        if all(abs(extremum_shift) < 0.5):
            break

        # Update the extremum position
        col += int(round(extremum_shift[0]))
        row += int(round(extremum_shift[1]))
        img_idx += int(round(extremum_shift[2]))

        # Check if the new position is within the image boundaries
        if not (border_width <= row < img_dimensions[0] - border_width and
                border_width <= col < img_dimensions[1] - border_width and
                1 <= img_idx <= intervals):
            is_extremum_outside = True
            break

    # Return None if extremum is outside or didn't converge
    if is_extremum_outside or attempt >= max_attempts - 1:
        return None

    # Evaluate the function value at the updated extremum position
    updated_value = decrypt(pixel_block[1, 1, 1]) + 0.5 * dot(gradient_vector, extremum_shift)

    # Apply contrast threshold check
    if abs(updated_value) * intervals >= threshold:
        # Compute Hessian's trace and determinant for edge response check
        xy_hessian = hessian_matrix[:2, :2]
        trace_hessian = trace(xy_hessian)
        determinant_hessian = det(xy_hessian)

        # Edge response elimination check
        if determinant_hessian > 0 and eigen_ratio * (trace_hessian ** 2) < ((eigen_ratio + 1) ** 2) * determinant_hessian:
            # Construct and return a KeyPoint object
            keypoint = KeyPoint()
            keypoint.pt = ((col + extremum_shift[0]) * (2 ** octave_lvl), (row + extremum_shift[1]) * (2 ** octave_lvl))
            keypoint.octave = octave_lvl + img_idx * (2 ** 8) + int(round((extremum_shift[2] + 0.5) * 255)) * (2 ** 16)
            keypoint.size = blur_sigma * (2 ** ((img_idx + extremum_shift[2]) / float32(intervals))) * (2 ** (octave_lvl + 1))
            keypoint.response = abs(updated_value)
            return keypoint, img_idx

    return None



def assignOrientationsToDecryptedKeypoints(detected_keypoint: KeyPoint, octave_level: int, blurred_image: np.ndarray, 
                                           radius_multiplier: Optional[int] = 3, 
                                           histogram_bins: Optional[int] = 36, 
                                           peak_threshold: Optional[float] = 0.8, 
                                           scale_multiplier: Optional[float] = 1.5) -> List[KeyPoint]:
    """
    Assign orientations to a given keypoint based on the gradients around it in the Gaussian image.

    Args:
    detected_keypoint (KeyPoint): The keypoint for which orientations need to be assigned.
    octave_level (int): The octave level of the keypoint.
    blurred_image (np.ndarray): The Gaussian-blurred image corresponding to the keypoint's octave.
    radius_multiplier (int): Factor to determine the radius of the region around the keypoint.
    histogram_bins (int): Number of bins in the orientation histogram.
    peak_threshold (float): Threshold to identify prominent orientations.
    scale_multiplier (float): Scale factor to determine the effective scale of the keypoint.

    Returns:
    List[KeyPoint]: A list of keypoints with assigned orientations.
    """

    print('Assigning orientations to keypoints...')

    # Initialize a list to hold keypoints with assigned orientations
    oriented_keypoints = []
    img_shape = blurred_image.shape

    # Calculate the effective scale and radius around the keypoint
    effective_scale = scale_multiplier * detected_keypoint.size / float32(2 ** (octave_level + 1))
    search_radius = int(round(radius_multiplier * effective_scale))
    gaussian_weight_factor = -0.5 / (effective_scale ** 2)
    orientation_histogram = zeros(histogram_bins)
    smoothed_histogram = zeros(histogram_bins)

    # Iterate over pixels in the region around the keypoint
    for i in range(-search_radius, search_radius + 1):
        region_y = int(round(detected_keypoint.pt[1] / float32(2 ** octave_level))) + i
        if 0 < region_y < img_shape[0] - 1:
            for j in range(-search_radius, search_radius + 1):
                region_x = int(round(detected_keypoint.pt[0] / float32(2 ** octave_level))) + j
                if 0 < region_x < img_shape[1] - 1:
                    # Compute gradients using decrypted image values
                    gradient_x = decrypt(homomorphicSubtraction(blurred_image[region_y, region_x + 1], blurred_image[region_y, region_x - 1])).astype(np.float32) / 100
                    gradient_y = decrypt(homomorphicSubtraction(blurred_image[region_y - 1, region_x], blurred_image[region_y + 1, region_x])).astype(np.float32) / 100

                    # Calculate gradient magnitude and orientation
                    magnitude = sqrt(gradient_x * gradient_x + gradient_y * gradient_y)
                    orientation = rad2deg(arctan2(gradient_y, gradient_x))
                    weight = exp(gaussian_weight_factor * (i ** 2 + j ** 2))
                    histogram_idx = int(round(orientation * histogram_bins / 360.)) % histogram_bins
                    orientation_histogram[histogram_idx] += weight * magnitude

    # Smooth the orientation histogram
    for n in range(histogram_bins):
        smoothed_histogram[n] = (6 * orientation_histogram[n] + 4 * (orientation_histogram[n - 1] + orientation_histogram[(n + 1) % histogram_bins]) + orientation_histogram[n - 2] + orientation_histogram[(n + 2) % histogram_bins]) / 16.

    # Identify peaks in the smoothed histogram
    max_orientation = max(smoothed_histogram)
    prominent_peaks = where(logical_and(smoothed_histogram > roll(smoothed_histogram, 1), smoothed_histogram > roll(smoothed_histogram, -1)))[0]
    for peak_idx in prominent_peaks:
        peak_value = smoothed_histogram[peak_idx]
        if peak_value >= peak_threshold * max_orientation:
            # Interpolate the peak position for higher accuracy
            left_neighbor = smoothed_histogram[(peak_idx - 1) % histogram_bins]
            right_neighbor = smoothed_histogram[(peak_idx + 1) % histogram_bins]
            interpolated_peak_idx = (peak_idx + 0.5 * (left_neighbor - right_neighbor) / (left_neighbor - 2 * peak_value + right_neighbor)) % histogram_bins
            final_orientation = 360. - interpolated_peak_idx * 360. / histogram_bins
            if abs(final_orientation - 360.) < float_tolerance:
                final_orientation = 0

            # Create a new keypoint with the assigned orientation
            new_keypoint = KeyPoint(*detected_keypoint.pt, detected_keypoint.size, final_orientation, detected_keypoint.response, detected_keypoint.octave)
            oriented_keypoints.append(new_keypoint)

    return oriented_keypoints



def detectScaleSpaceExtremaInEncryptedImages(blurred_imgs: List[List[np.ndarray]], dog_imgs: List[List[np.ndarray]], 
                                             interval_count: int, blur_sigma: float, border_size: int, 
                                             contrast_thresh: Optional[float] = 0.04) -> List[KeyPoint]:
    """
    Detect extrema in the scale-space of encrypted images.

    Args:
    blurred_imgs (List[List[np.ndarray]]): List of Gaussian-blurred images in the scale-space.
    dog_imgs (List[List[np.ndarray]]): List of Difference-of-Gaussian images in the scale-space.
    interval_count (int): Number of intervals in the scale-space.
    blur_sigma (float): Sigma value used for blurring.
    border_size (int): Width of the border to be ignored while detecting extrema.
    contrast_thresh (float): Threshold for contrast to filter out weak extrema.

    Returns:
    List[KeyPoint]: A list of keypoints detected in the scale-space.
    """

    print('Detecting scale-space extrema in encrypted images...')
    # Compute the threshold value for extrema detection
    detection_threshold = floor(0.5 * contrast_thresh / interval_count * 255)

    # Initialize a list to store detected keypoints
    detected_keypoints = []

    # Iterate over each octave in the scale-space
    for octave_idx, dog_octave in enumerate(dog_imgs):
        # Iterate over each set of three images in the DoG octave
        for img_idx, (prev_img, current_img, next_img) in enumerate(zip(dog_octave, dog_octave[1:], dog_octave[2:])):
            # Prepare tensors for center pixels and their neighbors
            center_pixels = []
            neighbor_pixels = []

            # Scan through the image, avoiding the borders
            for row in range(border_size, prev_img.shape[0] - border_size):
                for col in range(border_size, prev_img.shape[1] - border_size):
                    # Gather the 3x3 regions from the triplet of images
                    neighbor_prev = prev_img[row-1:row+2, col-1:col+2]
                    neighbor_current = current_img[row-1:row+2, col-1:col+2]
                    neighbor_next = next_img[row-1:row+2, col-1:col+2]
                    center_value = np.full((3,3), neighbor_current[1,1])

                    # Add center and neighbor pixels to the respective lists
                    center_pixels.append([center_value, center_value, center_value])
                    neighbor_pixels.append([neighbor_prev, neighbor_current, neighbor_next])

            # Convert lists to arrays for extrema checking
            center_pixels_array = np.array(center_pixels)
            neighbor_pixels_array = np.array(neighbor_pixels)
            extrema_results = checkEncryptedPixelExtremum(center_pixels_array, neighbor_pixels_array)

            # Iterate over the results to refine keypoints and assign orientations
            row, col = border_size, border_size
            for is_extremum in extrema_results:
                if is_extremum:
                    # Refine the position of each extremum
                    refined_result = refineExtremumPositionInScaleSpace(row, col, img_idx + 1, octave_idx, interval_count, dog_octave, blur_sigma, contrast_thresh, border_size)
                    if refined_result is not None:
                        refined_keypoint, refined_img_idx = refined_result
                        # Assign orientations to the refined keypoints
                        keypoints_with_orientations = assignOrientationsToDecryptedKeypoints(refined_keypoint, octave_idx, blurred_imgs[octave_idx][refined_img_idx])
                        detected_keypoints.extend(keypoints_with_orientations)

                # Update the row and column indices
                col += 1
                if col == prev_img.shape[1] - border_size:
                    col = border_size
                    row += 1

    return detected_keypoints


def orderKeypointsByFeatures(kp1: KeyPoint, kp2: KeyPoint) -> float:
    """
    Compare two keypoints and determine their order based on several attributes.

    Args:
    kp1, kp2 (KeyPoint): Two keypoints to be compared.

    Returns:
    float: A negative, zero, or positive value indicating the order of kp1 relative to kp2.
    """

    # Compare by x-coordinate
    if kp1.pt[0] != kp2.pt[0]:
        return kp1.pt[0] - kp2.pt[0]

    # Compare by y-coordinate
    if kp1.pt[1] != kp2.pt[1]:
        return kp1.pt[1] - kp2.pt[1]

    # Compare by size (larger keypoints first)
    if kp1.size != kp2.size:
        return kp2.size - kp1.size

    # Compare by angle
    if kp1.angle != kp2.angle:
        return kp1.angle - kp2.angle

    # Compare by response (stronger responses first)
    if kp1.response != kp2.response:
        return kp2.response - kp1.response

    # Compare by octave (higher octaves first)
    if kp1.octave != kp2.octave:
        return kp2.octave - kp1.octave

    # Lastly, compare by class ID
    return kp2.class_id - kp1.class_id


def filterOutDuplicateKeypoints(detected_keypoints: List[KeyPoint]) -> List[KeyPoint]:
    """
    Sort the detected keypoints and remove any duplicates.

    Args:
    detected_keypoints (List[KeyPoint]): A list of keypoints detected in the image.

    Returns:
    List[KeyPoint]: A list of unique keypoints after removing duplicates.
    """

    # Return the list as is if it has less than two keypoints
    if len(detected_keypoints) < 2:
        return detected_keypoints

    # Sort the keypoints using the custom comparison function
    detected_keypoints.sort(key=cmp_to_key(orderKeypointsByFeatures))

    # Initialize a list to hold the unique keypoints, starting with the first keypoint
    unique_kps = [detected_keypoints[0]]

    # Iterate over the sorted keypoints, starting from the second keypoint
    for current_kp in detected_keypoints[1:]:
        # Get the most recently added unique keypoint
        last_unique_kp = unique_kps[-1]

        # Check if the current keypoint differs from the last unique keypoint
        if (last_unique_kp.pt[0] != current_kp.pt[0] or
            last_unique_kp.pt[1] != current_kp.pt[1] or
            last_unique_kp.size != current_kp.size or
            last_unique_kp.angle != current_kp.angle):
            # If different, add it to the list of unique keypoints
            unique_kps.append(current_kp)

    return unique_kps


def resizeKeypointsToOriginalImageScale(detected_keypoints: List[KeyPoint]) -> List[KeyPoint]:
    """
    Adjust the scale of the detected keypoints to match the scale of the original input image.
    
    Args:
    detected_keypoints (List[KeyPoint]): A list of keypoints detected in the scaled image.

    Returns:
    List[KeyPoint]: A list of keypoints adjusted to the original image scale.
    """

    # Initialize a list to hold the keypoints converted to the original image scale
    scaled_keypoints = []

    for kp in detected_keypoints:
        # Adjust the keypoint position to the original image scale
        kp.pt = tuple(0.5 * array(kp.pt))

        # Adjust the keypoint size to the original image scale
        kp.size *= 0.5

        # Adjust the octave number of the keypoint
        kp.octave = (kp.octave & ~255) | ((kp.octave - 1) & 255)

        # Add the adjusted keypoint to the list
        scaled_keypoints.append(kp)

    return scaled_keypoints


def decodeKeypointOctaveInfo(kp: KeyPoint):
    """
    Decode the octave, layer, and scale information from a keypoint's octave value.

    Args:
    kp (KeyPoint): A keypoint with octave information encoded.

    Returns:
    tuple: A tuple containing the octave, layer, and scale corresponding to the keypoint.
    """

    # Extract the octave information from the keypoint
    octave_info = kp.octave & 255

    # Extract the layer information from the keypoint
    layer_info = (kp.octave >> 8) & 255

    # Adjust the octave value if it's greater than 128
    if octave_info >= 128:
        octave_info = octave_info | -128

    # Calculate the scale based on the octave
    # The scale is inversely proportional to the power of 2 of the octave number
    scale = 1 / float32(1 << octave_info) if octave_info >= 0 else float32(1 << -octave_info)

    return octave_info, layer_info, scale

def createKeypointDescriptors(identified_keypoints: List[KeyPoint], scale_space_images: List[List[np.ndarray]], 
                              descriptor_window: Optional[int] = 4, histogram_bins: Optional[int] = 8, 
                              scale_factor: Optional[int] = 3, max_descriptor_value: Optional[float]=0.2) -> np.ndarray:
    """
    Generate descriptors for each identified keypoint in the scale-space images.

    Args:
    identified_keypoints (List[KeyPoint]): List of keypoints for which descriptors are to be generated.
    scale_space_images (List[List[np.ndarray]]): Gaussian-blurred images at different scales.
    descriptor_window (int): Width of the square window around each keypoint for descriptor calculation.
    histogram_bins (int): Number of orientation bins for the histogram.
    scale_factor (float): Multiplier for determining the actual scale of a keypoint.
    max_descriptor_value (float): Maximum value for clipping the descriptor vector.

    Returns:
    np.ndarray: Array of descriptor vectors for each keypoint.
    """

    descriptors = []
    for kp in identified_keypoints:
        # Decode the keypoint octave information
        octave, layer, kp_scale = decodeKeypointOctaveInfo(kp)
        gaussian_img = scale_space_images[octave + 1, layer]

        # Calculate the descriptor window size and orientation bin properties
        descriptor_radius = int(round(scale_factor * 0.5 * kp_scale * kp.size * sqrt(2) * (descriptor_window + 1) * 0.5))
        descriptor_radius = min(descriptor_radius, int(sqrt(gaussian_img.shape[0] ** 2 + gaussian_img.shape[1] ** 2)))
        bins_per_degree = histogram_bins / 360.
        rotation_angle = 360. - kp.angle
        cos_rot = cos(deg2rad(rotation_angle))
        sin_rot = sin(deg2rad(rotation_angle))
        weight_factor = -0.5 / ((0.5 * descriptor_window) ** 2)
        
        # Initialize lists for histogram calculation
        row_bins, col_bins, magnitudes, orientation_bins = [], [], [], []
        histogram_tensor = zeros((descriptor_window + 2, descriptor_window + 2, histogram_bins))

        # Iterate over the window around the keypoint
        for row_offset in range(-descriptor_radius, descriptor_radius + 1):
            for col_offset in range(-descriptor_radius, descriptor_radius + 1):
                # Calculate rotated window coordinates
                row_rot = col_offset * sin_rot + row_offset * cos_rot
                col_rot = col_offset * cos_rot - row_offset * sin_rot
                row_bin = (row_rot / (scale_factor * kp_scale * kp.size)) + 0.5 * descriptor_window - 0.5
                col_bin = (col_rot / (scale_factor * kp_scale * kp.size)) + 0.5 * descriptor_window - 0.5
                
                # Compute gradients if within image boundaries
                if 0 <= row_bin < descriptor_window and 0 <= col_bin < descriptor_window:
                    window_row = int(round(kp.pt[1] * kp_scale) + row_offset)
                    window_col = int(round(kp.pt[0] * kp_scale) + col_offset)
                    if 0 < window_row < gaussian_img.shape[0] - 1 and 0 < window_col < gaussian_img.shape[1] - 1:
                        dx = decrypt(homomorphicSubtraction(gaussian_img[window_row, window_col + 1], gaussian_img[window_row, window_col - 1])).astype(np.float32) / 100
                        dy = decrypt(homomorphicSubtraction(gaussian_img[window_row - 1, window_col], gaussian_img[window_row + 1, window_col])).astype(np.float32) / 100
                        magnitude = sqrt(dx * dx + dy * dy)
                        orientation = rad2deg(arctan2(dy, dx)) % 360

                        # Accumulate gradient information in histogram bins
                        weight = exp(weight_factor * (row_rot ** 2 + col_rot ** 2))
                        row_bins.append(row_bin)
                        col_bins.append(col_bin)
                        magnitudes.append(weight * magnitude)
                        orientation_bins.append((orientation - rotation_angle) * bins_per_degree)

        # Populate the histogram tensor using trilinear interpolation
        for row_bin, col_bin, magnitude, orientation_bin in zip(row_bins, col_bins, magnitudes, orientation_bins):
            # Perform trilinear interpolation and update histogram tensor
            histogram_tensor = interpolateHistogram(histogram_tensor, row_bin, col_bin, orientation_bin, magnitude, histogram_bins)

        # Flatten and normalize the histogram tensor to create the descriptor
        descriptor_vector = normalizeDescriptor(histogram_tensor, max_descriptor_value)

        # Add the descriptor vector to the list
        descriptors.append(descriptor_vector)

    return array(descriptors, dtype='float32')

def interpolateHistogram(histogram_tensor: np.ndarray, row_bin: float, col_bin: float, orientation_bin: float, 
                         magnitude: float, num_bins: int) -> np.ndarray:
    """
    Helper function to perform trilinear interpolation and update the histogram tensor.

    Args:
    histogram_tensor (np.ndarray): Tensor to be updated with histogram values.
    row_bin, col_bin, orientation_bin (float): Binned coordinates for the gradient.
    magnitude (float): Gradient magnitude to be distributed.
    num_bins (int): Number of orientation bins.

    Returns:
    histogram_tensor (np.ndarray): updated histogram.
    """
    row_bin_floor, col_bin_floor, orientation_bin_floor = floor([row_bin, col_bin, orientation_bin]).astype(int)
    row_fraction, col_fraction, orientation_fraction = row_bin - row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor
    if orientation_bin_floor < 0:
        orientation_bin_floor += num_bins
    if orientation_bin_floor >= num_bins:
        orientation_bin_floor -= num_bins

    c1 = magnitude * row_fraction
    c0 = magnitude * (1 - row_fraction)
    c11 = c1 * col_fraction
    c10 = c1 * (1 - col_fraction)
    c01 = c0 * col_fraction
    c00 = c0 * (1 - col_fraction)
    c111 = c11 * orientation_fraction
    c110 = c11 * (1 - orientation_fraction)
    c101 = c10 * orientation_fraction
    c100 = c10 * (1 - orientation_fraction)
    c011 = c01 * orientation_fraction
    c010 = c01 * (1 - orientation_fraction)
    c001 = c00 * orientation_fraction
    c000 = c00 * (1 - orientation_fraction)

    histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += c000
    histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c001
    histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010
    histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c011
    histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
    histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c101
    histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
    histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c111

    return histogram_tensor


def normalizeDescriptor(histogram_tensor: np.ndarray, max_value: float) -> np.ndarray:
    """
    Helper function to normalize the descriptor vector.

    Args:
    histogram_tensor (np.ndarray): Tensor representing the unnormalized descriptor.
    max_value (float): Maximum allowed value in the normalized descriptor.

    Returns:
    np.ndarray: Normalized descriptor vector.
    """
    descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten()  # Remove histogram borders
    # Threshold and normalize descriptor_vector
    threshold = norm(descriptor_vector) * max_value
    descriptor_vector[descriptor_vector > threshold] = threshold
    descriptor_vector /= max(norm(descriptor_vector), float_tolerance)
    # Multiply by 512, round, and saturate between 0 and 255 to convert from float32 to unsigned char (OpenCV convention)
    descriptor_vector = round(512 * descriptor_vector)
    descriptor_vector[descriptor_vector < 0] = 0
    descriptor_vector[descriptor_vector > 255] = 255

    return descriptor_vector


def HESIFT(img: np.ndarray) -> Tuple[KeyPoint, np.ndarray]:
    baseImage = (rgb2gray(img) * 255).astype(np.int64)
    grayscale_image = (rgb2gray(img) * 255).astype(np.float32)
    baseImage = prepareInitialHESIFTImage(grayscale_image, 2, 0).astype(np.int64)
    num_octaves = computeNumberOfOctaves(baseImage.shape)
    gaussian_kernels = createGaussianBlurKernels(0.4, 3)
    encryptedBaseImage = encryptImage(baseImage)
    encryptedGaussianImages = createScaleSpacePyramid(encryptedBaseImage, num_octaves, gaussian_kernels)
    encryptedDOGImages = createEncryptedDoGPyramid(encryptedGaussianImages)
    keypoints_duplicate = detectScaleSpaceExtremaInEncryptedImages(encryptedGaussianImages,encryptedDOGImages, 3, 0.4, 5)
    keypoints = filterOutDuplicateKeypoints(keypoints_duplicate)
    keypoints = resizeKeypointsToOriginalImageScale(keypoints)
    descriptors = createKeypointDescriptors(keypoints, encryptedGaussianImages)
    return keypoints, descriptors