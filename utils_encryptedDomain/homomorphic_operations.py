from utils_encryptedDomain.cryptosystem import *

def homomorphicAddition(ciphertext1, ciphertext2):
    return (ciphertext1 * ciphertext2) % n_sq

def homomorphicScalarMultiplication(ciphertext, scalar):
    if scalar >= n - m:
        return fastPowering(findInverse(ciphertext), n - scalar, n_sq)
    else: 
        return fastPowering(ciphertext, scalar, n_sq)

def homomorphicSubtraction(ciphertext1, ciphertext2):
    return homomorphicAddition(ciphertext1, homomorphicScalarMultiplication(ciphertext2, encode(-1)))

#------------------------------------------------------------------------------------------------------

def tensor_homomorphicAddition(tensor_ciphertext1, tensor_ciphertext2):
    return (tensor_ciphertext1 * tensor_ciphertext2) % n_sq

def tensor_homomorphicScalarMultiplication(tensor_ciphertext, tensor_scalar):
    result = np.zeros(tensor_ciphertext.shape).astype(np.uint64)
    mask = (tensor_scalar >= n - m)
    if mask.any():
        result[mask] = fastPowering_matrixBaseAndExponent(tensor_findInverse(tensor_ciphertext[mask]), (n - tensor_scalar)[mask], n_sq)
        result[~mask] = fastPowering_matrixBaseAndExponent(tensor_ciphertext[~mask], tensor_scalar[~mask], n_sq)
    else: result = fastPowering_matrixBaseAndExponent(tensor_ciphertext, tensor_scalar, n_sq)
    return result

def tensor_homomorphicSubtraction(tensor_ciphertext1, tensor_ciphertext2):
    return tensor_homomorphicAddition(tensor_ciphertext1, tensor_homomorphicScalarMultiplication(tensor_ciphertext2, encodeImage(np.full_like(tensor_ciphertext2, -1, dtype=np.int64))))

#--------------------------------------------------------------------------------------------------------

def encryptedInnerProduct(encryptedTensor, encodedTensor):
    """
    Computes the inner (dot) product of two encrypted numpy arrays in the ciphertext domain. In the plaintext domain,
    the inner product is computed by multiplying the numpy arrays elementwise, then summing the products.
    In the encrypted domain, the inner product will be a product of modular exponentiations.

    Inputs:
    - tensor1: Numpy Array
    - tensor2: Numpy Array

    Output:
    - A single number of type np.uint64 representing the encrypted inner product of the two numpy arrays
    """
    terms = tensor_homomorphicScalarMultiplication(encryptedTensor, encodedTensor)
    innerProduct = np.uint64(1)
    # Note: A for loop was used instead of np.prod to ensure that no overflow occurs. This creates a slight overhead.
    for term in np.nditer(terms):
        innerProduct = (innerProduct * term) % n_sq
    return innerProduct

def process_segment(args):
    encryptedImageSegment, encodedKernel, x_start, y_start, xKernShape, yKernShape, strides, padding = args
    segment_output = np.zeros((encryptedImageSegment.shape[0] - xKernShape + 1, encryptedImageSegment.shape[1] - yKernShape + 1), dtype=np.uint64)

    for x in range(0, encryptedImageSegment.shape[0] - xKernShape + 1, strides):
        for y in range(0, encryptedImageSegment.shape[1] - yKernShape + 1, strides):
            conv_value = encryptedInnerProduct(encryptedImageSegment[x: x + xKernShape, y: y + yKernShape], encodedKernel)
            segment_output[x // strides, y // strides] = conv_value

    return (x_start, y_start, segment_output)

def encryptedConvolve2D(encryptedImage, kernel, padding=0, strides=1):
    kernel = np.flipud(np.fliplr(kernel))
    encodedKernel = encodeImage(kernel)

    xKernShape, yKernShape = encodedKernel.shape
    xImgShape, yImgShape = encryptedImage.shape
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput), dtype=np.uint64)

    imagePadded = np.pad(encryptedImage, ((padding, padding), (padding, padding)), mode='constant', constant_values=1) if padding != 0 else encryptedImage

    # Splitting the image into segments for parallel processing
    num_processes = cpu_count()
    # Splitting the image into segments with overlap
    segment_height = imagePadded.shape[0] // num_processes
    overlap = xKernShape - 1
    segments = []
    for i in range(num_processes):
        x_start = max(i * segment_height - overlap, 0)
        x_end = min((i + 1) * segment_height + overlap, imagePadded.shape[0]) if i != num_processes - 1 else imagePadded.shape[0]
        segment = imagePadded[x_start:x_end, :]
        segments.append((segment, encodedKernel, x_start, 0, xKernShape, yKernShape, strides, padding))

    # Parallel processing
    with Pool(num_processes) as pool:
        results = pool.map(process_segment, segments)

    # Combining results
    for segment_result in results:
        x_start, y_start, segment = segment_result
        for x in range(segment.shape[0]):
            for y in range(segment.shape[1]):
                output[x + x_start, y + y_start] = segment[x, y]

    return output
