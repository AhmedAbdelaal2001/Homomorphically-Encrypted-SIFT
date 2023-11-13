from utils_encryptedDomain.cryptosystem import *

def homomorphicScalarMultiplication(c, p):
    result = np.zeros(c.shape).astype(np.uint64)
    mask = (p >= n - m)
    if mask.any():
        result[mask] = fastPowering_matrixBaseAndExponent(findInverse(c[mask]), (n - p)[mask], n_sq)
        result[~mask] = fastPowering_matrixBaseAndExponent(c[~mask], p[~mask], n_sq)
    else: result = fastPowering_matrixBaseAndExponent(c, p, n_sq)
    return result

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
    terms = homomorphicScalarMultiplication(encryptedTensor, encodedTensor)
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
