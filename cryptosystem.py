# Functions used for encryption, decryption, and convolution in the encrypted domain can all be found in this file.

import numpy as np
import random
from multiprocessing import Pool, cpu_count

"""
The following values for the cryptosystem parameters were specifically chosen so that they would satisfy the conditions
necessary for the cryptosystem to work, while ensuring that overflows would never happen; by making sure that any
primitive operation we perform in the encrypted domain outputs a number that can be stored in the numpy.uint64 data type.

Note: All values were converted from int to numpy.uint64, to prevent the Python compiler from implicitly casting
those numbers to floats in certain sections of the code.
"""
p = np.uint64(251)
q = np.uint64(257)
n = np.uint64(64507)
n_sq = np.uint64(4161153049)
lam = np.uint64(32000)
g = np.uint64(2189576534)
mu = np.uint64(28964)
m = np.int64(32253)

def generate_random():
    """
    This function generates a random number r such that 1<r<n and gcd(r,n) == 1.

    Output:
    - The random integer r 
    """
    while True:
        r = random.randint(1, n - 1)
        if r % p != 0 and r % q != 0:  # r should be coprime to n
            return r

def generate_random_tensor(shape):
    """
    This function is a direct extension to the generate_random() function; it takes in a numpy array shape,
    and generates a numpy array of that shape where every element is a random number r such that 1<r<n and
    gcd(r, n) == 1

    Input:
    - A numpy array shape, specifying the shape of the random numpy array to be generated
    
    Output
    - A numpy array of the same shape given in the input containing random integers.
    """
    # Initialize an empty array of the given shape
    r_tensor = np.empty(shape, dtype=np.uint64)
    
    # Iterate over the array indices
    for index, _ in np.ndenumerate(r_tensor):
        while True:
            # Generate a random number
            r = random.randint(1, n - 1)
            # If it's coprime to n, assign it to the current position
            if r % p != 0 and r % q != 0:
                r_tensor[index] = r
                break

    return r_tensor

def fastPowering(g, A, N):
    """
    This function computes g^A (mod N) using the square-and-multiply algorithm, which runs in logarithmic time.
    (remember the algorithm used in the Discrete Mathematics course to calculate modular exponentiations).

    Inputs:
    - A: np.uint64
    - N: np.uint64
    - g: can be an integer or a numpy array. In the case of a numpy array, this function will execute the operation elementwise.
    
    Output:
    - if g is an integer, the output is an integer between 0 and N-1 that is congruent to g^A (mod N)
    - if g is a numpy array, the output is another numpy array where every element is a modular exponentiation
    """
    a = np.uint64(g)
    b = np.uint64(1)
    while A > 0:
        if (A % 2) == 1: b = (b * a) % N
        a = (a * a) % N
        A = A // 2
    return b

def fastPowering_matrixExponent(g, A, N):
    """
    Similar to the previous function, but the base is an integer and the exponent is a numpy array.

    Inputs:
    - g: np.uint64
    - N: np.uint64
    - A: A numpy array

    Output:
    - A single numpy array where every element is equal to g^A_ij (mod n)
    """
    a = g
    b = np.ones(A.shape).astype(np.uint64)
    while(np.sum(A) != 0):
        b[A % 2 == 1] = (b[A % 2 == 1] * a) % N
        a = (a * a) % N
        A = A // 2
    return b

def fastPowering_matrixBaseAndExponent(g, A, N):
    """
    A combination of the previous two functions; this one takes in two numpy array g and A of the same shape,
    as well as an integer N, and computes g ^ A (mod N). This calculation is done elementwise. That is, the output
    value at position (i, j) is equal to (g_ij)^(A_ij) (mod N)

    Inputs:
    - g: Numpy array
    - A: Numpy array
    - N: np.uint64

    Outputs:
    - A single numpy array of the same shape as the inputs where the output at position (i, j) is equal to
    (g_ij)^(A_ij) (mod N)
    """
    a = g.copy()
    b = np.ones_like(A, dtype=np.uint64)
    A = A.copy()

    # Use bitwise operations for efficiency
    while np.any(A != 0):
        mask = A % 2 == 1
        b[mask] = (b[mask] * a[mask]) % N
        a = (a * a) % N
        A >>= 1  # Equivalent to A = A // 2 but more efficient

    return b

def encode(number):
    return number % n

def decode(encodedNumber):
    decodedNumber = encodedNumber
    if decodedNumber > n // 2: decodedNumber -= n
    return decodedNumber

def encrypt(plaintext):
    """
    This function encrypts a single plaintext value.

    Input:
    - plaintext: np.int64

    Output:
    - ciphertext corresponding to the given plaintext.
    """
    encodedPlaintext = encode(plaintext)
    return (fastPowering(g, encodedPlaintext, n_sq) * fastPowering(generate_random(), n, n_sq)) % n_sq

def decrypt(ciphertext):
    """
    This function decrypts a single ciphertext value.

    Input:
    - ciphertext: np.uint64

    Output:
    - plaintext corresponding to the given ciphertext.
    """
    encodedDecryptedText = ((((fastPowering(ciphertext, lam, n_sq) - 1) // n) * mu) % n).astype(np.int64)
    return decode(encodedDecryptedText)

def encodeImage(image):
    return (image % n).astype(np.uint64)

def decodeImage(encodedImage):
    decodedImage = encodedImage.astype(np.int64)
    decodedImage[decodedImage > n // 2] -= n
    return decodedImage

def encryptImage(image):
    """
    This function encrypts an entire image, its operations utilize numpy's vectorized operations for maximum efficiency.

    Input:
    - image: Numpy array, where every value represents a pixel brightness between 0-255

    Output:
    - encrypted image corresponding to the given image.
    """
    encodedImage = encodeImage(image)
    return (fastPowering_matrixExponent(g, encodedImage, n_sq) * fastPowering(generate_random_tensor(encodedImage.shape), n, n_sq)) % n_sq

def decryptImage(encryptedImage):
    """
    This function decrypts an entire image, its operations utilize numpy's vectorized operations for maximum efficiency.

    Input:
    - encryptedImage: Numpy array, where every value represents a pixel brightness between 0-255

    Output:
    - Original image corresponding to the given encryptedImage.
    """
    encodedDecryptedImage = (((fastPowering(encryptedImage, lam, n_sq) - 1) // n) * mu) % n
    return decodeImage(encodedDecryptedImage)

def findInverse(a):
    u = np.ones_like(a, dtype=np.int64)
    g = a.astype(np.int64)
    x = np.zeros_like(a, dtype=np.int64)
    y = np.full_like(a, n_sq, dtype=np.int64)

    while np.any(y):
        mask = y != 0
        q, t = np.divmod(g[mask], y[mask])
        s = u[mask] - q * x[mask]
        u[mask], x[mask] = x[mask], s
        g[mask], y[mask] = y[mask], t

    return np.mod(u, n_sq).astype(np.uint64)

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

    imagePadded = np.pad(encryptedImage, ((padding, padding), (padding, padding)), mode='constant', constant_values=0) if padding != 0 else encryptedImage

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
