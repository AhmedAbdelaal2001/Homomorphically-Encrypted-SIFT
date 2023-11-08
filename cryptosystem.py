# Functions used for encryption, decryption, and convolution in the encrypted domain can all be found in this file.

import numpy as np
import random

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
    a = g
    b = np.ones(A.shape).astype(np.uint64)
    while(np.sum(A) != 0):
        b[A % 2 == 1] = (b[A % 2 == 1] * a[A % 2 == 1]) % N
        a = (a * a) % N
        A = A // 2
    return b

def encrypt(plaintext):
    """
    This function encrypts a single plaintext value.

    Input:
    - plaintext: np.uint64

    Output:
    - ciphertext corresponding to the given plaintext.
    """
    return (fastPowering(g, plaintext, n_sq) * fastPowering(generate_random(), n, n_sq)) % n_sq

def decrypt(ciphertext):
    """
    This function decrypts a single ciphertext value.

    Input:
    - ciphertext: np.uint64

    Output:
    - plaintext corresponding to the given ciphertext.
    """
    return (((fastPowering(ciphertext, lam, n_sq) - 1) // n) * mu) % n

def encryptImage(image):
    """
    This function encrypts an entire image, its operations utilize numpy's vectorized operations for maximum efficiency.

    Input:
    - image: Numpy array, where every value represents a pixel brightness between 0-255

    Output:
    - encrypted image corresponding to the given image.
    """
    return (fastPowering_matrixExponent(g, image, n_sq) * fastPowering(generate_random_tensor(image.shape), n, n_sq)) % n_sq

def decryptImage(encryptedImage):
    """
    This function decrypts an entire image, its operations utilize numpy's vectorized operations for maximum efficiency.

    Input:
    - encryptedImage: Numpy array, where every value represents a pixel brightness between 0-255

    Output:
    - Original image corresponding to the given encryptedImage.
    """
    return (((fastPowering(encryptedImage, lam, n_sq) - 1) // n) * mu) % n

def encryptedInnerProduct(tensor1, tensor2):
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
    terms = fastPowering_matrixBaseAndExponent(tensor1, tensor2, n_sq)
    innerProduct = np.uint64(1)
    # Note: A for loop was used instead of np.prod to ensure that no overflow occurs. This creates a slight overhead.
    for term in np.nditer(terms):
        innerProduct = (innerProduct * np.uint64(term)) % n_sq
    return innerProduct

def encryptedConvolve2D(image, kernel, padding=0, strides=1):
    """
    Applies a convolution between an encrypted image and a kernel in the ciphertext domain. This is done by
    trying out the possible shifts for the kernel, and for each shift, compute the encryptedInnerProduct between
    the kernel and a section of the encrypted image.

    Inputs:
    - Image: A numpy array containing the encrypted image
    - Kernel: The filter we wish to apply. This value is not encrypted.
    - Padding: The number of zeros we wish to add in each direction.
    - Strides: The step size taken when convolving the filter.

    Output:
    - A numpy array representing the encrypted form of the convolution result
    """
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
    output = np.zeros((xOutput, yOutput)).astype(np.uint64)

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = encryptedInnerProduct(imagePadded[x: x + xKernShape, y: y + yKernShape], kernel)
                except:
                    break

    return output