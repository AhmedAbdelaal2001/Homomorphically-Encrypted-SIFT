from phe import paillier
import numpy as np
import concurrent.futures

key_size = 50
public_key, private_key = paillier.generate_paillier_keypair(n_length=key_size)


def encrypt_element(x):
    return public_key.encrypt(int(x))

def decrypt_element(x):
    if isinstance(x,paillier.EncryptedNumber):
        return private_key.decrypt(x)
    return private_key.raw_decrypt(x)

def extract_ciphertext(enc_num):
    return enc_num.ciphertext() % 256

def encrypt_tensor(tensor):
    encrypt_func = np.vectorize(encrypt_element)  # Vectorize the encryption function
    return encrypt_func(tensor)

def decrypt_tensor(encrypted_tensor):
    decrypt_func = np.vectorize(decrypt_element)  # Vectorize the decryption function
    return decrypt_func(encrypted_tensor)

def extract_encrypted_image(enc_image_raw):
    vectorized_extract = np.vectorize(extract_ciphertext)
    return vectorized_extract(enc_image_raw)


def cipher_addition(c1,c2): #c2 can be another ciphered text or a plain text (image or single element)
    return c1.__add__(c2)

def cipher_multiplication(c1,c2): #c2 must be scalar (image or single element)
    return c1.__mul__(c2)


def convolve_encrypted_image(encrypted_image, filter):
    result = np.zeros_like(encrypted_image)
    filter = np.array([[int(val) for val in row] for row in filter], dtype=object)
    image_height, image_width = encrypted_image.shape
    filter_height, filter_width = filter.shape
    n_sq = public_key.nsquare
    for x in range(image_height):
        for y in range(image_width):
            convolution_result = 1  # Initialize the result for the current position
            for u in range(filter_height):
                for v in range(filter_width):
                    x_u = x - u
                    y_v = y - v
                    if 0 <= x_u < image_height and 0 <= y_v < image_width:
                        
                        convolution_result *= cipher_multiplication(encrypted_image[x_u, y_v],filter[u, v]).ciphertext()
            result[x, y] = convolution_result

    return result








def convolve_encrypted_image_optimized(encrypted_image, filter):
    image_height, image_width = encrypted_image.shape
    filter_height, filter_width = filter.shape
    filter = np.array([[int(val) for val in row] for row in filter], dtype=object)
    n_sq = public_key.nsquare
    
    def convolve_pixel(x, y):
        convolution_result = 1  # Initialize the result for the current position
        
        for u in range(filter_height):
            for v in range(filter_width):
                x_u = x - u
                y_v = y - v
                
                if 0 <= x_u < image_height and 0 <= y_v < image_width:
                    convolution_result *= cipher_multiplication(encrypted_image[x_u, y_v], filter[u, v]).ciphertext()
        
        return convolution_result
    
    # Use concurrent.futures for parallelization
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(convolve_pixel, x, y) for x in range(image_height) for y in range(image_width)]
    
    # Collect results
    result = np.zeros_like(encrypted_image)
    for x in range(image_height):
        for y in range(image_width):
            result[x, y] = futures[x * image_width + y].result()
    
    return result
