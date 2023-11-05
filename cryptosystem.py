from phe import paillier
import numpy as np

key_size = 50
public_key, private_key = paillier.generate_paillier_keypair(n_length=key_size)

def encrypt_element(x):
    return public_key.encrypt(int(x))

def decrypt_element(x):
    return private_key.decrypt(x)

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
