import numpy as np
import gmpy2
from gmpy2 import mpz
import random

def L(u, n):
    return (u-1)//n

def generate_keys(bit_length):
    p = gmpy2.next_prime(random.getrandbits(bit_length // 2))
    q = gmpy2.next_prime(random.getrandbits(bit_length // 2))

    n = p * q
    lam = gmpy2.lcm(p-1, q-1)
    n_sq = n * n
    g = n + 1
    mu = gmpy2.invert(L(gmpy2.powmod(g, lam, n_sq), n), n)

    public_key = (n, g)
    private_key = (lam, mu)

    return public_key, private_key

def encrypt(public_key, plaintext):
    n, g = public_key
    n_sq = n * n
    r = mpz(random.randrange(1, n))
    c = gmpy2.powmod(g, plaintext, n_sq) * gmpy2.powmod(r, n, n_sq) % n_sq
    return c

def decrypt(private_key, public_key, ciphertext):
    n, g = public_key
    lam, mu = private_key
    n_sq = n * n
    u = gmpy2.powmod(ciphertext, lam, n_sq)
    l_u = (u - 1) // n
    plaintext = (l_u * mu) % n
    return plaintext

def encrypt_tensor(public_key, tensor):
    shape = tensor.shape
    flattened_encrypted = [encrypt(public_key, mpz(element)) for element in tensor.flatten()]
    return np.array(flattened_encrypted).reshape(shape)

def decrypt_tensor(private_key, public_key, encrypted_tensor):
    shape = encrypted_tensor.shape
    flattened_decrypted = [decrypt(private_key, public_key, mpz(element)) for element in encrypted_tensor.flatten()]
    return np.array(flattened_decrypted).reshape(shape)

def image_mpz_to_int(image):
    return np.vectorize(lambda x: int(x))(image)