import numpy as np
import scipy.special as ss

def frequency_within_block_test(binary, M=128):
    """ Test the proportion of 1s in each block of size M. Should be around 2/M if numbers are random.
        Equivalent to the monobit test if M=1.
    Args:
        input (binary): binary data
        n (int): size of total binary string
        M (int, optional): Block size. Defaults to 128.
    """

    bytes = binary.packed
    nBlocks = binary.n // M
    bytesPerBlock = M // 8

    _bytes = bytes[:nBlocks*bytesPerBlock].reshape(nBlocks, bytesPerBlock)
    proportions = np.count_nonzero(
        np.unpackbits(_bytes, axis=1), axis=1
    ) / M

    chisq = np.sum(4.0 * M * ((proportions - 0.5) ** 2))
    p = ss.gammaincc((nBlocks/2.0), float(chisq)/2.0)
    success = (p >= 0.01)

    return [p, success]