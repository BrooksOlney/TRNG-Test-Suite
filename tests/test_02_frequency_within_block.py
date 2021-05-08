import numpy as np
import scipy.special as ss

def frequency_within_block_test(binary, M=128):
    """ Test the proportion of 1s in each block of size M. Should be around 2/M if numbers are random.
        Equivalent to the monobit test if M=1.
    Args:
        input (binary): binary data
        M (int, optional): Block size. Defaults to 128.
    """

    bytes = binary.packed.view(np.uint32)
    nBlocks = binary.n // M
    bytesPerBlock = M // 32

    # same binary tricks to compute popcount applied to numpy array
    _bytes = bytes[:nBlocks*bytesPerBlock].reshape(nBlocks, bytesPerBlock)
    _bytes = _bytes - ((_bytes >> 1) & 0x55555555)
    _bytes = (_bytes & 0x33333333) + ((_bytes>> 2) & 0x33333333)
    _bytes = (((_bytes + (_bytes >> 4) & 0xF0F0F0F) * 0x1010101) & 0xffffffff) >> 24
    proportions = np.sum(_bytes, axis=1) / M

    chisq = np.sum(4.0 * M * ((proportions - 0.5) ** 2))
    p = ss.gammaincc((nBlocks/2.0), float(chisq)/2.0)
    success = (p >= 0.01)

    return [p, success]