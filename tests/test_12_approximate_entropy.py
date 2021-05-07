import numpy as np
import math
import scipy.special as ss


def approximate_entropy_test(binary):
    """ As with the Serial test, the focus of this test is the
        frequency of all possible overlapping m-bit patterns across
        the entire sequence.
    """
    bits = binary.unpacked
    n = binary.n
    # M    = math.floor(np.log2(n)) - 5
    M = 10

    # mbits = np.concatenate([bits, bits[:M-1]])
    # m1bits = np.concatenate([bits, bits[:M]])

    mblocks = np.packbits(
        np.lib.stride_tricks.as_strided(
            np.concatenate([bits, bits[:M-1]]), (n, M), (1, 1), writeable=False
        ), axis=1
    ).view(np.uint16).reshape(-1)
    m1blocks = np.packbits(
        np.lib.stride_tricks.as_strided(
            np.concatenate([bits, bits[:M]]), (n, M+1), (1, 1), writeable=False
        ), axis=1
    ).view(np.uint16).reshape(-1)
    
    _, mcounts = np.unique(mblocks, return_counts=True)
    _, m1counts = np.unique(m1blocks, return_counts=True)

    mpsi = np.sum(mcounts * np.log(mcounts)) / n
    m1psi = np.sum(m1counts * np.log(m1counts)) / n

    chisq = 2*n*(math.log(2) - (mpsi - m1psi))

    p = ss.gammaincc(2**(M-1), chisq/2)

    success = (p >= 0.01)

    return [p, success]

def vectorized_sliding_window(x):
    print("hi")