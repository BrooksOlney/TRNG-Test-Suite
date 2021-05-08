import numpy as np
import scipy.special as ss

def serial_test(binary):
    """ The focus of this test is the frequency of all possible overlapping m-bit
        patterns across the entire sequence.

    """

    bits = binary.unpacked
    n = binary.n
    M = 16

    bits = np.concatenate([bits, bits[:M-1]])

    psisqs = []
    for i in range(3):
        m = M - i
        nrows = ((bits.size - m)) + 1
        _n = bits.strides[0]
        mpattern = np.packbits(
            np.lib.stride_tricks.as_strided(
                bits, shape=(nrows, m), strides=(_n, _n)
            ), axis=1
        ).view(np.uint16).reshape(-1)

        # get count of occurrences, compute psisq for that block size
        mcounts = np.bincount(mpattern)
        psisq = ((2**(M-i)) / n) * sum(mcounts**2) - n
        psisqs.append(psisq)

    dpsi = psisqs[0] - psisqs[1]
    d2psi = psisqs[0] - 2*psisqs[1] + psisqs[2]

    p1 = ss.gammainc(2**(M-2), dpsi/2)
    p2 = ss.gammainc(2**(M-3), d2psi/2)

    success1 = p1 >= 0.01
    success2 = p2 >= 0.01

    return [p1, success1, p2, success2]

def vectorized_sliding_window(x, n, m):
    
    bitsPerJob = 1_000_000
    r = math.ceil(n // bitsPerJob)

    micounts = np.zeros(2**(16))

    for i in range(r):
        
        _x = x[i*bitsPerJob:(i+1)*bitsPerJob]
        mblocks = np.packbits(
            np.lib.stride_tricks.as_strided(
                _x, (len(_x), m), (1,1)
            ), axis=1
        ).view(np.uint16).reshape(-1)

        counts = np.bincount(mblocks)
        micounts[range(counts.size)] += counts

    return micounts

def sliding_window(x, m):
    
    micounts = np.zeros(2**(16))
        
    mblocks = np.packbits(
        np.lib.stride_tricks.as_strided(
            x, (len(x), m), (1,1)
        ), axis=1
    ).view(np.uint16).reshape(-1)

    counts = np.bincount(mblocks)
    micounts[range(counts.size)] += counts

    return micounts