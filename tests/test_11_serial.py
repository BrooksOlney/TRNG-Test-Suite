import numpy as np
import scipy.special as ss
import multiprocessing as mp
import math
from functools import partial

def serial_test(binary):
    """ The focus of this test is the frequency of all possible overlapping m-bit
        patterns across the entire sequence.

    """

    bits = binary.unpacked
    n = binary.n
    M = 16
    bitsPerJob = 1_000_000
    r = math.ceil((n + M) // bitsPerJob) 

    psisqs = []
    
    if r > mp.cpu_count():
        with mp.Pool(mp.cpu_count()) as p:
            for j in range(3):
                m = M - j
                _bits = np.concatenate([bits, bits[:m - 1]])
                _bits = [bits[i*bitsPerJob : (i+1)*bitsPerJob + m] for i in range(r)]

                mcounts = np.sum([*p.imap(partial(sliding_window, m=m), _bits)], axis=0)
            
                psisq = ((2**(M-j)) / n) * sum(mcounts**2) - n
                psisqs.append(psisq)
    else:
        for j in range(3):
            m = M - j

            mcounts = sliding_window(bits, m)
        
            psisq = ((2**(M-j)) / n) * sum(mcounts**2) - n
            psisqs.append(psisq)

    dpsi  = psisqs[0] - psisqs[1]
    d2psi = psisqs[0] - 2*psisqs[1] + psisqs[2]

    p1 = ss.gammainc(2**(M-2), dpsi/2)
    p2 = ss.gammainc(2**(M-3), d2psi/2)

    success1 = p1 >= 0.01
    success2 = p2 >= 0.01

    return [p1, success1, p2, success2]

def sliding_window(x, m):
    micounts = np.zeros(2**(16))
    s1 = np.lib.stride_tricks.as_strided(x, (len(x) - m + 1, m), (x.itemsize, x.itemsize))
    mblocks = np.packbits(s1, axis=1).view(np.uint16).reshape(-1)
    counts = np.bincount(mblocks)
    micounts[range(counts.size)] += counts

    return micounts