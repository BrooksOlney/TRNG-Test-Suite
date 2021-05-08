import numpy as np
import math
import scipy.special as ss
import multiprocessing as mp
from functools import partial

def approximate_entropy_test(binary):
    """ As with the Serial test, the focus of this test is the
        frequency of all possible overlapping m-bit patterns across
        the entire sequence.
    """
    bits = binary.unpacked
    n = binary.n
    # M    = math.floor(np.log2(n)) - 5
    M = 10

    mbits  = np.concatenate([bits, bits[:M-1]])
    m1bits = np.concatenate([bits, bits[:M]])

    bitsPerJob = 1_000_000

    lm  = n + M
    lm1 = n + M + 1
    
    r = math.ceil(lm // bitsPerJob) 
    
    mbits = [mbits[i*bitsPerJob : (i+1)*bitsPerJob + M - 1] for i in range(r)]
    m1bits = [m1bits[i*bitsPerJob : (i+1)*bitsPerJob + M] for i in range(r)]

    with mp.Pool(mp.cpu_count()) as p:
        mcounts = np.sum([*p.imap(partial(sliding_window, m=M), mbits)], axis=0)
        m1counts = np.sum([*p.imap(partial(sliding_window, m=M+1), m1bits)], axis=0)

    mcounts[mcounts == 0] = 1
    m1counts[m1counts == 0] = 1

    mpsi = np.sum(mcounts * np.log(mcounts)) / n
    m1psi = np.sum(m1counts * np.log(m1counts)) / n

    chisq = 2*n*(math.log(2) - (mpsi - m1psi))

    p = ss.gammaincc(2**(M-1), chisq/2)

    success = (p >= 0.01)

    return [p, success]


def sliding_window(x, m):
    
    micounts = np.zeros(2**(16))
    s1 = np.lib.stride_tricks.as_strided(x, (len(x) - m + 1, m), (x.itemsize, x.itemsize))
    mblocks = np.packbits(s1, axis=1).view(np.uint16).reshape(-1)
    counts = np.bincount(mblocks)
    micounts[range(counts.size)] += counts

    return micounts