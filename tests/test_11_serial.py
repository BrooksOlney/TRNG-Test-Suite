import numpy as np
import scipy.special as ss
import multiprocessing as mp
import math
from functools import partial
from multiprocessing.dummy import Pool as ThreadPool

def serial_test(binary):
    """ The focus of this test is the frequency of all possible overlapping m-bit
        patterns across the entire sequence.

    """

    bits = binary.unpacked
    n = binary.n
    M = 16

    psisqs = []

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
    strides = np.lib.stride_tricks.sliding_window_view(x, window_shape=m)
    mask = np.array(1 << np.arange(m), dtype=np.uint16)[::-1]
    
    strides = np.array_split(strides, math.ceil(len(strides) / 1_000_000))
    
    with ThreadPool(mp.cpu_count()) as p:
        micounts = np.vstack([*p.imap(partial(convert_binary, mask=mask), strides)])    

    # for s in np.array_split(strides, math.ceil(len(strides) / 1_000_000)):
    #     repacked = s @ mask
    #     count = np.bincount(repacked)
    #     micounts[range(len(count))] += count

    # mblocks = np.packbits(strides, axis=1).view(np.uint16).reshape(-1)
    # counts = np.bincount(mblocks)
    # micounts[range(counts.size)] += counts
    # return micounts
    return np.sum(micounts, axis=0)

def convert_binary(x, mask):
    repacked = x @ mask
    micounts = np.zeros(2**(16))
    counts = np.bincount(repacked)
    micounts[range(counts.shape[0])] = counts
    return micounts