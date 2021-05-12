import numpy as np
import math
import scipy.special as ss
import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool
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

    mcounts = sliding_window(mbits, M)
    m1counts = sliding_window(m1bits, M+1)

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
    # s1 = np.lib.stride_tricks.as_strided(x, (len(x) - m + 1, m), (x.itemsize, x.itemsize))
    strides = np.lib.stride_tricks.sliding_window_view(x, window_shape=m)
    mask = np.array(1 << np.arange(m), dtype=np.uint16)[::-1]

    strides = np.array_split(strides, math.ceil(len(strides) / 1_000_000))

    with ThreadPool(mp.cpu_count()) as p:
        micounts = np.vstack([*p.imap(partial(convert_binary, mask=mask), strides)])
    # for s in np.array_split(strides, math.ceil(len(strides) / 10_000_000)):
    #     repacked = s @ mask
    #     count = np.bincount(repacked)
    #     micounts[range(len(count))] += count
    return np.sum(micounts, axis=0)

def convert_binary(x, mask):
    repacked = x @ mask
    micounts = np.zeros(2**(16))
    counts = np.bincount(repacked)
    micounts[range(counts.shape[0])] = counts
    return micounts