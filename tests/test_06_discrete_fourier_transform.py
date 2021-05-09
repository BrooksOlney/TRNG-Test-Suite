import numpy as np
import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool
import scipy.fft as fft
import scipy.special as ss
from scipy.stats import chisquare
import math

def discrete_fourier_transform_test(binary, m=1):

    bits = 2*binary.unpacked.astype(np.int8) - 1

    # two-level test for bitstreams that are too large
    if binary.n > 10_000_000:
        m = 1000
    
    blockSize = binary.n // m
    blocks = bits.reshape(m, blockSize)
    T = math.sqrt(math.log(1.0/0.05)*blockSize)
    
    with ThreadPool(mp.cpu_count()) as p:
        s = p.map(vectorized_fft, blocks)

    N0 = 0.95 * blockSize / 2.0
    N1 = np.array([len(np.where(si < T)[0]) for si in s])

    d = (N1 - N0)/math.sqrt((blockSize*0.95*0.05)/4)

    # compute proportion of p-values that pass
    p = np.array([math.erfc(abs(di)/math.sqrt(2)) for di in d])
    alpha = 0.01
    mean  = 1.0 - alpha
    std   = math.sqrt((alpha*mean / m))

    prop = len(np.where(p >= alpha)[0]) / m
    cilow = mean - 3*std
    cihigh = mean + 3*std

    # check if passes proportion test
    propPass = prop >= cilow and prop <= cihigh

    # perform chisq test on computed p-values
    chsq, uniformityp = chisquare(p, ddof=10)
    pt = ss.gammainc(10/2,chsq/2)

    success = (p >= 0.01)

    return [prop, propPass, pt, pt >= 0.0001]

def vectorized_fft(x):
    return np.abs(fft.rfft(x)[:len(x) // 2]).astype(np.float32)