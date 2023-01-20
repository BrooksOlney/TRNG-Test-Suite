from multiprocessing.dummy import Pool as ThreadPool
from itertools import repeat

import numpy as np
import multiprocessing as mp
import scipy.special as ss

def overlapping_template_matching_test(binary, m=9, K=5):

        def get_prob(u, x):
            out = 1.0 * np.exp(-x)
            if u != 0:
                out = 1.0 * x * np.exp(2 * -x) * (2 ** -u) * ss.hyp1f1(u + 1, 2, x)
            return out

        bits = binary.unpacked
        n = len(bits)
        M =  1032
        N = n // M

        blocks = bits[:N*M].reshape(N,M)

        template = np.ones(m, dtype=np.uint8)
        template = np.dot(template, 1 << np.arange(m, dtype=np.uint16)[::-1])

        matches = overlapping_matches(blocks, m, template)
        lmbda = (M-m+1)/(2**m)
        nu = lmbda/2

        vs = np.array([len(matches[matches == i]) for i in range(6)])
        vs[5] = len(matches[matches >= 5])
        # pis = np.array([0.364091, 0.185659, 0.139381, 0.100571, 0.070432, 0.139865])
        pis = [get_prob(i, nu) for i in range(5)]
        pis = np.array([*pis, 1-sum(pis)])
        
        chisq = sum((vs - N*pis)**2 / (N*pis))

        p = ss.gammaincc(K/2, chisq/2)

        success = (p >= 0.01)
        
        return [p, success]

def overlapping_matches(block, m, template):
    strides = np.lib.stride_tricks.sliding_window_view(block, window_shape=m, axis=1)
    mask = np.array(1 << np.arange(m), dtype=np.uint16)[::-1]

    with ThreadPool(mp.cpu_count()) as p:
        repacked = np.array([*p.starmap(np.matmul, zip(strides, repeat(mask)))])

    return np.count_nonzero(repacked == template, axis=1)