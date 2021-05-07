import numpy as np
import multiprocessing as mp
import scipy.special as ss
import itertools as it

def non_overlapping_template_matching_test(binary, B=1, m=9):
    
    def template_matches(block, template):
        strides = np.packbits(np.lib.stride_tricks.as_strided(block, shape=((block.size - m + 1), m), strides=(1,1), writeable=False), axis=1).view(np.uint16).reshape(-1)
        inds    = np.where(strides == template)[0]
        dists   = np.diff(inds)

        return len(inds) - np.count_nonzero(dists < m)

    bits = binary.unpacked
    n = len(bits)
    M = n // 8
    N = n // M

    blocks = bits[:N*M].reshape(N,M)
    template = np.unpackbits(np.array(B, dtype=np.uint8))
    
    if len(template) > m:
        template = template[len(template) - m:]
    elif len(template) < m:
        template = np.concatenate([np.zeros((m - len(template)), dtype=np.uint8), template])

    template = np.packbits(template).view(np.uint16)

    with mp.Pool(mp.cpu_count()) as p:
        matches = np.array(p.starmap(non_overlapping_matches, zip(blocks, it.repeat(m), it.repeat(template))))

    # matches = np.array([template_matches(block, template) for block in blocks])

    mu = (M - m + 1) / (2**m)
    std = M * ((1/(2**m))- (2*m-1)/(2**(2*m)))
    
    chisq = np.sum(((matches - mu)**2) / std)

    p = ss.gammaincc(N/2, chisq/2)
    
    success = (p >= 0.01)

    return [p, success]

def non_overlapping_matches(block, m, template):
    strides = np.packbits(np.lib.stride_tricks.as_strided(block, shape=((block.size - m + 1), m), strides=(1,1), writeable=False), axis=1).view(np.uint16).reshape(-1)
    inds    = np.where(strides == template)[0]
    dists   = np.diff(inds)

    return len(inds) - np.count_nonzero(dists < m)