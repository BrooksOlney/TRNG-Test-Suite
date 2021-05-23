import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing as mp
import scipy.special as ss
from functools import partial
import math

def non_overlapping_template_matching_test(binary, B=1, m=9):
    
    def template_matches(block, template):
        strides = np.packbits(np.lib.stride_tricks.as_strided(block, shape=((block.size - m + 1), m), strides=(1,1)), axis=1).view(np.uint16).reshape(-1)
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

    
    template = np.dot(template, 1 << np.arange(m, dtype=np.uint16)[::-1])

    # if n > 10_000_000:
    #     with ThreadPool(mp.cpu_count()) as p:
    #         matches = np.array([*p.imap(partial(non_overlapping_matches, m=m, template=template), blocks)])
    # else:
    #     matches = np.array([template_matches(block, template) for block in blocks])

    matches = non_overlapping_matches(blocks, m, template)

    mu = (M - m + 1) / (2**m)
    std = M * ((1/(2**m))- (2*m-1)/(2**(2*m)))
    
    chisq = np.sum(((matches - mu)**2) / std)

    p = ss.gammaincc(N/2, chisq/2)
    
    success = (p >= 0.01)

    return [p, success]

def non_overlapping_matches(block, m, template):
    strides = np.lib.stride_tricks.sliding_window_view(block, window_shape=m, axis=1)
    mask = np.array(1 << np.arange(m), dtype=np.uint16)[::-1]
    repacked = np.array([s @ mask for s in strides])

    # repacked = np.packbits(strides,axis=2).view(np.uint16).reshape(block.shape[0], -1)

    # inds  = [np.argwhere(repacked[i] == template) for i in range(len(repacked))]
    # dists = [np.diff(ind) for ind in inds]

    with ThreadPool(mp.cpu_count()) as p:
        inds  = [*p.imap(np.argwhere, repacked == template)]
        dists = [*p.imap(np.diff, inds)]

    lens = np.array([b.shape[0] for b in inds])
    
    overlaps = np.array([np.count_nonzero(b < m) for b in dists])

    return lens - overlaps