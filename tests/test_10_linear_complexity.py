import numpy as np
import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool
import scipy.special as ss

def init_pool(data):
    global blocks
    blocks = data

def linear_complexity_test(binary, M=512, K=6):
    
    blocksize = M // 8
    bytes = binary.packed
    N = len(bytes) // blocksize
    
    # shr = mp.shared_memory.SharedMemory(create=True, size=N*(M // 8))
    # blocks = np.frombuffer(shr._buf, dtype=np.uint8).reshape(N, blocksize)
    blocks = bytes[:N*(M // 8)].reshape(N, blocksize)

    blocks = np.array([int.from_bytes(block.tobytes(), 'big') for block in blocks])

    nJobs = mp.cpu_count()

    if len(blocks) // 1000 > nJobs:
        with ThreadPool(nJobs) as p:
            Ls = np.hstack([*p.imap(vectorized_berlekamp_massey, np.array_split(blocks, nJobs))])
    else:
        Ls = vectorized_berlekamp_massey(blocks)

    # compute expected average and test statistic for each block
    mu = (M / 2) + ((9 + ((-1)**(M+1)))/36) - (((M/3) + (2/9)) / (2**M))
    Ts = np.array([((-1)**(M)) * (Li - mu) + (2/9) for Li in Ls]) * -1

    # compute the values for V_0...V_6 based on the given bounds
    vs = np.histogram(
        Ts, bins=[-9e10, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 9e10]
    )[0][::-1]  # take first tuple and reverse it

    # compute chisq using V and given probabilities for K=6
    pis = [0.010417, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833]
    chisq = sum(np.array([
        (v-(N*pis[i]))**2 / (N*pis[i]) for i, v in enumerate(vs) if i < 7
    ]))

    p = ss.gammaincc(K/2, chisq/2)
    success = (p >= 0.01)

    # shr.close()
    # shr.unlink()

    return [p, success]

def vectorized_berlekamp_massey(blocks):
        # start = time.time()
    n = len(blocks)
    Dc = Db = blocks
    L = np.zeros(n, dtype=np.uint16)
    # m = np.array([-1] * n, dtype=np.int16)
    i = 0
    M = 512

    while i < M: 
        d = (Dc & 1).astype(bool)
        Dc = Dc >> 1
        mask = L <= i / 2

        T = Dc[d & mask]

        Dc[d] ^= (~Db[d] & (2**(M-i + 1) - 1))

        L[d & mask] = i + 1 - L[d & mask] 
        Db[d & mask] = T

        i += 1

    return L

def berlekamp_massey(block):
    n = len(block)
    c = np.zeros(n, dtype=np.uint8)
    b = np.zeros(n, dtype=np.uint8)
    c[0], b[0] = 1, 1
    L, m = 0, -1
    i = 0
    while i < n:
        d = (block[i] + np.dot(block[i-L:i][::-1], c[1:L+1])) % 2

        if d:
            p = np.zeros(n, dtype=np.uint8)
            p[i-m:L+i-m] = b[:L]
            if L <= 0.5 * i:
                L = i + 1 - L
                m = i
                b = c
            c = (c + p) % 2
        i += 1
    return L

def berlekamp_massey_opt(block):
    Dc = Db = int.from_bytes(block.tobytes(), 'big')
    L = 0
    m = -1
    i = 0
    M = len(block) * 8
    while i < M: 
        d = Dc & 1
        Dc = Dc >> 1
        if d:
            T  = Dc
            Dc = Dc ^ (~Db & (2**(M-i + 1) - 1))
            if L <= i / 2:
                L = i + 1 - L
                Db = T
                m = i

        i += 1

    return L
