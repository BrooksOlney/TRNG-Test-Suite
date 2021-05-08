import numpy as np
import math

def random_excursion_variant_test(binary):
    bits = 2*binary.unpacked.astype(np.int8)-1
    bitsPerJob = 1_000_000

    J = 0
    s = np.arange(-9,10)

    # if len(bits) > bitsPerJob:
    #     bits = [bits[i*bitsPerJob:(i+1)*bitsPerJob] for i in range(r)]

    # else:
    #     s = batched_cumsum(bits)
    #     J = s[9] + 1
    # for i in range(r):
    #     _J, _s = batched_cumsum(bits[i*bitsPerJob:(i+1)*bitsPerJob])
    #     J += _J
    #     s[range(len(_s))] += _s
    # s = batched_cumsum(bits, 1_000_000)
    s = np.add.accumulate(bits, dtype=np.int32)
    J = len(s[s==0]) + 1
    s = s[(s >= -9) & (s <= 9)]

    num, counts = np.unique(s, return_counts=True)

    # compute p value for each class (18 total classes)
    ps = []
    for num, c in zip(num, counts):
        if num != 0:
            p = math.erfc(abs(c-J)/math.sqrt(2*J*(4*abs(num)-2)))
            ps.append(p)

    return [(p, p >= 0.01) for p in ps]

def batched_cumsum(bits, batchSize):
    r = math.ceil(len(bits) // bitsPerJob)
    
    s = []
    counts = np.zeros(19, dtype=np.uint32)
    for i in range(r):
        s = np.cumsum([s[-1], bits[i*batchSize : (i+1)*batchSize]], dtype=np.int32)
        # J = len(s == 0) + 1
        count = np.bincount(s[(s >= -9) & (s <= 9)] + 9)
        counts[range(len(count))] += counts

    return J, s