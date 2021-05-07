import numpy as np
import math

def random_excursion_variant_test(binary):
    bits = 2*binary.unpacked.astype(np.int8)-1

    s = np.sort(np.cumsum(bits, dtype=np.int32))
    J = len(np.where(s == 0)[0]) + 1
    s = s[(s >= -9) & (s <= 9)]

    num, counts = np.unique(s, return_counts=True)

    # compute p value for each class (18 total classes)
    ps = []
    for num, c in zip(num, counts):
        if num != 0:
            p = math.erfc(abs(c-J)/math.sqrt(2*J*(4*abs(num)-2)))
            ps.append(p)

    return [(p, p >= 0.01) for p in ps]