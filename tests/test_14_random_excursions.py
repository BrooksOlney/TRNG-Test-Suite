from multiprocessing.pool import ThreadPool
import numpy as np
import scipy.special as ss

def random_excursion_test(binary):

    def get_probability(x, k):
        pi = 0
        x -= 4
        if k == 0:
            pi = 1 - 1 / (2*abs(x))
        elif k == 5:
            pi = (1/(2*abs(x)))*(1-(1/(2*abs(x))))**4
        else:
            pi = (1/(4*x**2))*(1-(1/(2*abs(x))))**(k-1)

        return pi

    bits = 2*binary.unpacked.astype(np.int16) - 1

    s = np.add.accumulate(bits, dtype=np.int16)
    s = np.pad(s, (1, 1), 'constant')
    zcrosses = np.argwhere(s == 0).reshape(-1)
    inds2d = zcrosses.reshape(-1,2)
 
    test = [np.array(s[r[0]:r[1]]) for r in inds2d]
    test2 = [t[(t >= -4) & (t <= 4)] + 4 for t in test]

    with ThreadPool() as p:
        counts = np.vstack([np.pad(r, [0, 9 - len(r)]) for r in p.imap(np.bincount, test2)])
    
    

    states = np.zeros((9, 6), dtype=np.int64)
    for i in range(len(zcrosses) - 1):
        subarr = s[zcrosses[i]:zcrosses[i+1]]
        subarr = subarr[(subarr >= -4) & (subarr <= 4)] + 4

        counts = np.zeros(9, dtype=np.int64)
        # for x in subarr:
        #     counts[x] += 1
        _counts = np.bincount(subarr)
        counts[range(len(_counts))] += _counts
        counts[counts > 5] = 5
        states[range(9), counts] += 1

    J = len(zcrosses) - 1

    chisqs = np.zeros(9, dtype=np.float32)
    for x in range(9):
        if x == 4:
            continue
        for k in range(6):
            factor = J*get_probability(x, k)
            chisqs[x] += (states[x, k] - factor)**2 / factor

    ps = [ss.gammaincc(5/2, chisq/2) for chisq in chisqs]

    return [(p, p >= 0.01) for p in ps]