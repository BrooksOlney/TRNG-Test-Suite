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

    bits = binary.unpacked.astype(np.int16)
    bits = 2*bits - 1

    s = np.add.accumulate(bits, dtype=np.int16)
    s = np.pad(s, (1, 1), 'constant')
    zcrosses = np.where(s == 0)[0]

    mi = np.min(s)
    ma = np.max(s)

    states = np.zeros((9, 6), dtype=np.int16)
    for i in range(len(zcrosses) - 1):
        subarr = s[zcrosses[i]:zcrosses[i+1]]
        subarr = subarr[(subarr >= -4) & (subarr <= 4)] + 4

        counts = np.zeros(9, dtype=np.uint8)
        for x in subarr:
            counts[x] += 1
        counts[counts > 5] = 5
        states[range(9), counts] += 1

    # J = np.sum(states)
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