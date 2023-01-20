import numpy as np
from scipy.stats import norm
import math

def cumulative_sums_test(binary, mode=0):
    """ The focus of this test is the maximal excursion (from zero) of the random walk defined 
        by the cumulative sum of adjusted (-1, +1) digits in the sequence.  
    
        The purpose of the test is to determine whether the cumulative sum of the partial sequences 
        occurring in the tested sequence is too large or too small relative to the expected behavior 
        of that cumulative sum for random sequences.  This cumulative sum may be considered as a random walk.
    """
    def compute_psummation_func(k, i1, i2):
    
        t1 = ((4*k + i1) * z) / math.sqrt(binary.n)
        t2 = ((4*k + i2) * z) / math.sqrt(binary.n)

        return norm.cdf(t1) - norm.cdf(t2)

    # convert to binary then to -1/1 for this test
    bits = 2*binary.unpacked.astype(np.int16) - 1
    n = binary.n

    # compute cumulative sums - may require large amount of memory/time
    css = np.add.accumulate(bits, dtype=np.int16)

    z = np.max(np.abs(css))

    # ranges for summations in computing p-value
    s1range = np.array([*range(int(((-n/z)+1)/4), int(((n/z)-1)/4))])
    s2range = np.array([*range(int(((-n/z)-3)/4), int(((n/z)-1)/4))])

    s1 = np.sum(compute_psummation_func(s1range, +1, -1))
    s2 = np.sum(compute_psummation_func(s2range, +3, +1))

    p = 1 - s1 + s2
    success = (p >= 0.01)

    return [p, success]