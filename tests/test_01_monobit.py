import numpy as np
import math

def monobit_test(binary):
    """ Tests the proportion of 1s to 0s in the entire binary sequence.

    Returns:
        s (int): test statistic
        p (float) : p-value for the test
        sucecss (bool): test passed/failed
    """

    ones = np.count_nonzero(binary.unpacked)
    s = 2 * ones - binary.n

    p = math.erfc(s/(math.sqrt(float(binary.n))*math.sqrt(2.0)))
    success = p >= 0.01

    return [p, success]