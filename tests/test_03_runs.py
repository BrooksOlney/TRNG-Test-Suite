import numpy as np
import math

def runs_test(binary):
    """
        This tests the "oscillation" of the sequence,
        i.e. how much it switches from 1->0 and 0->1.
    """

    bits = binary.unpacked

    # proportion of 1s in the binary string
    prop = np.sum(bits) / len(bits)

    # test statistic
    vobs = 1 + (bits[:-1] ^ bits[1:]).sum()
    p = math.erfc(abs(vobs-(2.0*binary.n*prop*(1.0-prop)))/(2.0*math.sqrt(2.0*binary.n)*prop*(1-prop)))

    success = (p >= 0.01)

    return [p, success]