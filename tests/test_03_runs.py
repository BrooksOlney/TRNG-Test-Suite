import numpy as np
import math

def runs_test(binary):
    """
        This tests the "oscillation" of the sequence,
        i.e. how much it switches from 1->0 and 0->1.
    """
    # proportion of 1s in the binary string
    ones = binary.packed.view(np.uint32)
    ones = ones - ((ones >> 1) & 0x55555555)
    ones = (ones & 0x33333333) + ((ones>> 2) & 0x33333333)
    ones = (((ones + (ones >> 4) & 0xF0F0F0F) * 0x1010101) & 0xffffffff) >> 24
    prop = np.sum(ones) / binary.n

    # test statistic
    vobs = 1 + np.sum(binary.unpacked[:-1] ^ binary.unpacked[1:])
    p = math.erfc(abs(vobs-(2.0*binary.n*prop*(1.0-prop)))/(2.0*math.sqrt(2.0*binary.n)*prop*(1-prop)))

    success = (p >= 0.01)

    return [p, success]