import numpy as np
import scipy.special as ss

def longest_run_within_block_test(binary, KM=0):
    """
        Like longest run test, but within blocks of the binary string.
    """
    def longest_run_in_block(a):
        x = int.from_bytes(np.packbits(a).tobytes(), byteorder='big')

        count = 0
        while x:
            x = (x & (x << 1)) 
            count += 1

        return count

    bits = binary.unpacked
    n = binary.n

    if n < 128:
        print("ERROR! Not enough data to run this test. (Longest run within block test)")
        return -1
    elif n <= 6272:
        K, M = 3, 8
        vclasses = [1, 2, 3, 4]
        vprobs = [0.2148, 0.3672, 0.2305, 0.1875]
    elif n <= 28160:
        K, M = 5, 128
        vclasses = [4, 5, 6, 7, 8, 9]
        vprobs = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
    elif n <= 75000:
        K, M = 5, 512
        vclasses = [6, 7, 8, 9, 10, 11]
        vprobs = [0.1170, 0.2460, 0.2523, 0.1755, 0.1027, 0.1124]
    elif n <= 750000:
        K, M = 5, 1000
        vclasses = [7, 8, 9, 10, 11, 12]
        vprobs = [0.1307, 0.2437, 0.2452, 0.1714, 0.1002, 0.1088]
    else:
        K, M = 6, 10000
        vclasses = [10, 11, 12, 13, 14, 15, 16]
        vprobs = [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]

    N = n // M
    vs = [0.0] * len(vprobs)

    numBlocks = len(bits) // M

    bits = bits[:numBlocks * M].reshape(numBlocks, M)
    runs = np.apply_along_axis(longest_run_in_block, 1, bits)

    # compute the frequency of lengths based on the monitored classes
    freqs = np.histogram(runs, bins=[-9e10,*vclasses, 9e10])[0]
    freqs = [sum(freqs[:2]), *freqs[2:]]
    vs = np.array(freqs, dtype=np.float32)

    chisq = sum(((vs[i] - N*vprobs[i])**2 / (N*vprobs[i]) for i in range(K)))

    p = ss.gammaincc(K/2, chisq/2)

    success = (p >= 0.01)

    return [p, success]