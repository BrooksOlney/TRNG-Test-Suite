import numpy as np
import math


def maurers_universal_test(binary):
    bits = binary.unpacked
    n = binary.n

    L = 5
    if n >= 387840:
        L = 6
    if n >= 904960:
        L = 7
    if n >= 2068480:
        L = 8
    if n >= 4654080:
        L = 9
    if n >= 10342400:
        L = 10
    if n >= 22753280:
        L = 11
    if n >= 49643520:
        L = 12
    if n >= 107560960:
        L = 13
    if n >= 231669760:
        L = 14
    if n >= 496435200:
        L = 15
    if n >= 1059061760:
        L = 16

    _var = [2.954, 3.125, 3.238, 3.311, 3.356, 3.384, 3.401, 3.410, 3.416, 3.419, 3.421]
    _EV  = [5.2177052, 6.1962507, 7.1836656, 8.1764248, 9.1723243, 10.170032, 11.168765, 12.168070, 13.167693, 14.167488, 15.167379]

    var = _var[L - 6]
    EV = _EV[L - 6]

    if L == 5:
        return -1
    Q = 10 * (2**L)
    K = (n // L) - Q

    encoding = np.uint8 if L <= 8 else np.uint16
    dtype = "B" if L <= 8 else "I"
    # shr = mp.Array(ctypes.c_uint16, K)
    # shr = shared_memory.SharedMemory(create=True, size=K if L <=8 else K*2)
    # testSeg = np.frombuffer(shr.buf, dtype=encoding)

    initSeg = np.packbits(bits[:Q*L].reshape(Q,L), axis=1).view(encoding).reshape(Q)
    testSeg = np.packbits(bits[Q*L:(K+Q)*L].reshape(K, L), axis=1).view(encoding).reshape(K)

    qvals, qlast = np.unique(initSeg[::-1], return_index=True)
    qlast = Q - qlast

    qdict = np.zeros(2**(encoding(0).nbytes * 8),dtype=np.uint64)
    qdict[qvals] = qlast
    
    kvals = np.unique(testSeg)
    qinds = qdict[kvals]
    
    idx = np.argsort(testSeg)
    res = np.split(idx, np.flatnonzero(np.diff(testSeg[idx])) + 1)

    largestBin = max(res, key=len)
    
    paddedArr = np.array([np.hstack([(largestBin.size - len(arr) + 1) * [qinds[i]], arr + Q + 1]) for i,arr in enumerate(res)], dtype=np.uint32)

    paddedArr = np.sort(paddedArr, axis=1)
    test = np.diff(paddedArr, axis=1)
    test[test == 0] = 1
    sms = np.sum(np.log2(test))

    fn = (1/K) * sms
    c = 0.7 - 0.8 / L + (4 + 32 / L) * pow(K, -3 / L) / 15
    sigma = c * math.sqrt(var / K)

    stat = abs((fn - EV)) / (math.sqrt(2)*sigma)
    p = math.erfc(stat)

    success = (p >= 0.01)

    return [p, success]