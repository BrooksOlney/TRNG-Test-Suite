import numpy as np
import math
import scipy.special as ss
import time
from scipy.stats import rankdata, norm
import matplotlib.pyplot as plt
import scipy.fft as fft
from scipy.stats import chisquare
import multiprocessing as mp
import ctypes
import cProfile, pstats, io
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor
import itertools as it

# plt.style.use(r'F:\research\USF-HHL\settings.mplstyle')

def random_matrix():
    return [random.getrandbits(32) for row in range(32)]

def random_matrices(count):
    return [random_matrix() for _ in range(count)]

def gf2_rank(rows):
    """
        Find rank of a matrix over GF^2.

        The rows of the binary matrix are represented as integers,
        in a numpy array (rows).
    """
    ranks = np.zeros(len(rows), dtype=np.uint8)

    for i in range(32):
        pivot_row = rows[:,i].reshape(-1,1)
        lsb = pivot_row & -pivot_row
        ranks += np.count_nonzero(pivot_row, axis=1).astype(np.uint8)
        inds = rows & lsb > 0
        rows[inds] ^= np.repeat(pivot_row, 32, axis=1)[inds]

    return ranks

def overlapping_matches(block, m, template):
    strides = np.packbits(np.lib.stride_tricks.as_strided(block, shape=((block.size - m + 1), m), strides=(1,1)), axis=1).view(np.uint16).reshape(-1)
    return np.count_nonzero(strides == template)

def non_overlapping_matches(block, m, template):
    strides = np.packbits(np.lib.stride_tricks.as_strided(block, shape=((block.size - m + 1), m), strides=(1,1), writeable=False), axis=1).view(np.uint16).reshape(-1)
    inds    = np.where(strides == template)[0]
    dists   = np.diff(inds)

    return len(inds) - np.count_nonzero(dists < m)

def vectorized_fft(x):
    return np.abs(fft.rfft(x)[:len(x) // 2]).astype(np.float32)

def init_pool(data):
    global blocks
    blocks = data

def analyze_inds(ds, inds):

    sms = 0.0
    # for d, ind in np.column_stack((ds,inds)):
    #     _inds = np.hstack([ind, np.nonzero(testSeg == d)[0] + Q + 1])
    #     sms += sum(np.log2(np.diff(_inds)))
    for i in range(len(ds)):

        sms += analyze_ind(ds[i], inds[i])

        # for i in _inds:
        #     sms += np.log2(i - ind)
        #     ind = i

    return sms

def analyze_ind(d, ind):
    
    sms = 0.0
    _inds = np.hstack([ind, np.nonzero(testSeg == d)[0] + Q + 1])
    sms = sum(np.log2(np.diff(_inds)))

    return sms

class TRNGtester:

    def __init__(self, binaryFile=None, bits=-1, binformat='bytes'):
        if binaryFile != None:
            self.binary = self.BinaryData(binaryFile, bits=bits, binformat=binformat)
        else:
            self.binary = None

    class BinaryData:
        def __init__(self, filename, bits=-1, binformat='bytes'):
            if filename.endswith(".txt") or binformat != 'bytes': 
                lines = open(filename).readlines()
                rawData = ''.join(lines).replace('\n','').replace(' ','').replace('\t','')
                n = bits // 8 if bits != -1 else len(rawData) // 8
                self.packed = np.array([int(rawData[i*8:(i+1)*8], 2) for i in range(n)],dtype=np.uint8)
            else:
                self.packed = np.fromfile(filename, dtype=np.uint8, count=bits//8)

            self.unpacked = np.unpackbits(self.packed)
            self.n = len(self.packed) * 8

    def read_binary(self, filename, bits=-1):
        return self.BinaryData(filename, bits=bits)

    def plot_cumsum(self):

        bits = 2*self.binary.unpacked.astype(np.int8) - 1

        bcs = np.cumsum(bits, dtype=np.int16)

        plt.plot(range(len(bcs)), bcs, linewidth=1)
        plt.title('Cumulative Sums of -1/+1 Encoded Sequence')
        plt.show()

    def plot_nums(self):

        bytes = self.binary.packed.astype(np.int8)

        plt.scatter(range(len(bytes)), bytes)
        plt.show()


    def run_nist_tests(self):
        nistFuncs = [
                    self.monobit_test, self.frequency_within_block_test, self.runs_test, self.longest_run_within_block_test,
                    self.binary_matrix_rank_test, self.discrete_fourier_transform_test, 
                    self.non_overlapping_template_matching_test,
                    self.overlapping_template_matching_test, self.maurers_universal_test, 
                    self.linear_complexity_test,
                    self.serial_test, self.approximate_entropy_test, self.cumulative_sums_test, self.random_excursion_test, self.random_excursion_variant_test
        ]
        # binary = self.read_binary(r'F:\Research\USF-HHL\Labs\03-P_TRNG\e.txt',bits=1000000)
        # self.binary = binary
        for func in nistFuncs:
            start = time.time()
            print("Starting: {}".format(func.__name__))
            ret = func()
            end = time.time() - start
            print("\t\tResults (p-value(s), Pass/Fail) = \n\t\t\t{}".format(ret))
            print("Test completed in {}s".format(end))
            print("\n")


    def monobit_test(self):
        """ Tests the proportion of 1s to 0s in the entire binary sequence.

        Returns:
            s (int): test statistic
            p (float) : p-value for the test
            sucecss (bool): test passed/failed
        """

        ones = np.count_nonzero(self.binary.unpacked)
        s = 2 * ones - self.binary.n

        p = math.erfc(s/(math.sqrt(float(self.binary.n))*math.sqrt(2.0)))
        success = p >= 0.01

        return [p, success]

    def frequency_within_block_test(self, M=128):
        """ Test the proportion of 1s in each block of size M. Should be around 2/M if numbers are random.
            Equivalent to the monobit test if M=1.
        Args:
            input (binary): binary data
            n (int): size of total binary string
            M (int, optional): Block size. Defaults to 128.
        """

        bytes = self.binary.packed
        nBlocks = self.binary.n // M
        bytesPerBlock = M // 8

        _bytes = bytes[:nBlocks*bytesPerBlock].reshape(nBlocks, bytesPerBlock)
        proportions = np.count_nonzero(
            np.unpackbits(_bytes, axis=1), axis=1
        ) / M

        chisq = np.sum(4.0 * M * ((proportions - 0.5) ** 2))
        p = ss.gammaincc((nBlocks/2.0), float(chisq)/2.0)
        success = (p >= 0.01)

        return [p, success]

    def runs_test(self):
        """
            This tests the "oscillation" of the sequence,
            i.e. how much it switches from 1->0 and 0->1.
        """

        bits = self.binary.unpacked

        # proportion of 1s in the binary string
        prop = np.sum(bits) / len(bits)

        # test statistic
        vobs = 1 + (bits[:-1] ^ bits[1:]).sum()
        p = math.erfc(abs(vobs-(2.0*self.binary.n*prop*(1.0-prop)))/(2.0*math.sqrt(2.0*self.binary.n)*prop*(1-prop)))

        success = (p >= 0.01)

        return [p, success]

    def longest_run_within_block_test(self, KM=0):
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
            # test = np.diff(np.where(np.concatenate(([a[0]], a[:-1] != a[1:], [True])))[0])[::2]
            # return max(test, default=0)

        bits = self.binary.packed
        n = len(bits) * 8

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


        bits = self.binary.unpacked
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

    def binary_matrix_rank_test(self, M=32, Q=32):
        """The focus of the test is the rank of disjoint sub-matrices of the entire sequence.  
            This test checks for linear dependence among fixed length substrings of the original sequence.

        Args:
            M (int): Number of rows in the rank matrix
            Q (int): Number of columns in the rank matrix
        """
        # repack bytes into proper format, where each number is a row of the matrix
        # so an array of 32 numbers is effectively a 32x32 matrix
        # this allows us to compute matrix rank very quickly using bitwise operations
        N = self.binary.n // (M * Q)
        bytes = self.binary.packed.tobytes()[:N * M * Q // 8]
        repacked = np.frombuffer(bytes, dtype=np.uint32).reshape(-1, 32)

        # ranks = np.zeros(len(repacked), dtype=np.uint8)
        ranks = []

        # parallelize an already vectorized operation, 16 is a good number why not
        with mp.Pool(mp.cpu_count()) as p:
            ranks = np.hstack(p.map(gf2_rank, np.array_split(repacked, mp.cpu_count() * 16)))

        # number of matrices with rank == M, M-1, and otherwise
        FM = ranks.tolist().count(M)
        FMM = ranks.tolist().count(M-1)
        rem = N - FM - FMM

        # generate the probabilities for each matrix/rank correlation
        probs = []
        for r in [M, M-1]:
            product = 1.0
            for i in range(r):
                u1 = (1.0 - (2.0**(i-Q)))
                u2 = (1.0 - (2.0**(i-Q)))
                L = (1-(2.0**(i-r)))
                product *= (u1*u2)/L

            probs.append(product * (2.0**((r*(Q+M-r)) - (M*Q))))

        # set probabilities
        pFM, pFMM = probs
        prem = 1.0 - (pFM + pFMM)

        # compute chisq and pvalue
        chisq = (((FM - pFM*N)**2) / (pFM*N)) + (((FMM - pFMM*N)**2) / (pFMM*N)) + (((rem - prem*N)**2) / (prem*N))
        p = np.exp(-chisq/2)

        success = (p >= 0.01)
        
        return [p, success]

    def discrete_fourier_transform_test(self, m=1):

        bits = 2*self.binary.unpacked.astype(np.int8) - 1

        # two-level test for bitstreams that are too large
        if self.binary.n > 10_000_000:
            m = 1000
        
        blockSize = self.binary.n // m
        blocks = bits.reshape(m, blockSize)
        T = math.sqrt(math.log(1.0/0.05)*blockSize)

        with mp.Pool(mp.cpu_count()) as p:
            s = p.map(vectorized_fft, blocks)

        N0 = 0.95 * blockSize / 2.0
        N1 = np.array([len(np.where(si < T)[0]) for si in s])

        d = (N1 - N0)/math.sqrt((blockSize*0.95*0.05)/4)

        # compute proportion of p-values that pass
        p = np.array([math.erfc(abs(di)/math.sqrt(2)) for di in d])
        alpha = 0.01
        mean  = 1.0 - alpha
        std   = math.sqrt((alpha*mean / m))

        prop = len(np.where(p >= alpha)[0]) / m
        cilow = mean - 3*std
        cihigh = mean + 3*std

        # check if passes proportion test
        propPass = prop >= cilow and prop <= cihigh

        # perform chisq test on computed p-values
        chsq, uniformityp = chisquare(p, ddof=10)
        pt = ss.gammainc(10/2,chsq/2)

        success = (p >= 0.01)

        return [prop, propPass, pt, pt >= 0.0001]

    def non_overlapping_template_matching_test(self, B=1, m=9):

        def template_matches(block, template):
            strides = np.packbits(np.lib.stride_tricks.as_strided(block, shape=((block.size - m + 1), m), strides=(1,1), writeable=False), axis=1).view(np.uint16).reshape(-1)
            inds    = np.where(strides == template)[0]
            dists   = np.diff(inds)

            return len(inds) - np.count_nonzero(dists < m)

        bits = self.binary.unpacked
        n = len(bits)
        M = n // 8
        N = n // M

        blocks = bits[:N*M].reshape(N,M)
        template = np.unpackbits(np.array(B, dtype=np.uint8))
        
        if len(template) > m:
            template = template[len(template) - m:]
        elif len(template) < m:
            template = np.concatenate([np.zeros((m - len(template)), dtype=np.uint8), template])

        template = np.packbits(template).view(np.uint16)

        with mp.Pool(mp.cpu_count()) as p:
            matches = np.array(p.starmap(non_overlapping_matches, zip(blocks, it.repeat(m), it.repeat(template))))

        # matches = np.array([template_matches(block, template) for block in blocks])

        mu = (M - m + 1) / (2**m)
        std = M * ((1/(2**m))- (2*m-1)/(2**(2*m)))
        
        chisq = np.sum(((matches - mu)**2) / std)

        p = ss.gammaincc(N/2, chisq/2)
       
        success = (p >= 0.01)

        return [p, success]

    def overlapping_template_matching_test(self, m=9, K=5):
        

        def get_prob(u, x):
            out = 1.0 * np.exp(-x)
            if u != 0:
                out = 1.0 * x * np.exp(2 * -x) * (2 ** -u) * ss.hyp1f1(u + 1, 2, x)
            return out

        bits = self.binary.unpacked
        n = len(bits)
        M =  1032
        N = n // M

        blocks = bits[:N*M].reshape(N,M)

        template = np.ones(m, dtype=np.uint8)
        template = np.packbits(template).view(np.uint16).reshape(-1)
        
        with mp.Pool(mp.cpu_count()) as p:
            matches = np.array(p.starmap(overlapping_matches, zip(blocks, it.repeat(m), it.repeat(template))))

        # matches = np.array([template_matches(block) for block in blocks])
        lmbda = (M-m+1)/(2**m)
        nu = lmbda/2

        vs = np.array([len(matches[matches == i]) for i in range(6)])
        vs[5] = len(matches[matches >= 5])
        # pis = np.array([0.364091, 0.185659, 0.139381, 0.100571, 0.070432, 0.139865])
        pis = [get_prob(i, nu) for i in range(5)]
        pis = np.array([*pis, 1-sum(pis)])
        
        chisq = sum((vs - N*pis)**2 / (N*pis))

        p = ss.gammaincc(K/2, chisq/2)

        success = (p >= 0.01)
        
        return [p, success]

    def maurers_universal_test(self):
        bits = self.binary.unpacked
        n = len(bits)

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
        
        numJobs = 16
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

    def linear_complexity_test(self, M=512, K=6):

        def berlekamp_massey(block):
            n = len(block)
            c = np.zeros(n, dtype=np.uint8)
            b = np.zeros(n, dtype=np.uint8)
            c[0], b[0] = 1, 1
            L, m = 0, -1
            i = 0
            while i < n:
                d = (block[i] + np.dot(block[i-L:i][::-1], c[1:L+1])) % 2

                if d:
                    p = np.zeros(n, dtype=np.uint8)
                    p[i-m:L+i-m] = b[:L]
                    if L <= 0.5 * i:
                        L = i + 1 - L
                        m = i
                        b = c
                    c = (c + p) % 2
                i += 1
            return L

        def largest_factor(n):
            for i in range(2, math.floor(math.sqrt(n))):
                if n % i == 0: return i

            return 1

        blocksize = M // 8
        bytes = self.binary.packed
        N = len(bytes) // blocksize
        
        shr = shared_memory.SharedMemory(create=True, size=N*(M // 8))
        blocks = np.frombuffer(shr._buf, dtype=np.uint8).reshape(N, blocksize)
        blocks[:] = bytes[:N*(M // 8)].reshape(N, blocksize)

        blocks = np.array([int.from_bytes(block.tobytes(), 'big') for block in blocks], dtype=np.int(bytes=))
        blocks = blocks.byteswap()
        # Ls = self.vectorized_berlekamp_massey(blocks)
        # Ls = [*map(berlekamp_massey, np.unpackbits(blocks, axis=1))]

        nJobs = mp.cpu_count()
        lf = largest_factor(blocks.size)
        Ls = self.vectorized_berlekamp_massey(blocks)

        # compute expected average and test statistic for each block
        mu = (M / 2) + ((9 + ((-1)**(M+1)))/36) - (((M/3) + (2/9)) / (2**M))
        Ts = np.array([((-1)**(M)) * (Li - mu) + (2/9) for Li in Ls]) * -1

        # compute the values for V_0...V_6 based on the given bounds
        vs = np.histogram(
            Ts, bins=[-9e10, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 9e10]
        )[0][::-1]  # take first tuple and reverse it

        # compute chisq using V and given probabilities for K=6
        pis = [0.010417, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833]
        chisq = sum(np.array([
            (v-(N*pis[i]))**2 / (N*pis[i]) for i, v in enumerate(vs) if i < 7
        ]))

        p = ss.gammaincc(K/2, chisq/2)
        success = (p >= 0.01)

        return [p, success]

    def vectorized_berlekamp_massey(self, blocks):
            # start = time.time()
        n = len(blocks)
        Dc = Db = blocks
        L = np.zeros(n)
        m = np.array([-1] * n, dtype=np.int16)
        i = 0
        M = blocks[0].bit_length()

        while i < M: 
            d = (Dc & 1).astype(bool)
            Dc = Dc >> 1
            mask = L <= i / 2

            T = Dc[d & mask]

            Dc[d] ^= (~Db[d] & (2**(M-i + 1) - 1))

            L[d & mask] = i + 1 - L[d & mask] 
            Db[d & mask] = T
            
            # if d:
            #     Dc = Dc ^ (~Db & (2**(M-i + 1) - 1))
            #     if L <= i / 2:
            #         L = i + 1 - L
            #         Db = T

            i += 1

        # print(time.time() - start)
        return L

    def berlekamp_massey_opt(self, block):
        # start = time.time()
        Dc = Db = int.from_bytes(block.tobytes(), 'big')
        L = 0
        m = -1
        i = 0
        M = len(block) * 8
        while i < M: 
            d = Dc & 1
            Dc = Dc >> 1
            if d:
                T  = Dc
                Dc = Dc ^ (~Db & (2**(M-i + 1) - 1))
                if L <= i / 2:
                    L = i + 1 - L
                    Db = T
                    m = i

            i += 1

        # print(time.time() - start)
        return L

    def serial_test(self):
        """ The focus of this test is the frequency of all possible overlapping m-bit
            patterns across the entire sequence.

        """

        bits = self.binary.unpacked
        n = self.binary.n
        M = 16

        bits = np.concatenate([bits, bits[:M-1]])

        psisqs = []
        for i in range(3):
            m = M - i
            nrows = ((bits.size - m)) + 1
            _n = bits.strides[0]
            mpattern = np.packbits(
                np.lib.stride_tricks.as_strided(
                    bits, shape=(nrows, m), strides=(_n, _n)
                ), axis=1
            ).view(np.uint16).reshape(-1)

            # get count of occurrences, compute psisq for that block size
            mcounts = np.bincount(mpattern)
            psisq = ((2**(M-i)) / n) * sum(mcounts**2) - n
            psisqs.append(psisq)

        dpsi = psisqs[0] - psisqs[1]
        d2psi = psisqs[0] - 2*psisqs[1] + psisqs[2]

        p1 = ss.gammainc(2**(M-2), dpsi/2)
        p2 = ss.gammainc(2**(M-3), d2psi/2)

        success1 = p1 >= 0.01
        success2 = p2 >= 0.01

        return [p1, success1, p2, success2]

    def approximate_entropy_test(self):
        """ As with the Serial test, the focus of this test is the
            frequency of all possible overlapping m-bit patterns across
            the entire sequence.
        """
        bits = np.unpackbits(self.binary.packed)
        n = self.binary.n
        # M    = math.floor(np.log2(n)) - 5
        M = 10

        # mbits = np.concatenate([bits, bits[:M-1]])
        # m1bits = np.concatenate([bits, bits[:M]])

        mblocks = np.packbits(
            np.lib.stride_tricks.as_strided(
                np.concatenate([bits, bits[:M-1]]), (n, M), (1, 1), writeable=False
            ), axis=1
        ).view(np.uint16).reshape(-1)
        m1blocks = np.packbits(
            np.lib.stride_tricks.as_strided(
                np.concatenate([bits, bits[:M]]), (n, M+1), (1, 1), writeable=False
            ), axis=1
        ).view(np.uint16).reshape(-1)
        
        mcounts = np.unique(mblocks, return_counts=True)[1]
        m1counts = np.unique(m1blocks, return_counts=True)[1]

        mpsi = np.sum(mcounts * np.log(mcounts)) / n
        m1psi = np.sum(m1counts * np.log(m1counts)) / n

        chisq = 2*n*(math.log(2) - (mpsi - m1psi))

        p = ss.gammaincc(2**(M-1), chisq/2)

        success = (p >= 0.01)

        return [p, success]

    def cumulative_sums_test(self, mode=0):
        """ The focus of this test is the maximal excursion (from zero) of the random walk defined 
            by the cumulative sum of adjusted (-1, +1) digits in the sequence.  
        
            The purpose of the test is to determine whether the cumulative sum of the partial sequences 
            occurring in the tested sequence is too large or too small relative to the expected behavior 
            of that cumulative sum for random sequences.  This cumulative sum may be considered as a random walk.
        """
        def compute_psummation_func(k, i1, i2):

            t1 = ((4*k + i1) * z) / math.sqrt(self.binary.n)
            t2 = ((4*k + i2) * z) / math.sqrt(self.binary.n)

            return norm.cdf(t1) - norm.cdf(t2)

        # convert to binary then to -1/1 for this test
        bits = 2*np.unpackbits(self.binary.packed).astype(np.int8) - 1
        n = self.binary.n

        # compute cumulative sums - may require large amount of memory/time
        css = bits.cumsum(dtype=np.int32)

        z = np.max(np.abs(css))

        # ranges for summations in computing p-value
        s1range = range(int(((-n/z)+1)/4), int(((n/z)-1)/4))
        s2range = range(int(((-n/z)-3)/4), int(((n/z)-1)/4))

        s1, s2 = 0, 0
        for k in s1range:
            s1 += compute_psummation_func(k, +1, -1)
        for k in s2range:
            s2 += compute_psummation_func(k, +3, +1)

        p = 1 - s1 + s2
        success = (p >= 0.01)

        return [p, success]

    def random_excursion_test(self):

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

        bits = np.unpackbits(self.binary.packed).astype(np.int8)
        bits = 2*bits - 1

        s = bits.cumsum(dtype=np.int32)
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

    def random_excursion_variant_test(self):
        bits = 2*self.binary.unpacked.astype(np.int8)-1

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


def main():
    # nist = TRNGtester(r'F:\Research\USF-HHL\Labs\03-P_TRNG\robert-data')
    # nist = TRNGtester(r'F:\Research\USF-HHL\Labs\03-P_TRNG\data\random-data', bits=1_000_000_000)
    # nist = TRNGtester(r'F:\Research\USF-HHL\Labs\03-P_TRNG\2010-01-01.bin')
    nist = TRNGtester(r'F:\Research\USF-HHL\Labs\03-P_TRNG\data\e.txt',bits=1_000_000)
    # nist = TRNGtester(r'F:\Research\USF-HHL\Labs\03-P_TRNG\data\data\data.pi',bits=1_000_000)
    start = time.time()
    print(nist.linear_complexity_test())
    print(time.time() - start)


if __name__ == "__main__":
    mp.freeze_support()
    main()
