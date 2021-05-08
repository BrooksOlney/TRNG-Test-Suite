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
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor
import itertools as it

from tests import *

def random_matrix():
    return [random.getrandbits(32) for row in range(32)]

def random_matrices(count):
    return [random_matrix() for _ in range(count)]



def analyze_inds(ds, inds):

    sms = 0.0
    for i in range(len(ds)):
        sms += analyze_ind(ds[i], inds[i])
    
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
                    monobit_test, frequency_within_block_test, runs_test, longest_run_within_block_test,
                    binary_matrix_rank_test, discrete_fourier_transform_test, 
                    non_overlapping_template_matching_test,
                    overlapping_template_matching_test, maurers_universal_test, 
                    linear_complexity_test,
                    serial_test, approximate_entropy_test, cumulative_sums_test, random_excursion_test, random_excursion_variant_test
        ]

        for func in nistFuncs:
            start = time.time()
            print("Starting: {}".format(func.__name__))
            ret = func(self.binary)
            end = time.time() - start
            print("\t\tResults (p-value(s), Pass/Fail) = \n\t\t\t{}".format(ret))
            print("Test completed in {}s".format(end))
            print("\n")


def main():
    # nist = TRNGtester(r'F:\Research\USF-HHL\Labs\03-P_TRNG\robert-data')
    nist = TRNGtester(r'F:\Research\USF-HHL\Labs\03-P_TRNG\data\random-data', bits=1_000_000_000)
    # nist = TRNGtester(r'F:\Research\USF-HHL\Labs\03-P_TRNG\2010-01-01.bin')
    # nist = TRNGtester(r'F:\Research\USF-HHL\Labs\03-P_TRNG\data\e.txt',bits=1_000_000)
    # nist = TRNGtester(r'F:\Research\USF-HHL\Labs\03-P_TRNG\data\data\data.pi',bits=1_000_000)
    start = time.time()
    # print(nist.run_nist_tests())
    print(overlapping_template_matching_test(nist.binary))
    print(time.time() - start)


if __name__ == "__main__":
    mp.freeze_support()
    main()
