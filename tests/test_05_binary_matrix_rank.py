import numpy as np
import multiprocessing as mp


def binary_matrix_rank_test(binary, M=32, Q=32):
    """The focus of the test is the rank of disjoint sub-matrices of the entire sequence.  
        This test checks for linear dependence among fixed length substrings of the original sequence.

    Args:
        M (int): Number of rows in the rank matrix
        Q (int): Number of columns in the rank matrix
    """
    # repack bytes into proper format, where each number is a row of the matrix
    # so an array of 32 numbers is effectively a 32x32 matrix
    # this allows us to compute matrix rank very quickly using bitwise operations
    N = binary.n // (M * Q)
    bytes = binary.packed.tobytes()[:N * M * Q // 8]
    repacked = np.frombuffer(bytes, dtype=np.uint32).reshape(-1, 32)

    # ranks = np.zeros(len(repacked), dtype=np.uint8)
    ranks = []

    # parallelize an already vectorized operation, 16 is a good number why not
    if binary.n > 10_000_000:
        with mp.Pool(mp.cpu_count()) as p:
            ranks = np.hstack([*p.imap(gf2_rank, np.array_split(repacked, mp.cpu_count() * 16))])
    else:
        ranks = gf2_rank(repacked.copy())

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