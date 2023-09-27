import numpy as np
from scipy import linalg
import timeit
from print_lecture import print_custom as prt
from custom_band import read_banded
from custom_band import matmul_banded
from custom_band import read_banded_h
from custom_band import matmul_banded_h
from custom_sp import matmul_toeplitz
from custom_sp import matmul_circulant
from custom_decomp import perm_from_piv

A = np.array([[1, 2], [3, 4]])
