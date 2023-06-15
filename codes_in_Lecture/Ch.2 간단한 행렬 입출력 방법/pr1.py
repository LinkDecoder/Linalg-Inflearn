import numpy as np
from print_lecture import print_custom as prt

A = np.genfromtxt('pr1_inp1.txt', delimiter=',', dtype=np.complex128)

np.savetxt('pr1_out1.txt', A ,fmt="%0.1e")

prt(A,fmt="%0.2f")