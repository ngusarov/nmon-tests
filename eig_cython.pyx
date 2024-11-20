# eig_cython.pyx
from scipy.sparse.linalg import eigsh
import numpy as np
cimport numpy as cnp  # Import C-level NumPy

# Tell Cython the type of the array
def compute_eigsh(cnp.ndarray[cnp.float64_t, ndim=2] H, int k=3):
    return eigsh(H, which='SA', k=k)
