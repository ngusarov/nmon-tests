# cython: language_level=3
from numpy.linalg import eigh
import numpy as np
cimport numpy as cnp  # Import C-level NumPy

# Define function to compute eigenvalues and eigenvectors
def compute_eigsh_np(cnp.ndarray[cnp.complex128_t, ndim=2] H):
    """
    Compute eigenvalues and eigenvectors of a Hermitian matrix using NumPy's eigh.
    Accepts a NumPy array and returns two NumPy arrays.
    """
    return eigh(H)
