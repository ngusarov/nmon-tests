

from itertools import product

import numpy as np
import sympy
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.operatorordering import normal_ordered_form

from pymablock import block_diagonalize

symbols = sympy.symbols(r"\omega_{t} \omega_{r} \alpha g", real=True, positive=True)
omega_t, omega_r, alpha, g = symbols

a_t, a_r = BosonOp("a_t"), BosonOp("a_r")

H_0 = (
    -omega_t * Dagger(a_t) * a_t + omega_r * Dagger(a_r) * a_r
    + alpha * Dagger(a_t)**2 * a_t**2 / 2
)

H_p = (
    -g * (Dagger(a_t) - a_t) * (Dagger(a_r) - a_r)
)

def collect_constant(expr):
    expr = normal_ordered_form(expr.expand(), independent=True)
    constant_terms = []
    for term in expr.as_ordered_terms():
        if not term.has(sympy.physics.quantum.Operator):
            constant_terms.append(term)
    return sum(constant_terms)


def to_matrix(ham, basis):
    """Compute the matrix elements"""
    N = len(basis)
    ham = normal_ordered_form(ham.expand(), independent=True)
    all_brakets = product(basis, basis)
    flat_matrix = [
        collect_constant(braket[0] * ham * Dagger(braket[1])) for braket in all_brakets
    ]
    return sympy.Matrix(np.array(flat_matrix).reshape(N, N))


# Construct the matrix Hamiltonian
basis = [
    a_t**i * a_r**j / sympy.sqrt(sympy.factorial(i) * sympy.factorial(j))
    for i in range(3)
    for j in range(3)
]

H_0_matrix = to_matrix(H_0, basis)
H_p_matrix = to_matrix(H_p, basis)

H = H_0_matrix + H_p_matrix

subspaces = {state: n for n, state in enumerate([1, a_t, a_r, a_t * a_r])}
subspace_indices = [subspaces.get(element, 4) for element in basis]
H_tilde, U, U_adjoint = block_diagonalize(
    H, subspace_indices=subspace_indices, symbols=[g]
)

H_tilde.shape

xi = (H_tilde[0, 0, 2] - H_tilde[1, 1, 2] - H_tilde[2, 2, 2] + H_tilde[3, 3, 2])[0, 0]

display_eq(r"\chi", xi)