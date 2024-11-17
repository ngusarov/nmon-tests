from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import scqubits as scq
import qutip as qt
import scipy
from tqdm import tqdm
from scipy.optimize import curve_fit
import pandas
from matplotlib import colormaps
import random
from scipy.integrate import quad
import sccircuitbuilder as circuitbuilder
from IPython.display import display, Markdown, HTML, clear_output
from sympy import *
import scqubits.core.units as units
import scipy.sparse.linalg as spla
import sympy as sp
import itertools
import tensorflow as tf

from itertools import combinations_with_replacement
import heapq

e = 1.6e-19
hplank = 6.626e-34
Phi0 = 2.067e-15

class Nmon:

    def __init__(self, N=1, M=1, EJN=1, EJM=1, EC_shunt=0.15) -> None:
        self.N = N
        self.M = M

        # self.C_shunt = 1e-13

        self.EC_shunt = EC_shunt #e**2 / (2 * (self.C_shunt)) / hplank / 1e9

        self.C_shunt = e**2 / (2 * (self.EC_shunt * hplank * 1e9))

        self.EJN =  EJN # GHz "EJN" = Phi0**2/(4*np.pi**2 * L)
        self.EJM =  EJM # GHz "EJM"

        self.LN = Phi0**2/( 4*np.pi**2 * self.EJN * 1e9 * hplank)
        self.LM = Phi0**2/( 4*np.pi**2 * self.EJM * 1e9 * hplank)

        self.L_total = 1/ ( 1/(self.N*self.LN) + 1/(self.M*self.LM) )

        omega_p = 30 # GHz # !!!!!!!!!!!!!!!!!!!!!!!!!

        self.ECJN = omega_p**2 / (8 * self.EJN)
        self.ECJM = omega_p**2 / (8 * self.EJM)

        self.CJN = e**2 / (2 * self.ECJN * 1e9 * hplank)
        self.CJM = e**2 / (2 * self.ECJM * 1e9 * hplank)

        self.C_total = self.C_shunt + 1/(self.N/self.CJN) + 1/(self.M/self.CJM)

        self.EC_total = e**2 / (2 * self.C_total) / hplank / 1e9 

        self.dims = None

        self.theta_coefs = None

        self.H = None
        self.H_sparse = None
        self.sym_hamiltonian = None
        self.evecs = None
        self.evals = None

        self.left_phi = -np.pi
        self.right_phi = np.pi
        self.N_phi = 100

        self.phi_list = np.linspace(self.left_phi, self.right_phi, self.N_phi)

        self.wavefunctions = None

        self.flux = None
        self.ng = None

        self.flag_calc_transitions = True
        self.ready_dominating_transitions = None

        # ----------------------

        self.dims = None

        custom = None

        if self.N ==1 and self.M == 3:
            custom = """
            branches:
            - [C,1,0, """ + str(self.EC_shunt) + """] # 0

            - [JJ,1,0, """ + str(self.EJN) + """, """ + str(self.ECJN) + """] # 2

            - [JJ,3,0, """ + str(self.EJM) + """, """ + str(self.ECJM) + """] # 4
            - [JJ,2,3, """ + str(self.EJM) + """, """ + str(self.ECJM) + """] # 5
            - [JJ,1,2, """ + str(self.EJM) + """, """ + str(self.ECJM) + """] # 5
            """
        
        elif self.N ==1 and self.M == 4:
            custom = """
            branches:
            - [C,1,0, """ + str(self.EC_shunt) + """] # 0

            - [JJ,1,0, """ + str(self.EJN) + """, """ + str(self.ECJN) + """] # 2

            - [JJ,4,0, """ + str(self.EJM) + """, """ + str(self.ECJM) + """] # 4
            - [JJ,3,4, """ + str(self.EJM) + """, """ + str(self.ECJM) + """] # 5
            - [JJ,2,3, """ + str(self.EJM) + """, """ + str(self.ECJM) + """] # 5
            - [JJ,1,2, """ + str(self.EJM) + """, """ + str(self.ECJM) + """] # 5
            """

        elif self.N ==2 and self.M == 3:
            custom = """
            branches:
            - [C,1,0, """ + str(self.EC_shunt) + """] # 0

            - [JJ,2,0, """ + str(self.EJN) + """, """ + str(self.ECJN) + """] # 2
            - [JJ,1,2, """ + str(self.EJN) + """, """ + str(self.ECJN) + """] # 2

            - [JJ,3,0, """ + str(self.EJM) + """, """ + str(self.ECJM) + """] # 4
            - [JJ,4,3, """ + str(self.EJM) + """, """ + str(self.ECJM) + """] # 5
            - [JJ,1,4, """ + str(self.EJM) + """, """ + str(self.ECJM) + """] # 5
            """
        
        elif self.N ==2 and self.M == 2:
            custom = """
            branches:
            - [C,1,0, """ + str(self.EC_shunt) + """] # 0

            - [JJ,2,0, """ + str(self.EJN) + """, """ + str(self.ECJN) + """] # 2
            - [JJ,1,2, """ + str(self.EJN) + """, """ + str(self.ECJN) + """] # 2

            - [JJ,3,0, """ + str(self.EJM) + """, """ + str(self.ECJM) + """] # 4
            - [JJ,1,3, """ + str(self.EJM) + """, """ + str(self.ECJM) + """] # 5
            """

        elif self.N ==1 and self.M == 2:

            custom = """
            branches:
            - [C,1,0, """ + str(self.EC_shunt) + """] # 0

            - [JJ,1,0, """ + str(self.EJN) + """, """ + str(self.ECJN) + """] # 2

            - [JJ,2,0, """ + str(self.EJM) + """, """ + str(self.ECJM) + """] # 4
            - [JJ,1,2, """ + str(self.EJM) + """, """ + str(self.ECJM) + """] # 5
            """

        elif self.N ==1 and self.M == 1:
            custom = """
            branches:
            - [C,1,0, """ + str(self.EC_shunt) + """] # 0

            - [JJ,1,0, """ + str(self.EJN) + """, """ + str(self.ECJN) + """] # 2

            - [JJ,1,0, """ + str(self.EJM) + """, """ + str(self.ECJM) + """] # 4
            """

        self.nmon_circ = scq.Circuit(custom, from_file=False, initiate_sym_calc=True, ext_basis="discretized", basis_completion="heuristic")

        #----------------
        # system_hierarchy = [0]#[list(np.arange(1, self.N + self.M))]
        # scq.truncation_template(system_hierarchy)

        if self.N+self.M-1 == 1:
            self.nmon_circ.cutoff_n_1 = 31
        elif self.N+self.M-1 == 2:
            self.nmon_circ.cutoff_n_1 = 10
            self.nmon_circ.cutoff_n_2 = 10
        elif self.N + self.M-1 == 3:
            self.nmon_circ.cutoff_n_1 = 2
            self.nmon_circ.cutoff_n_2 = 2
            self.nmon_circ.cutoff_n_3 = 2
        elif self.N + self.M-1 == 4:
            self.nmon_circ.cutoff_n_1 = 2
            self.nmon_circ.cutoff_n_2 = 2
            self.nmon_circ.cutoff_n_3 = 2
            self.nmon_circ.cutoff_n_4 = 2
        elif self.N + self.M-1 == 5:
            self.nmon_circ.cutoff_n_1 = 1
            self.nmon_circ.cutoff_n_2 = 1
            self.nmon_circ.cutoff_n_3 = 1
            self.nmon_circ.cutoff_n_4 = 1
            self.nmon_circ.cutoff_n_5 = 1
        
        # self.nmon_circ.configure(system_hierarchy=system_hierarchy, subsystem_trunc_dims=[num_levels])
            


    def hamiltonian_calc(self, flux, ng, make_plot=False, num_levels=6, just_H=False, cutoff=6):

        if self.N+self.M-1 == 1:
            self.nmon_circ.cutoff_n_1 = cutoff
        elif self.N+self.M-1 == 2:
            self.nmon_circ.cutoff_n_1 = cutoff
            self.nmon_circ.cutoff_n_2 = cutoff
        elif self.N + self.M-1 == 3:
            self.nmon_circ.cutoff_n_1 = cutoff
            self.nmon_circ.cutoff_n_2 = cutoff
            self.nmon_circ.cutoff_n_3 = cutoff
        elif self.N + self.M-1 == 4:
            self.nmon_circ.cutoff_n_1 = cutoff
            self.nmon_circ.cutoff_n_2 = cutoff
            self.nmon_circ.cutoff_n_3 = cutoff
            self.nmon_circ.cutoff_n_4 = cutoff
        elif self.N + self.M-1 == 5:
            self.nmon_circ.cutoff_n_1 = cutoff
            self.nmon_circ.cutoff_n_2 = cutoff
            self.nmon_circ.cutoff_n_3 = cutoff
            self.nmon_circ.cutoff_n_4 = cutoff
            self.nmon_circ.cutoff_n_5 = cutoff

        self.dims = num_levels

        self.flux = flux
        self.ng = ng


        self.nmon_circ.Φ1 = flux

        self.nmon_circ.ng1 = ng[0]
        if self.N+self.M-1 > 1:
            self.nmon_circ.ng2 = ng[1]
        if self.N + self.M-1 > 2:
            self.nmon_circ.ng3 = ng[2]
        if self.N + self.M-1 > 3:
            self.nmon_circ.ng4 = ng[3]
        if self.N + self.M-1 > 4:
            self.nmon_circ.ng4 = ng[4]

        self.calc_theta_phi_transform()

        ###################################################

        if self.M == 1 and self.N == 1 and make_plot:
            transmon = scq.TunableTransmon(EJmax=self.EJM + self.EJN, EC=self.EC_total,
                                            d=(self.EJM - self.EJN)/(self.EJM + self.EJN),
                                              flux=flux, ng=ng, ncut=31, truncated_dim=self.dims)
            tmon_evals = transmon.eigenvals()
            
            print("tmon", tmon_evals)

        ###################################################

        



        self.H = self.nmon_circ.hamiltonian() # sparse array (or not)
        print("type: ", type(self.H))
        # Convert to dense NumPy array
        if type(self.H) == scipy.sparse._csc.csc_matrix:
            self.H_sparse = self.H
            self.H = self.H.toarray()
        
        if just_H:
            return self.H

        self.sym_hamiltonian  = self.nmon_circ.sym_hamiltonian(return_expr=True)



        # Solve eigenvalue problem
        eigenvalues, eigenvectors = tf.linalg.eigh(self.H)
        # Convert to numpy for inspection (if needed)
        eigenvalues = eigenvalues.numpy()
        eigenvectors = eigenvectors.numpy()
        
        # Sort eigenvalues and eigenvectors
        idx = np.argsort(eigenvalues.real)

        self.evals = eigenvalues[idx]
        self.evecs = eigenvectors[:, idx]
    
        self.evals = self.evals[:self.dims]
        self.evecs = self.evecs[:, :self.dims]

        self.bound_state_energies = self.evals.copy()
        




        if make_plot:
            print("nmon", self.evals)

            self.calc_potential(make_plot=make_plot)

        # self.calc_wavefunctions(make_plot=make_plot)

        self.transition_matrix = None
        self.calc_transition_matrix(make_plot=make_plot)

        self.transition_freqs = []
        self.relative_anharm = None

        self.calc_transitions()



    def calc_theta_phi_transform(self):
        Matr = self.nmon_circ.transformation_matrix
        Matr_inv = np.linalg.inv(Matr)

        main_node = 1
        M_nodes = np.arange(self.N+1, self.M+self.N) # should be sorted
        N_nodes = np.arange(main_node+1, self.N+1) # should be sorted

        solution_coefs = np.zeros(self.N + self.M) # coefs for phi_i for i from 0 to M+N-1

        solution_coefs[1] = self.N*self.M

        for position, node_num in enumerate(M_nodes):
            solution_coefs[node_num] = self.N * (position + 1)

        for position, node_num in enumerate(N_nodes):
            solution_coefs[node_num] = self.M * (position + 1)

        self.theta_coefs = Matr_inv@solution_coefs[1:] # theta coefs

    def calc_potential(self, make_plot=False):
        potential_list = np.zeros_like(self.phi_list)
        for i, phi in enumerate(self.phi_list):
            if self.N + self.M-1 == 1:
                potential_list[i] = self.nmon_circ.potential_energy(θ1=phi*self.theta_coefs[0],
                                                                    # θ2=phi*self.theta_coefs[1],
                                                                    # θ3 = phi*self.theta_coefs[2],
                                                                    # θ4 = phi*self.theta_coefs[3]
                                                                    )
            elif self.N + self.M-1 == 2:
                potential_list[i] = self.nmon_circ.potential_energy(θ1=phi*self.theta_coefs[0],
                                                                    θ2=phi*self.theta_coefs[1],
                                                                    # θ3 = phi*self.theta_coefs[2],
                                                                    # θ4 = phi*self.theta_coefs[3]
                                                                    )
            elif self.N + self.M-1 == 3:
                potential_list[i] = self.nmon_circ.potential_energy(θ1=phi*self.theta_coefs[0],
                                                                    θ2=phi*self.theta_coefs[1],
                                                                    θ3 = phi*self.theta_coefs[2],
                                                                    # θ4 = phi*self.theta_coefs[3]
                                                                    )
            elif self.N + self.M-1 == 4:
                potential_list[i] = self.nmon_circ.potential_energy(θ1=phi*self.theta_coefs[0],
                                                                    θ2=phi*self.theta_coefs[1],
                                                                    θ3 = phi*self.theta_coefs[2],
                                                                    θ4 = phi*self.theta_coefs[3]
                                                                    )

        # max_pot = max(potential_list)
        # bound_state_energies = []
        # i=0
        # while (self.evals[i] <= max_pot or len(bound_state_energies) < 100):
        #     eval = self.evals[i]
        # # for i, eval in enumerate(self.evals):
        #     if eval > max_pot:
        #         pass
        #         # break
        #         # print(f"Filling with a technically non-bound state {round(eval, 2)} (max pot {round(max_pot, 2)})", "Len", len(bound_state_energies))
        #     bound_state_energies.append(eval)

        #     i+=1
        #     if i >= len(self.evals):
        #         break
            
        # # print("bound_states", bound_state_energies)

        # self.bound_state_energies = bound_state_energies.copy()

        shifted_evals = self.evals #+ 2*np.abs(np.min(evals))
        shifted_potential_list = potential_list# - min(potential_list)

        if make_plot:
            plt.plot(self.phi_list, shifted_potential_list)

            for eval in shifted_evals:
                plt.axhline(eval)

            plt.grid(which='both')
            plt.title("EC={}".format(self.EC_total))
            # plt.ylim(-10, 10)
            plt.show()


    def calc_wavefunctions(self, make_plot=False):

        make_plot = False

        if not self.flag_calc_transitions:
            return 0

        theta_grids = [scq.core.discretization.Grid1d(self.left_phi*self.theta_coefs[i], self.right_phi*self.theta_coefs[i], self.N_phi) for i in range(self.N + self.M - 1)]
        wavefunctions = []

        var_indices = list(np.arange(1, self.N + self.M))
        grids_dict = {}
        for j in range(1, self.N + self.M):
            grids_dict[j] = theta_grids[j-1]

        for i in range(len(self.bound_state_energies)):
            try:
                full_wf = self.nmon_circ.generate_wf_plot_data(which=i, var_indices=var_indices, \
                                                                            grids_dict=grids_dict, mode='real')
            except Exception:
                try:
                    full_wf = self.nmon_circ.subsystems[0].generate_wf_plot_data(which=i, var_indices=var_indices, \
                                                                                grids_dict=grids_dict, mode='real')
                except Exception:
                    self.flag_calc_transitions = False
                    return 0 

            full_wf = np.einsum('{}->i'.format("i"*(self.M + self.N - 1)), full_wf)

            try:
                imag_full_wf = self.nmon_circ.generate_wf_plot_data(which=i, var_indices=var_indices, \
                                                                            grids_dict=grids_dict, mode='imag')
            except Exception:
                try:
                    imag_full_wf = self.nmon_circ.subsystems[0].generate_wf_plot_data(which=i, var_indices=var_indices, \
                                                                                grids_dict=grids_dict, mode='imag')
                except Exception:
                    self.flag_calc_transitions = False
                    return 0 
                
            imag_full_wf = np.einsum('{}->i'.format("i"*(self.M + self.N - 1)), imag_full_wf)

            wavefunctions.append(full_wf+ 1j*imag_full_wf)

            if make_plot:
                plt.plot(self.phi_list, full_wf)
                plt.plot(self.phi_list, imag_full_wf)
                plt.show()
        
        self.wavefunctions = wavefunctions.copy()
    

    def calc_transition_matrix_phase(self, make_plot=False):
        # Assume theta_grids and nmon are properly defined earlier in your context
        # For each wavefunction index i from 0 to 3 (first four wavefunctions)
        transition_matrix = np.zeros((len(self.bound_state_energies), len(self.bound_state_energies)), dtype=np.complex128)  # Initialize the transition matrix

        for i in range(len(self.bound_state_energies)):  # Iterate over the initial state indices
            for j in range(len(self.bound_state_energies)):  # Iterate over the final state indices
                # Generate the wavefunction data for the j-th final state
                # Compute the matrix element for the transition i -> j

                transition_matrix[i, j] = np.sum(self.wavefunctions[i].conj() * \
                                                ((self.M-1)*self.EJM*np.sin(self.N*self.phi_list) + self.EJM*np.sin(2*np.pi*self.flux + self.N*self.phi_list) + \
                                                self.N*self.EJN*np.sin(self.M*self.phi_list))*\
                                                self.wavefunctions[j])

        self.transition_matrix = transition_matrix.copy()
        if make_plot:
            # Plotting the matrix of transition elements
            plt.imshow(np.absolute(transition_matrix), cmap='viridis', interpolation='nearest')
            plt.colorbar()
            plt.title('Transition Matrix Elements')
            plt.xlabel('Final State Index')
            plt.ylabel('Initial State Index')
            plt.xticks(ticks=np.arange(len(self.bound_state_energies)), labels=[f'{i}' for i in range(len(self.bound_state_energies))])
            plt.yticks(ticks=np.arange(len(self.bound_state_energies)), labels=[f'{i}' for i in range(len(self.bound_state_energies))])
            plt.show()

    def calc_transition_matrix(self, make_plot=False):
        eigenvectors = self.evecs
        eigenvalues = self.evals


        # Parameters
        ncut = self.nmon_circ.cutoff_n_1  # Charge basis cutoff
        n_vals = np.arange(-ncut, ncut + 1)
        dim_size = len(n_vals)  # Should be 7

        # Number of variables (phases)
        N = self.N  # Replace with your actual N
        M = self.M  # Replace with your actual M
        num_vars = N + M - 1  # Total number of phase variables (e.g., 3)

        n_grids = np.meshgrid(*([n_vals] * num_vars), indexing='ij')

        # Generate all possible combinations of charge states
        state_list = list(itertools.product(n_vals, repeat=num_vars))
        N_states = len(state_list)  # Should be 7 ** num_vars

        # Mapping from multi-dimensional index to 1D index
        index_map = {state: idx for idx, state in enumerate(state_list)}

        # Mapping from 1D index to multi-dimensional index
        inverse_index_map = {idx: state for idx, state in enumerate(state_list)}

        num_levels = eigenvectors.shape[1]  # Number of energy levels

        # Assuming eigenvectors is of shape (N_states, num_levels)
        dim_sizes = [dim_size] * num_vars
        eigenvectors_md_shape = dim_sizes + [num_levels]
        eigenvectors_md = np.zeros(eigenvectors_md_shape, dtype=complex)

        # Populate the multi-dimensional eigenvector arrays
        for idx in range(N_states):
            indices = inverse_index_map[idx]
            indices_array = np.array(indices) + ncut  # Adjust indices to start from 0
            eigenvectors_md[tuple(indices_array)] = eigenvectors[idx, :]


        # Generate symbolic phase variables theta1, theta2, ..., thetan
        n_symbols = sp.symbols(f'n1:{num_vars+1}')  # Generates theta1 to thetan
        ng_symbols = sp.symbols(f'n_g1:{num_vars+1}')  # Generates theta1 to thetan
        _2pi_Phi1 = sp.symbols('(2πΦ_{1})')

        # Compute derivatives with respect to each theta_i
        derivatives = []
        for ng_sym in ng_symbols:
            dH_dng = self.sym_hamiltonian.diff(ng_sym)
            derivatives.append(dH_dng)

        # Sum all the derivatives to get the gradient
        gradient_H = sum(derivatives)

        gradient_H = gradient_H.subs({_2pi_Phi1: 2*np.pi*self.flux})
        for i, ng_sym in enumerate(ng_symbols):
            gradient_H = gradient_H.subs({ng_sym: self.ng[i]})

        # Convert the symbolic expression to a numerical function
        gradient_func = sp.lambdify(n_symbols, gradient_H, modules=['numpy'])

        G_ng = gradient_func(*n_grids)

        transition_matrix = np.zeros((num_levels, num_levels), dtype=float)

        for i in range(num_levels):
            psi_i = eigenvectors_md[..., i]
            for j in range(num_levels):
                psi_j = eigenvectors_md[..., j]
                # Compute the element-wise product and sum over all indices
                M_ij = np.sum(np.conj(psi_i) * G_ng * psi_j) / ( np.sum(np.conj(psi_i) * psi_i) * np.sum(np.conj(psi_j) * psi_j) )
                transition_matrix[i, j] = np.real(M_ij)

        self.transition_matrix = transition_matrix

        if make_plot:

            transition_matrix = np.abs(transition_matrix)

            # Plot heatmap
            fig, ax = plt.subplots(figsize=(11, 10))
            cax = ax.imshow(transition_matrix, cmap='GnBu', interpolation='nearest')

            # Add text annotations for each cell
            for i in range(transition_matrix.shape[0]):
                for j in range(transition_matrix.shape[1]):
                    ax.text(j, i, f'{transition_matrix[i, j]:.2E}', ha='center', va='center', color='black')

            # Colorbar
            cbar = fig.colorbar(cax)
            plt.show()

            
            diagonal_elements = np.abs(np.diag(transition_matrix))
            mask = np.abs(transition_matrix) >= 1e-2 #diagonal_elements[:, None]
            np.fill_diagonal(mask, 0)
            result_matrix = mask.astype(int)
            plt.imshow(np.absolute(result_matrix[:, :]), cmap='viridis', interpolation='nearest')
            plt.colorbar()
            plt.show()

    
    def calc_transitions(self):

        def find_dominating_transitions(matrix):
            n = matrix.shape[0]  # Number of states

            transitions = []
            start_state = 0
            while len(transitions) == 0 and start_state < n-1: 

                current_state = start_state  # Start from state 0
                

                # We will look for a maximum of n-1 transitions to avoid an infinite loop
                for _ in range(n - 1):

                    # Find the index of the maximum element in the current row (dominating transition)
                    # next_state = (current_state+1)+np.argmax(matrix[current_state][current_state+1:])
                    
                    # print(matrix[current_state][current_state+1:])
                    # print(matrix[current_state, current_state])

                    # print(np.where(np.array(matrix[current_state][current_state+1:]) 
                    #                                            >= 1e3*matrix[current_state, current_state]))
                    
                    next_state_index = np.where(np.array(matrix[current_state][current_state+1:]) 
                                                            >= 1e-2 )[0]
                    
                    if len(next_state_index) == 0:
                        break

                    next_state = (current_state+1) + next_state_index[0]

                    if next_state < n:
                        # Store the transition and its value
                        transitions.append((current_state, next_state, matrix[current_state, next_state]))
                        # Move to the next state
                        current_state = next_state

                    if next_state >= n-1:
                        break

                start_state += 1

            return transitions

        dominating_transitions = None
        if self.flag_calc_transitions:
            # Find and print the dominating transitions starting from state 0
            dominating_transitions = find_dominating_transitions(np.absolute(self.transition_matrix))
        elif self.ready_dominating_transitions == None:
            dominating_transitions = [[0, 1, 0], [1, 2, 0], [2, 3, 0]]
        else:
            dominating_transitions = self.ready_dominating_transitions

        self.ready_dominating_transitions = dominating_transitions

        self.transition_freqs = []
        for i, transition in enumerate(dominating_transitions):
            # print(f"Transition from state {transition[0]} to state {transition[1]} with probability {transition[2]:.2f}")
            wij = self.bound_state_energies[transition[1]] - self.bound_state_energies[transition[0]] 
            # print("w{}{}".format(i, i+1), wij)
            self.transition_freqs.append(wij)

        if len(self.transition_freqs) == 0:
            self.transition_freqs = [self.evals[1] - self.evals[0]]

        if len(dominating_transitions) > 1:
            # print("w12 - w01", self.transition_freqs[1] - self.transition_freqs[0])
            self.relative_anharm = (self.transition_freqs[1] - self.transition_freqs[0]) / self.transition_freqs[0]



def compute_cutoff(EJN, EJM, EC, cutoff_space=[2, 10]):
    """Logarithmically adjust the cutoff based on max(EJN/EC, EJM/EC)."""
    ratio = max(EJN / EC, EJM / EC)
    # Map log10(ratio) from [log10(1), log10(100)] to [2, 10]
    cutoff = np.interp(np.log10(ratio), [0, 2], cutoff_space)
    return int(np.ceil(cutoff))