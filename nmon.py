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

e = 1.6e-19
hplank = 6.626e-34
Phi0 = 2.067e-15

class Nmon:

    def __init__(self, EJN=1, EJM=1) -> None:
        self.N = 1
        self.M = 1

        self.C_shunt = 1e-13

        # self.CN = 1e-14/2 * 7/4
        # self.CM = 1e-14/2 * 1/4
        # self.ECN = (e**2/(2*self.CN)) / hplank / 1e9
        # self.ECM = (e**2/(2*self.CM)) / hplank / 1e9

        # self.kappa = self.CM / (self.CN + self.CM)

        # self.EC = ( 2*e**2 / (4 * (self.N*self.M)**2 * (self.CM + self.CN)) ) / hplank / 1e9 # e**2 / 2*C

        self.EC_shunt = e**2 / (2 * (self.C_shunt)) / hplank / 1e9 

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

        self.dims = 6

        self.theta_coefs = None

        self.evals = None

        self.left_phi = -np.pi
        self.right_phi = np.pi
        self.N_phi = 100

        self.phi_list = np.linspace(self.left_phi, self.right_phi, self.N_phi)

        self.wavefunctions = None

        self.flux = None
        self.ng = None

    def hamiltonian_calc(self, flux, ng, make_plot=False):

        self.flux = flux
        self.ng = ng

        # custom = """
        # branches:
        # - [C,1,0, """ + str(self.ECN) + """] # 0
        # - [C,1,0, """ + str(self.ECM) + """] # 1

        # - [JJ,2,0, """ + str(self.EJN) + """, """ + str(self.ECJN) + """] # 2
        # - [JJ,1,2, """ + str(self.EJN) + """, """ + str(self.ECJN) + """] # 3

        # - [JJ,3,0, """ + str(self.EJM) + """, """ + str(self.ECJM) + """] # 4
        # - [JJ,4,3, """ + str(self.EJM) + """, """ + str(self.ECJM) + """] # 5
        # - [JJ,1,4, """ + str(self.EJM) + """, """ + str(self.ECJM) + """] # 6
        # """

        # custom = """
        # branches:
        # - [C,1,0, """ + str(self.ECN) + """] # 0
        # - [C,1,0, """ + str(self.ECM) + """] # 1

        # - [JJ,1,0, """ + str(self.EJN) + """, """ + str(self.ECJN) + """] # 2

        # - [JJ,2,0, """ + str(self.EJM) + """, """ + str(self.ECJM) + """] # 4
        # - [JJ,1,2, """ + str(self.EJM) + """, """ + str(self.ECJM) + """] # 5
        # """

        custom = """
        branches:
        - [C,1,0, """ + str(self.EC_shunt) + """] # 0

        - [JJ,1,0, """ + str(self.EJN) + """, """ + str(self.ECJN) + """] # 2

        - [JJ,1,0, """ + str(self.EJM) + """, """ + str(self.ECJM) + """] # 4
        """

        self.nmon_circ = scq.Circuit(custom, from_file=False, initiate_sym_calc=True, ext_basis="discretized", basis_completion="heuristic")

        #----------------
        system_hierarchy = [list(np.arange(1, self.N + self.M))]
        scq.truncation_template(system_hierarchy)


        self.nmon_circ.cutoff_n_1 = 31
        # self.nmon_circ.cutoff_n_2 = 5
        # self.nmon_circ.cutoff_n_3 = 5
        # self.nmon_circ.cutoff_n_4 = 5
        self.nmon_circ.configure(system_hierarchy=system_hierarchy, subsystem_trunc_dims=[6])

        self.nmon_circ.Φ1 = flux
        self.nmon_circ.ng1 = ng
        # self.nmon_circ.ng2 = 0.0
        # self.nmon_circ.ng3 = 0.0
        # self.nmon_circ.ng4 = 0.0

        self.calc_theta_phi_transform()

        ###################################################

        transmon = scq.TunableTransmon(EJmax=self.EJM + self.EJN, EC=self.EC_total, d=0, flux=flux, ng=ng, ncut=31, truncated_dim=6)
        tmon_evals = transmon.eigenvals()
        if make_plot:
            print("tmon", tmon_evals)

        ###################################################


        self.evals = self.nmon_circ.subsystems[0].eigenvals()
        self.bound_state_energies = None
        
        if make_plot:
            print("nmon", self.evals)

        self.calc_potential(make_plot=make_plot)

        self.calc_wavefunctions(make_plot=make_plot)

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
            potential_list[i] = self.nmon_circ.potential_energy(θ1=phi*self.theta_coefs[0],
                                                                # θ2=phi*self.theta_coefs[1],
                                                                # θ3 = phi*self.theta_coefs[2],
                                                                # θ4 = phi*self.theta_coefs[3]
                                                                )

        max_pot = max(potential_list)
        bound_state_energies = []
        while eval < max_pot or len(bound_state_energies) < 3:
        # for i, eval in enumerate(self.evals):
            if eval > max_pot:
                # break
                print(f"Filling with a technically non-bound state {round(eval, 2)} (max pot {round(max_pot, 2)})")
            bound_state_energies.append(eval)
            
        # print("bound_states", bound_state_energies)

        self.bound_state_energies = bound_state_energies.copy()

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
                full_wf = self.nmon_circ.subsystems[0].generate_wf_plot_data(which=i, var_indices=var_indices, \
                                                                            grids_dict=grids_dict, mode='real')
            full_wf = np.einsum('{}->i'.format("i"*(self.M + self.N - 1)), full_wf)

            try:
                imag_full_wf = self.nmon_circ.generate_wf_plot_data(which=i, var_indices=var_indices, \
                                                                            grids_dict=grids_dict, mode='imag')
            except Exception:
                imag_full_wf = self.nmon_circ.subsystems[0].generate_wf_plot_data(which=i, var_indices=var_indices, \
                                                                            grids_dict=grids_dict, mode='imag')
            imag_full_wf = np.einsum('{}->i'.format("i"*(self.M + self.N - 1)), imag_full_wf)

            wavefunctions.append(full_wf+ 1j*imag_full_wf)

            if make_plot:
                plt.plot(self.phi_list, full_wf)
                plt.plot(self.phi_list, imag_full_wf)
                plt.show()
        
        self.wavefunctions = wavefunctions.copy()
    

    def calc_transition_matrix(self, make_plot=False):
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
            plt.figure(figsize=(10, 8))
            plt.imshow(np.absolute(transition_matrix), cmap='viridis', interpolation='nearest')
            plt.colorbar()
            plt.title('Transition Matrix Elements')
            plt.xlabel('Final State Index')
            plt.ylabel('Initial State Index')
            plt.xticks(ticks=np.arange(len(self.bound_state_energies)), labels=[f'{i}' for i in range(len(self.bound_state_energies))])
            plt.yticks(ticks=np.arange(len(self.bound_state_energies)), labels=[f'{i}' for i in range(len(self.bound_state_energies))])
            plt.show()

    
    def calc_transitions(self):

        def find_dominating_transitions(matrix):
            n = matrix.shape[0]  # Number of states
            current_state = 0  # Start from state 0
            transitions = []
            # We will look for a maximum of n-1 transitions to avoid an infinite loop
            for _ in range(n - 1):
                # Find the index of the maximum element in the current row (dominating transition)
                next_state = (current_state+1)+np.argmax(matrix[current_state][current_state+1:])
                if next_state < n and \
                    matrix[current_state, next_state] >= 10*matrix[current_state, current_state]:
                    # Store the transition and its value
                    transitions.append((current_state, next_state, matrix[current_state, next_state]))
                    # Move to the next state
                    current_state = next_state

                if next_state == n-1:
                    break
            return transitions

        # Find and print the dominating transitions starting from state 0
        dominating_transitions = find_dominating_transitions(np.absolute(self.transition_matrix))

        self.transition_freqs = []
        for i, transition in enumerate(dominating_transitions):
            # print(f"Transition from state {transition[0]} to state {transition[1]} with probability {transition[2]:.2f}")
            wij = self.bound_state_energies[transition[1]] - self.bound_state_energies[transition[0]] 
            # print("w{}{}".format(i, i+1), wij)
            self.transition_freqs.append(wij)

        if len(dominating_transitions) > 1:
            # print("w12 - w01", self.transition_freqs[1] - self.transition_freqs[0])
            self.relative_anharm = (self.transition_freqs[1] - self.transition_freqs[0]) / self.transition_freqs[0]

