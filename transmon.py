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

    def __init__(self) -> None:
        self.N = 1
        self.M = 1

        self.CN = 1e-12/2
        self.CM = 1e-12/2
        self.ECN = 1#(e**2/(2*self.CN)) / hplank / 1e9
        self.ECM = 1#(e**2/(2*self.CM)) / hplank / 1e9

        self.kappa = self.CM / (self.CN + self.CM)

        self.EC = ( 2*e**2 / (4 * (self.N*self.M)**2 * (self.CM + self.CN)) ) / hplank / 1e9 # e**2 / 2*C

        self.EJN =  10 #4.5 # GHz "EJN" = Phi0**2/(4*np.pi**2 * L)
        self.EJM =  10 #0.9 # GHz "EJM"

        self.LN = Phi0**2/( 4*np.pi**2 * self.EJN * 1e9 * hplank)
        self.LM = Phi0**2/( 4*np.pi**2 * self.EJM * 1e9 * hplank)

        self.L_total = 1/ (1/(self.N*self.LN + self.M*self.LM))
        self.C_total = self.CN + self.CM

        self.C_from_EC = e**2 / (2*self.EC * hplank * 1e9)


        omega_p = 30 # GHz

        self.ECJN = 1#omega_p**2 / (8 * self.EJN)
        self.ECJM = 1#omega_p**2 / (8 * self.EJM)

        self.dims = 6

    def hamiltonian_calc(self, flux):


        custom = """
        branches:
        - [C,1,0, """ + str(self.ECN) + """] # 0


        - [JJ,1,0, """ + str(self.EJN) + """, """ + str(self.ECJN) + """] # 2
        

        - [JJ,1,0, """ + str(self.EJM) + """, """ + str(self.ECJM) + """] # 3
        """

        self.nmon_circ = scq.Circuit(custom, from_file=False, initiate_sym_calc=True, ext_basis="discretized", basis_completion="heuristic")

        # system_hierarchy = [[1]] 

        # closure_branches = [nmon.branches[i] for i in [1, 2, 4, 5]] #  
        # nmon.configure(closure_branches=closure_branches, system_hierarchy=system_hierarchy, subsystem_trunc_dims = [self.dims]) 

        # # nmon.Φ1 = flux
        # # nmon.ng1 = 0 #ng
        # # nmon.cutoff_n_1 = 6
        # # nmon.cutoff_ext_2 = 6

        # # nmon.generate_subsystems() # generate subsystems to measure interaction 

        # # self.transmon = nmon
        # # self.subsystem_0 = nmon.subsystems[0]
        # # self.subsystem_1 = nmon.subsystems[1]

        # H_ = nmon.hamiltonian().toarray()
        # self.H_scq = qt.Qobj(np.real(H_), dims=[[self.dims],[self.dims]]) + qt.Qobj(np.imag(H_), dims=[[self.dims],[self.dims]]) # Hamiltonian 

        
