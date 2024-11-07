import getpass
import os
import numpy as np
import sys 
os.environ['OMP_NUM_THREADS'] = '1' # to avoid using too many CPUs, important for MPI

if getpass.getuser() == "grasser": # when running from LEM
    os.environ['pRT_input_data_path'] ="/net/lem/data2/pRT_input_data"
    from mpi4py import MPI 
    comm = MPI.COMM_WORLD # important for MPI
    rank = comm.Get_rank() # important for MPI
    import matplotlib
    matplotlib.use('Agg') # disable interactive plotting
elif getpass.getuser() == "natalie": # when testing from Natalie's laptop
    os.environ['pRT_input_data_path'] = "/home/natalie/.local/lib/python3.8/site-packages/petitRADTRANS/input_data_std/input_data"
from target import Target
from retrieval import Retrieval
from parameters import Parameters

# pass configuration as command line argument
# example: config_run.py freechem 200 5
chem = sys.argv[1] # freechem / equchem / quequchem
Nlive= int(sys.argv[2]) # number of live points (integer)
tol= float(sys.argv[3]) # evidence tolerance (float)

def init_retrieval(chem,Nlive,tol):

    brown_dwarf = Target('SP0829')
    output=f'{chem}_N{Nlive}_ev{tol}' # output folder name

    constant_params={} # add if needed
    free_params = {'rv': ([2,20],r'$v_{\rm rad}$'),
                'vsini': ([0,40],r'$v$ sin$i$'),
                'log_g':([3,5],r'log $g$'),
                'epsilon_limb': ([0.2,1], r'$\epsilon_\mathrm{limb}$')} # limb-darkening coefficient (0-1)

    pt_params={'T0' : ([1000,4000], r'$T_0$'), # bottom of the atmosphere (hotter)
            'T1' : ([0,4000], r'$T_1$'),
            'T2' : ([0,4000], r'$T_2$'),
            'T3' : ([0,4000], r'$T_3$'),
            'T4' : ([0,4000], r'$T_4$'),} # top of atmosphere (cooler)
    free_params.update(pt_params)

    # if equilibrium chemistry, define [Fe/H], C/O, and isotopologue ratios
    if chem in ['equchem','quequchem']:
        chemistry={'C/O':([0,1], r'C/O'), 
                'Fe/H': ([-1.5,1.5], r'[Fe/H]'), 
                'log_C12_13_ratio': ([1,12], r'log $\mathrm{^{12}C/^{13}C}$'), 
                'log_O16_18_ratio': ([1,12], r'log $\mathrm{^{16}O/^{18}O}$'), 
                'log_O16_17_ratio': ([1,12], r'log $\mathrm{^{16}O/^{17}O}$')}
            
    if chem=='quequchem': # quenched equilibrium chemistry, define quench pressure of certain species
        quenching={'log_Pqu_CO_CH4': ([-6,2], r'log P$_{qu}$(CO,CH$_4$,H$_2$O)'),
                'log_Pqu_NH3': ([-6,2], r'log P$_{qu}$(NH$_3$)'),
                'log_Pqu_HCN': ([-6,2], r'log P$_{qu}$(HCN)')}  
        chemistry.update(quenching)
        
    # if free chemistry, define VMRs
    if chem=='freechem': 
        chemistry={'log_H2O':([-12,-1],r'log H$_2$O'),
                'log_12CO':([-12,-1],r'log $^{12}$CO'),
                'log_13CO':([-12,-1],r'log $^{13}$CO'),
                'log_C18O':([-12,-1],r'log C$^{18}$O'),
                'log_C17O':([-12,-1],r'log C$^{17}$O'),
                'log_CH4':([-12,-1],r'log CH$_4$'),
                'log_NH3':([-12,-1],r'log NH$_3$'),
                'log_HCN':([-12,-1],r'log HCN'),
                'log_HF':([-12,-1],r'log HF'),
                'log_H2(18)O':([-12,-1],r'log H$_2^{18}$O'),
                'log_H2S':([-12,-1],r'log H$_2$S')}

    free_params.update(chemistry)
    parameters = Parameters(free_params, constant_params)
    cube = np.random.rand(parameters.n_params)
    parameters(cube)
    retrieval=Retrieval(target=brown_dwarf,parameters=parameters,output_name=output,chemistry=chem)

    return retrieval

retrieval=init_retrieval(chem=chem,Nlive=Nlive,tol=tol)
retrieval.run_retrieval(N_live_points=Nlive,evidence_tolerance=tol)