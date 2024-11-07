import os
from pRT_model import pRT_spectrum
import figures as figs
from covariance import *
from log_likelihood import *

import numpy as np
import pymultinest
import pathlib
import pickle
from petitRADTRANS import Radtrans
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # pRT warning
#warnings.filterwarnings("ignore", category=np.linalg.LinAlgError) 

class Retrieval:

    def __init__(self,target,parameters,output_name,chemistry='freechem'):
        
        self.target=target
        self.data_wave,self.data_flux,self.data_err=target.load_spectrum()
        self.mask_isfinite=target.get_mask_isfinite() # mask nans, shape (orders,detectors)
        self.K2166=target.K2166
        self.parameters=parameters
        self.chemistry=chemistry # freechem/equchem/quequchem
        self.species=self.get_species(param_dict=self.parameters.params,chemistry=self.chemistry)

        self.n_orders, self.n_dets, _ = self.data_flux.shape # shape (orders,detectors,pixels)
        self.n_params = len(parameters.free_params)
        self.output_name=output_name
        self.cwd = os.getcwd()
        self.output_dir = pathlib.Path(f'{self.cwd}/{self.target.name}/{self.output_name}')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.lbl_opacity_sampling=3
        self.n_atm_layers=50
        self.pressure = np.logspace(-6,2,self.n_atm_layers)  # like in deRegt+2024

        self.Cov = np.empty((self.n_orders,self.n_dets), dtype=object) # covariance matrix
        for i in range(self.n_orders):
            for j in range(self.n_dets):
                mask_ij = self.mask_isfinite[i,j] # only finite pixels
                if not mask_ij.any(): # skip empty order/detector pairs
                    continue
                self.Cov[i,j] = Covariance(err=self.data_err[i,j,mask_ij]) # simple diagonal covariance matrix
    
        self.LogLike = LogLikelihood(retrieval_object=self,scale_flux=True,scale_err=True)

        # load atmosphere objects here and not in likelihood/pRT_model to make it faster
        # redo atmosphere objects when introdocuing new species
        self.atmosphere_objects=self.get_atmosphere_objects()
        self.callback_label='live_' # label for plots
        self.prefix='pmn_'
        self.color1=target.color1

    def get_species(self,param_dict,chemistry): # get pRT species name from parameters dict
        species_info = pd.read_csv(os.path.join('species_info.csv'), index_col=0)
        if chemistry=='freechem':
            self.chem_species=[]
            for par in param_dict:
                if 'log_' in par: # get all species in params dict, they are in log, ignore other log values
                    if par in ['log_g','log_Kzz','log_P_base_gray','log_opa_base_gray',
                               'log_C12_13_ratio','log_O16_17_ratio','log_O16_18_ratio',
                               'log_Pqu_CO_CH4','log_Pqu_NH3','log_Pqu_HCN']: # skip
                        pass
                    else:
                        self.chem_species.append(par)
            species=[]
            for chemspec in self.chem_species:
                species.append(species_info.loc[chemspec[4:],'pRT_name'])
        elif chemistry in ['equchem','quequchem']:
            self.chem_species=['H2O','12CO','13CO','C18O','C17O','CH4','NH3',
                         'HCN','H2(18)O','H2S','CO2','HF','OH'] # HF, OH not in pRT chem equ table
            species=[]
            for chemspec in self.chem_species:
                species.append(species_info.loc[chemspec,'pRT_name'])
        return species

    def get_atmosphere_objects(self,redo=False):

        atmosphere_objects=[]
        file=pathlib.Path(f'atmosphere_objects.pickle')
        if file.exists() and redo==False:
            with open(file,'rb') as file:
                atmosphere_objects=pickle.load(file)
                return atmosphere_objects
        else:
            for order in range(self.n_orders):
                wl_pad=7 # wavelength padding because spectrum is not wavelength shifted yet
                wlmin=np.min(self.K2166[order])-wl_pad
                wlmax=np.max(self.K2166[order])+wl_pad
                wlen_range=np.array([wlmin,wlmax])*1e-3 # nm to microns

                atmosphere = Radtrans(line_species=self.species,
                                    rayleigh_species = ['H2', 'He'],
                                    continuum_opacities = ['H2-H2', 'H2-He'],
                                    wlen_bords_micron=wlen_range, 
                                    mode='lbl',
                                    lbl_opacity_sampling=self.lbl_opacity_sampling) # take every nth point (=3 in deRegt+2024)
                
                atmosphere.setup_opa_structure(self.pressure)
                atmosphere_objects.append(atmosphere)
            with open(file,'wb') as file:
                pickle.dump(atmosphere_objects,file)
            return atmosphere_objects

    def PMN_lnL(self,cube=None,ndim=None,nparams=None):
        self.model_object=pRT_spectrum(self)
        self.model_flux=self.model_object.make_spectrum()
        for j in range(self.n_orders): # reset covariance matrix
            for k in range(self.n_dets):
                if not self.mask_isfinite[j,k].any(): # skip empty order/detector
                    continue
                self.Cov[j,k].cov_reset()
        ln_L = self.LogLike(self.model_flux, self.Cov) # retrieve log-likelihood
        return ln_L

    def PMN_run(self,N_live_points=400,evidence_tolerance=0.5,resume=True):
        pymultinest.run(LogLikelihood=self.PMN_lnL,Prior=self.parameters,n_dims=self.parameters.n_params, 
                        outputfiles_basename=f'{self.output_dir}/{self.prefix}', 
                        verbose=True,const_efficiency_mode=True, sampling_efficiency = 0.5,
                        n_live_points=N_live_points,resume=resume,
                        evidence_tolerance=evidence_tolerance, # default is 0.5, high number -> stops earlier
                        dump_callback=self.PMN_callback,n_iter_before_update=10)

    def PMN_callback(self,n_samples,n_live,n_params,live_points,posterior, 
                    stats,max_ln_L,ln_Z,ln_Z_err,nullcontext):
        self.bestfit_params = posterior[np.argmax(posterior[:,-2]),:-2] # parameters of best-fitting model
        self.posterior = posterior[:,:-2] # remove last 2 columns
        self.params_dict,self.model_flux=self.get_params_and_spectrum()
        figs.summary_plot(self)
        if self.chemistry in ['equchem','quequchem']:
            figs.VMR_plot(self)
     
    def PMN_analyse(self):
        analyzer = pymultinest.Analyzer(n_params=self.parameters.n_params, 
                                        outputfiles_basename=f'{self.output_dir}/{self.prefix}')  # set up analyzer object
        stats = analyzer.get_stats()
        self.posterior = analyzer.get_equal_weighted_posterior() # equally-weighted posterior distribution
        self.posterior = self.posterior[:,:-1] # shape 
        np.save(f'{self.output_dir}/{self.callback_label}posterior.npy',self.posterior)
        self.bestfit_params = np.array(stats['modes'][0]['maximum a posterior']) # read params of best-fitting model, highest likelihood
        self.lnZ = stats['nested importance sampling global log-evidence']

    def get_quantiles(self,posterior):
        # input entire posterior of all retrieved parameters
        quantiles = np.array([np.percentile(posterior[:,i], [16.0,50.0,84.0], axis=-1) for i in range(posterior.shape[1])])
        medians=quantiles[:,1] # median of all params
        plus_err=quantiles[:,2]-medians # +error
        minus_err=quantiles[:,0]-medians # -error
        return medians,minus_err,plus_err

    def get_params_and_spectrum(self): 
        
        # make dict of constant params + evaluated params + their errors
        self.params_dict=self.parameters.constant_params.copy() # initialize dict with constant params
        medians,minus_err,plus_err=self.get_quantiles(self.posterior)

        for i,key in enumerate(self.parameters.param_keys):
            self.params_dict[key]=medians[i] # add median of evaluated params (more robust than bestfit)
            
        # add errors in a different loop to avoid messing up order of params (needed later for indexing)
        for i,key in enumerate(self.parameters.param_keys):
            self.params_dict[f'{key}_err']=(minus_err[i],plus_err[i]) # add errors of evaluated params

        # create final spectrum
        self.model_object=pRT_spectrum(self)
        model_flux0=self.model_object.make_spectrum()
        if self.chemistry=='freechem':
            self.params_dict['[Fe/H]']=self.model_object.FeH
            self.params_dict['C/O']=self.model_object.CO

        # get scaling parameters phi_ij and s2_ij of bestfit model through likelihood
        self.log_likelihood = self.LogLike(model_flux0, self.Cov)
        self.params_dict['phi_ij']=self.LogLike.phi
        self.params_dict['s2_ij']=self.LogLike.s2
        if self.callback_label=='final_':
            self.params_dict['chi2']=self.LogLike.chi2_0_red # save reduced chi^2 of fiducial model
            self.params_dict['lnZ']=self.lnZ # save lnZ of fiducial model

        self.model_flux=np.zeros_like(model_flux0)
        phi_ij=self.params_dict['phi_ij']
        for order in range(self.n_orders):
            for det in range(self.n_dets):
                self.model_flux[order,det]=phi_ij[order,det]*model_flux0[order,det] # scale model accordingly
        
        spectrum=np.full(shape=(2048*7*3,2),fill_value=np.nan)
        spectrum[:,0]=self.data_wave.flatten()
        spectrum[:,1]=self.model_flux.flatten()

        if self.callback_label=='final_':
            with open(f'{self.output_dir}/params_dict.pickle','wb') as file:
                pickle.dump(self.params_dict,file)
            np.savetxt(f'{self.output_dir}/bestfit_spectrum.txt',spectrum,delimiter=' ',header='wavelength(nm) flux')
        
        return self.params_dict,self.model_flux

    def evaluate(self,only_abundances=False,only_params=None,split_corner=True,
                 callback_label='final_',makefigs=True):
        self.callback_label=callback_label
        self.PMN_analyse() # get/save bestfit params and final posterior
        self.params_dict,self.model_flux=self.get_params_and_spectrum() # all params: constant + free + scaling phi_ij + s2_ij
        if makefigs:
            if callback_label=='final_':
                figs.make_all_plots(self,only_abundances=only_abundances,only_params=only_params,split_corner=split_corner)
            else:
                figs.summary_plot(self)

    def run_retrieval(self,N_live_points=400,evidence_tolerance=0.5): 

        self.N_live_points=N_live_points
        self.evidence_tolerance=evidence_tolerance

        print(f'\n ------ {self.chemistry} - Nlive: {self.N_live_points} - ev: {self.evidence_tolerance} ------ \n')

        # run retrieval if hasn't been run yet, else just redo plots
        final_dict=pathlib.Path(f'{self.output_dir}/params_dict.pickle')
        if final_dict.exists()==False:
            print('\nStarting retrieval\n')
            self.PMN_run(N_live_points=self.N_live_points,evidence_tolerance=self.evidence_tolerance)
        else:
            print('\nRetrieval exists\n')
        self.evaluate() # creates plots and saves self.params_dict

        output_file=pathlib.Path('retrieval.out')
        if output_file.exists():
            os.system(f"mv {output_file} {self.output_dir}")

        print('\n ----------------- Done ---------------- \n')