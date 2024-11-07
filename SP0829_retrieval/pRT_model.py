import numpy as np
import os
from scipy.interpolate import CubicSpline
from petitRADTRANS.poor_mans_nonequ_chem import interpol_abundances
from PyAstronomy.pyasl import fastRotBroad, helcorr
from astropy import constants as const
from astropy import units as u
from astropy.coordinates import SkyCoord
import pandas as pd
from scipy.ndimage import gaussian_filter
from spectrum import Spectrum, convolve_to_resolution

class pRT_spectrum:

    def __init__(self,
                 retrieval_object,
                 spectral_resolution=100_000,  
                 contribution=False): # only for plotting atmosphere.contr_em
        
        self.params=retrieval_object.parameters.params
        self.data_wave=retrieval_object.data_wave
        self.target=retrieval_object.target
        self.coords = SkyCoord(ra=self.target.ra, dec=self.target.dec, frame='icrs')
        self.species=retrieval_object.species
        self.spectral_resolution=spectral_resolution
        self.chemistry=retrieval_object.chemistry
        self.atmosphere_objects=retrieval_object.atmosphere_objects
        self.lbl_opacity_sampling=retrieval_object.lbl_opacity_sampling

        self.n_atm_layers=retrieval_object.n_atm_layers
        self.pressure = retrieval_object.pressure
        self.temperature = self.make_pt() #P-T profile

        self.gravity = 10**self.params['log_g'] 
        self.contribution=contribution

        if self.chemistry=='freechem': # use free chemistry with defined VMRs
            self.mass_fractions, self.CO, self.FeH = self.free_chemistry(self.species,self.params)
            self.MMW = self.mass_fractions['MMW']

        if self.chemistry in ['equchem','quequchem']: # use equilibium chemistry
            self.abunds = self.abundances(self.pressure,self.temperature,self.params['Fe/H'],self.params['C/O'])
            self.mass_fractions = self.get_abundance_dict(self.species,self.abunds)
            # update mass_fractions with isotopologue ratios
            self.mass_fractions = self.get_isotope_mass_fractions(self.species,self.mass_fractions,self.params) 
            self.MMW = self.abunds['MMW']

        self.spectrum_orders=[]
        self.n_orders=retrieval_object.n_orders

    def abundances(self,press, temp, feh, C_O):
        COs = np.ones_like(press)*C_O
        fehs = np.ones_like(press)*feh
        mass_fractions = interpol_abundances(COs,fehs,temp,press)
        if self.chemistry=='quequchem':
            for species in ['CO','H2O','CH4']:
                Pqu=10**self.params['log_Pqu_CO_CH4'] # is in log
                idx=find_nearest(self.pressure,Pqu)
                quenched_fraction=mass_fractions[species][idx]
                mass_fractions[species][:idx]=quenched_fraction
            for species in ['NH3','HCN']:    
                Pqu=10**self.params[f'log_Pqu_{species}'] # is in log
                idx=find_nearest(self.pressure,Pqu)
                quenched_fraction=mass_fractions[species][idx]
                mass_fractions[species][:idx]=quenched_fraction
        return mass_fractions

    def get_abundance_dict(self,species,abunds): # does not inlcude isotopes
        mass_fractions = {}
        for specie in species:
            if specie in ['H2O_main_iso','H2O_pokazatel_main_iso']:
                mass_fractions[specie] = abunds['H2O']
            elif specie=='CO_main_iso':
                mass_fractions[specie] = abunds['CO']
            elif specie in ['CH4_main_iso','CH4_hargreaves_main_iso']:
                mass_fractions[specie] = abunds['CH4']
            elif specie=='HCN_main_iso':
                mass_fractions[specie] = abunds['HCN']
            elif specie=='NH3_coles_main_iso':
                mass_fractions[specie] = abunds['NH3']
            elif specie=='HF_main_iso':
                mass_fractions[specie] = 1e-12*np.ones(self.n_atm_layers) #abunds['HF'] not in pRT chem equ table
            elif specie=='H2S_ExoMol_main_iso':
                mass_fractions[specie] = abunds['H2S']
            elif specie=='OH_main_iso':
                mass_fractions[specie] = 1e-12*np.ones(self.n_atm_layers) #abunds['OH'] not in pRT chem equ table
            elif specie=='CO2_main_iso':
                mass_fractions[specie] = abunds['CO2']
        mass_fractions['H2'] = abunds['H2']
        mass_fractions['He'] = abunds['He']
        return mass_fractions
    
    def read_species_info(self,species,info_key):
        species_info = pd.read_csv(os.path.join('species_info.csv'), index_col=0)
        if info_key == 'pRT_name':
            return species_info.loc[species,info_key]
        if info_key == 'pyfc_name':
            return species_info.loc[species,'Hill_notation']
        if info_key == 'mass':
            return species_info.loc[species,info_key]
        if info_key == 'COH':
            return list(species_info.loc[species,['C','O','H']])
        if info_key in ['C','O','H']:
            return species_info.loc[species,info_key]
        if info_key == 'c' or info_key == 'color':
            return species_info.loc[species,'color']
        if info_key == 'label':
            return species_info.loc[species,'mathtext_name']
    
    def get_isotope_mass_fractions(self,species,mass_fractions,params):
        #https://github.com/samderegt/retrieval_base/blob/main/retrieval_base/chemistry.py
        mass_ratio_13CO_12CO = self.read_species_info('13CO','mass')/self.read_species_info('12CO','mass')
        mass_ratio_C18O_C16O = self.read_species_info('C18O','mass')/self.read_species_info('12CO','mass')
        mass_ratio_C17O_C16O = self.read_species_info('C17O','mass')/self.read_species_info('12CO','mass')
        mass_ratio_H218O_H2O = self.read_species_info('H2(18)O','mass')/self.read_species_info('H2O','mass')
        self.C13_12_ratio = 10**(-params.get('log_C12_13_ratio',-12))
        self.O18_16_ratio = 10**(-params.get('log_O16_18_ratio',-12))
        self.O17_16_ratio = 10**(-params.get('log_O16_17_ratio',-12))

        for species_i in species:
            if (species_i=='CO_main_iso'): # 12CO mass fraction
                mass_fractions[species_i]=(1-self.C13_12_ratio*mass_ratio_13CO_12CO
                                            -self.O18_16_ratio*mass_ratio_C18O_C16O
                                            -self.O17_16_ratio*mass_ratio_C17O_C16O)*mass_fractions['CO_main_iso']
                continue
            if (species_i=='CO_36'): # 13CO mass fraction
                mass_fractions[species_i]=self.C13_12_ratio*mass_ratio_13CO_12CO*mass_fractions['CO_main_iso']
                continue
            if (species_i=='CO_28'): # C18O mass fraction
                mass_fractions[species_i]=self.O18_16_ratio*mass_ratio_C18O_C16O*mass_fractions['CO_main_iso']
                continue
            if (species_i=='CO_27'): # C17O mass fraction
                mass_fractions[species_i]=self.O17_16_ratio*mass_ratio_C17O_C16O*mass_fractions['CO_main_iso']
                continue
            if (species_i in ['H2O_main_iso','H2O_pokazatel_main_iso']): # H2O mass fraction
                H2O_linelist=species_i
                mass_fractions[species_i]=(1-self.O18_16_ratio*mass_ratio_H218O_H2O)*mass_fractions[species_i]
                continue
            if (species_i=='H2O_181_HotWat78'): # H2_18O mass fraction
                mass_fractions[species_i]=self.O18_16_ratio*mass_ratio_H218O_H2O*mass_fractions[H2O_linelist]
                continue
            
        return mass_fractions
    
    def free_chemistry(self,line_species,params):
        species_info = pd.read_csv(os.path.join('species_info.csv'), index_col=0)
        VMR_He = 0.15
        VMR_wo_H2 = 0 + VMR_He  # Total VMR without H2, starting with He
        mass_fractions = {} # Create a dictionary for all used species
        C, O, H = 0, 0, 0

        for species_i in species_info.index:
            line_species_i = self.read_species_info(species_i,'pRT_name')
            mass_i = self.read_species_info(species_i, 'mass')
            COH_i  = self.read_species_info(species_i, 'COH')

            if species_i in ['H2', 'He']:
                continue
            if line_species_i in line_species:
                VMR_i = 10**(params[f'log_{species_i}'])*np.ones(self.n_atm_layers) #  use constant, vertical profile

                # Convert VMR to mass fraction using molecular mass number
                mass_fractions[line_species_i] = mass_i * VMR_i
                VMR_wo_H2 += VMR_i

                # Record C, O, and H bearing species for C/O and metallicity
                C += COH_i[0] * VMR_i
                O += COH_i[1] * VMR_i
                H += COH_i[2] * VMR_i

        # Add the H2 and He abundances
        mass_fractions['He'] = self.read_species_info('He', 'mass')*VMR_He
        mass_fractions['H2'] = self.read_species_info('H2', 'mass')*(1-VMR_wo_H2)
        H += self.read_species_info('H2','H')*(1-VMR_wo_H2) # Add to the H-bearing species
        
        if VMR_wo_H2.any() > 1:
            print('VMR_wo_H2 > 1. Other species are too abundant!')

        MMW = 0 # Compute the mean molecular weight from all species
        for mass_i in mass_fractions.values():
            MMW += mass_i
        MMW *= np.ones(self.n_atm_layers)
        
        for line_species_i in mass_fractions.keys():
            mass_fractions[line_species_i] /= MMW # Turn the molecular masses into mass fractions
        mass_fractions['MMW'] = MMW # pRT requires MMW in mass fractions dictionary
        CO = C/O
        log_CH_solar = 8.46 - 12 # Asplund et al. (2021)
        FeH = np.log10(C/H)-log_CH_solar
        CO = np.nanmean(CO)
        FeH = np.nanmean(FeH)

        return mass_fractions, CO, FeH
    
    def make_spectrum(self):

        spectrum_orders=[]
        self.wlshift_orders=[]
        waves_orders=[]
        self.contr_em_orders=[]
        for order in range(self.n_orders):
            atmosphere=self.atmosphere_objects[order]

            atmosphere.calc_flux(self.temperature,
                            self.mass_fractions,
                            self.gravity,
                            self.MMW,
                            contribution=self.contribution)

            wl = const.c.to(u.km/u.s).value/atmosphere.freq/1e-9 # mircons
            flux=atmosphere.flux

            # RV+bary shifting and rotational broadening
            v_bary, _ = helcorr(obs_long=-70.40, obs_lat=-24.62, obs_alt=2635, # of Cerro Paranal
                            ra2000=self.coords.ra.value,dec2000=self.coords.dec.value,jd=self.target.JD) # https://ssd.jpl.nasa.gov/tools/jdc/#/cd
            wl_shifted= wl*(1.0+(self.params['rv']-v_bary)/const.c.to('km/s').value)
            self.wlshift_orders.append(wl_shifted)
            spec = Spectrum(flux, wl_shifted)
            waves_even = np.linspace(np.min(wl), np.max(wl), wl.size) # wavelength array has to be regularly spaced
            spec = fastRotBroad(waves_even, spec.at(waves_even), self.params['epsilon_limb'], self.params['vsini']) # limb-darkening coefficient (0-1)
            spec = Spectrum(spec, waves_even)
            spec = convolve_to_resolution(spec,self.spectral_resolution)

            #https://github.com/samderegt/retrieval_base/blob/main/retrieval_base/spectrum.py#L289
            self.resolution = int(1e6/self.lbl_opacity_sampling)
            flux=self.instr_broadening(spec.wavelengths*1e3,spec,out_res=self.resolution,in_res=500000)

            # Interpolate/rebin onto the data's wavelength grid
            ref_wave = self.data_wave[order].flatten() # [nm]
            flux = np.interp(ref_wave, spec.wavelengths*1e3, flux) # pRT wavelengths from microns to nm

            # reshape to (detectors,pixels) so that we can store as shape (orders,detectors,pixels)
            flux=flux.reshape(self.data_wave.shape[1],self.data_wave.shape[2])

            spectrum_orders.append(flux)
            waves_orders.append(waves_even*1e3) # from um to nm

            if self.contribution==True:
                contr_em = atmosphere.contr_em # emission contribution
                summed_contr = np.nansum(contr_em,axis=1) # sum over all wavelengths
                self.contr_em_orders.append(summed_contr)
    
        spectrum_orders=np.array(spectrum_orders)
        spectrum_orders/=np.nanmedian(spectrum_orders) # normalize in same way as data spectrum
        return spectrum_orders
            
    # create pressure-temperature profile from 5 temperature knots
    def make_pt(self): 

        self.T_knots = np.array([self.params['T4'],self.params['T3'],self.params['T2'],self.params['T1'],self.params['T0']])
        self.log_P_knots= np.linspace(np.log10(np.min(self.pressure)),np.log10(np.max(self.pressure)),num=len(self.T_knots))
        sort = np.argsort(self.log_P_knots)
        self.temperature = CubicSpline(self.log_P_knots[sort],self.T_knots[sort])(np.log10(self.pressure))
        
        return self.temperature
    
    def instr_broadening(self, wave, flux, out_res=1e6, in_res=1e6):

        # Delta lambda of resolution element is FWHM of the LSF's standard deviation
        sigma_LSF = np.sqrt(1/out_res**2-1/in_res**2)/(2*np.sqrt(2*np.log(2)))
        spacing = np.mean(2*np.diff(wave) / (wave[1:] + wave[:-1]))

        # Calculate the sigma to be used in the gauss filter in pixels
        sigma_LSF_gauss_filter = sigma_LSF / spacing
        
        # Apply gaussian filter to broaden with the spectral resolution
        flux_LSF = gaussian_filter(flux, sigma=sigma_LSF_gauss_filter,mode='nearest')

        return flux_LSF

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx