import numpy as np
import corner
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
from pRT_model import pRT_spectrum


def plot_spectrum_inset(retrieval_object,inset=True,fs=10,**kwargs):

    wave=retrieval_object.data_wave
    flux=retrieval_object.data_flux
    err=retrieval_object.data_err
    flux_m=retrieval_object.model_flux

    if 'ax' in kwargs:
        ax=kwargs.get('ax')
    else:
        fig,ax=plt.subplots(2,1,figsize=(9.5,3),dpi=200,gridspec_kw={'height_ratios':[2,0.7]})

    for order in range(7):
        for det in range(3):
            lower=flux[order,det]-err[order,det]*retrieval_object.params_dict['s2_ij'][order,det]
            upper=flux[order,det]+err[order,det]*retrieval_object.params_dict['s2_ij'][order,det]
            ax[0].plot(wave[order,det],flux[order,det],lw=0.8,alpha=1,c='k',label='data')
            ax[0].fill_between(wave[order,det],lower,upper,color='k',alpha=0.15,label=f'1 $\sigma$')
            ax[0].plot(wave[order,det],flux_m[order,det],lw=0.8,alpha=0.8,c=retrieval_object.color1,label='model')
            
            ax[1].plot(wave[order,det],flux[order,det]-flux_m[order,det],lw=0.8,c=retrieval_object.color1,label='residuals')
            if order==0 and det==0:
                lines = [Line2D([0], [0], color='k',linewidth=2,label='Data'),
                        mpatches.Patch(color='k',alpha=0.15,label='1$\sigma$'),
                        Line2D([0], [0], color=retrieval_object.color1, linewidth=2,label='Bestfit')]
                ax[0].legend(handles=lines,fontsize=fs) # to only have it once
        ax[1].plot([np.min(wave[order]),np.max(wave[order])],[0,0],lw=0.8,alpha=1,c='k')
        
    ax[0].set_ylabel('Normalized Flux',fontsize=fs)
    ax[1].set_ylabel('Residuals',fontsize=fs)
    ax[0].set_xlim(np.min(wave)-10,np.max(wave)+10)
    ax[1].set_xlim(np.min(wave)-10,np.max(wave)+10)
    tick_spacing=10
    ax[1].xaxis.set_minor_locator(ticker.MultipleLocator(tick_spacing))
    ax[0].tick_params(labelsize=fs)
    ax[1].tick_params(labelsize=fs)

    if inset==True:
        ord=5 
        axins = ax[0].inset_axes([0,-1.3,1,0.8])
        for det in range(3):
            lower=flux[ord,det]-err[ord,det]*retrieval_object.params_dict['s2_ij'][ord,det]
            upper=flux[ord,det]+err[ord,det]*retrieval_object.params_dict['s2_ij'][ord,det]
            axins.fill_between(wave[ord,det],lower,upper,color='k',alpha=0.15,label=f'1 $\sigma$')
            axins.plot(wave[ord,det],flux[ord,det],lw=0.8,c='k')
            axins.plot(wave[ord,det],flux_m[ord,det],lw=0.8,c=retrieval_object.color1,alpha=0.8)
        x1, x2 = np.min(wave[ord]),np.max(wave[ord])
        axins.set_xlim(x1, x2)
        box,lines=ax[0].indicate_inset_zoom(axins,edgecolor="black",alpha=0.2,lw=0.8,zorder=1e3)
        axins.set_ylabel('Normalized Flux',fontsize=fs)
        axins.tick_params(labelsize=fs)
        ax[1].set_facecolor('none') # to avoid hiding lines
        ax[0].set_xticks([])
        
        axins2 = axins.inset_axes([0,-0.3,1,0.3])
        for det in range(3):
            axins2.plot(wave[ord,det],flux[ord,det]-flux_m[ord,det],lw=0.8,c=retrieval_object.color1)
            axins2.plot(wave[ord,det],np.zeros_like(wave[ord,det]),lw=0.8,alpha=1,c='k')
        axins2.set_xlim(x1, x2)
        axins2.set_xlabel('Wavelength [nm]',fontsize=fs)
        axins2.set_ylabel('Res.',fontsize=fs)
        tick_spacing=1
        axins2.xaxis.set_minor_locator(ticker.MultipleLocator(tick_spacing))
        axins2.tick_params(labelsize=fs)
    else:
        ax[1].set_xlabel('Wavelength [nm]',fontsize=fs) # if no inset

    plt.subplots_adjust(wspace=0, hspace=0)
    if 'ax' not in kwargs:
        name = 'bestfit_spectrum_inset' if retrieval_object.callback_label=='final_' else f'{retrieval_object.callback_label}bestfit_spectrum_inset'
        fig.savefig(f'{retrieval_object.output_dir}/{name}.pdf',
                    bbox_inches='tight')
        plt.close()

def plot_spectrum_split(retrieval_object):

    retrieval=retrieval_object
    residuals=(retrieval.data_flux-retrieval.model_flux)
    fig,ax=plt.subplots(20,1,figsize=(10,13),dpi=200,gridspec_kw={'height_ratios':[2,0.9,0.57]*6+[2,0.9]})
    x=0
    for order in range(7): 
        ax1=ax[x]
        ax2=ax[x+1]
        
        if x!=18: # last ax cannot be spacer, or xlabel also invisible
            ax3=ax[x+2] #for spacing
        for det in range(3):
            ax1.plot(retrieval.data_wave[order,det],retrieval.data_flux[order,det],lw=0.8,alpha=1,c='k',label='data')
            lower=retrieval.data_flux[order,det]-retrieval.data_err[order,det]*retrieval.params_dict['s2_ij'][order,det]
            upper=retrieval.data_flux[order,det]+retrieval.data_err[order,det]*retrieval.params_dict['s2_ij'][order,det]
            ax1.fill_between(retrieval.data_wave[order,det],lower,upper,color='k',alpha=0.15,label=f'1 $\sigma$')
            ax1.plot(retrieval.data_wave[order,det],retrieval.model_flux[order,det],lw=0.8,alpha=0.8,c=retrieval_object.color1,label='model')
            ax1.set_xlim(np.nanmin(retrieval.data_wave[order])-1,np.nanmax(retrieval.data_wave[order])+1)
            
            ax2.plot(retrieval.data_wave[order,det],residuals[order,det],lw=0.8,alpha=1,c=retrieval_object.color1,label='residuals')
            ax2.set_xlim(np.nanmin(retrieval.data_wave[order])-1,np.nanmax(retrieval.data_wave[order])+1)

            # add error for scale
            errmean=np.nanmean(retrieval.data_err[order,det]*retrieval.params_dict['s2_ij'][order,det])
            if np.nansum(retrieval.data_flux[order])!=0: # skip empty orders
                ax2.errorbar(np.min(retrieval.data_wave[order,det])-0.3, 0, yerr=errmean, 
                            ecolor=retrieval_object.color1, elinewidth=1, capsize=2)
            
            if x==0 and det==0:
                lines = [Line2D([0], [0], color='k',linewidth=2,label='Data'),
                        mpatches.Patch(color='k',alpha=0.15,label='1$\sigma$'),
                        Line2D([0], [0], color=retrieval.color1, linewidth=2,label='Bestfit')]
                ax1.legend(handles=lines,fontsize=12,ncol=3,bbox_to_anchor=(0.47,1.4),loc='upper center')
                #leg.get_frame().set_linewidth(0.0)

            ax2.plot([np.min(retrieval.data_wave[order,det]),np.max(retrieval.data_wave[order,det])],[0,0],lw=0.8,c='k')

        min1=np.nanmin(np.array([retrieval.data_flux[order]-retrieval.data_err[order],retrieval.model_flux[order]]))
        max1=np.nanmax(np.array([retrieval.data_flux[order]+retrieval.data_err[order],retrieval.model_flux[order]]))
        ax1.set_ylim(min1,max1)
        if np.nansum(residuals[order])!=0:
            ax2.set_ylim(np.nanmin(residuals[order]),np.nanmax(residuals[order]))
        else:# if empty order full of nans
            ax2.set_ylim(-0.1,0.1)
        ax1.tick_params(labelbottom=False)  # don't put tick labels at bottom
        ax1.tick_params(axis="both")
        ax2.tick_params(axis="both")
        ax1.set_ylabel('Normalized Flux')
        ax2.set_ylabel('Res.')
        ax1.tick_params(labelsize=9)
        ax2.tick_params(labelsize=9)
        tick_spacing=1
        ax2.xaxis.set_minor_locator(ticker.MultipleLocator(tick_spacing))
        if x!=18:
            ax3.set_visible(False) # invisible for spacing
        x+=3
    ax[19].set_xlabel('Wavelength [nm]')
    fig.tight_layout()
    plt.subplots_adjust(wspace=0,hspace=0)
    name = 'bestfit_spectrum' if retrieval_object.callback_label=='final_' else f'{retrieval_object.callback_label}bestfit_spectrum'
    fig.savefig(f'{retrieval_object.output_dir}/{name}.pdf')
    plt.close()

def plot_pt(retrieval_object,fs=12,**kwargs):

    if 'ax' in kwargs:
        ax=kwargs.get('ax')
    else:
        fig,ax=plt.subplots(1,1,figsize=(5,5),dpi=200)

    lines=[]
    # plot PT-profile + errors on retrieved temperatures
    def plot_temperature(retr_obj,ax): 

        #ax.plot(retr_obj.model_object.temperature, retr_obj.model_object.pressure,color=retr_obj.color1,lw=2) 
        medians=[]
        errs=[]
        log_P_knots=retr_obj.model_object.log_P_knots
        for key in ['T4','T3','T2','T1','T0']: # order T4,T3,T2,T1,T0 like log_P_knots
            medians.append(retr_obj.params_dict[key])
            errs.append(retr_obj.params_dict[f'{key}_err'])
        errs=np.array(errs)
        for x in [1,2,3]: # plot 1-3 sigma errors
            lower = CubicSpline(log_P_knots,medians+x*errs[:,0])(np.log10(retr_obj.pressure))
            upper = CubicSpline(log_P_knots,medians+x*errs[:,1])(np.log10(retr_obj.pressure))
            ax.fill_betweenx(retr_obj.pressure,lower,upper,color=retr_obj.color1,alpha=0.15)
        ax.scatter(medians,10**retr_obj.model_object.log_P_knots,color=retr_obj.color1)
        median_temperature=CubicSpline(log_P_knots,medians)(np.log10(retr_obj.pressure))
        ax.plot(median_temperature, retr_obj.model_object.pressure,color=retr_obj.color1,lw=2) 
        xmin=np.min(lower)-100
        xmax=np.max(upper)+100
        lines.append(Line2D([0],[0],marker='o',color=retrieval_object.color1,markerfacecolor=retrieval_object.color1,
                linewidth=2,linestyle='-',label='Retrieval'))

        return xmin,xmax

    xmin,xmax=plot_temperature(retrieval_object,ax)
    model_object=pRT_spectrum(retrieval_object,contribution=True) # get emission contribution
    model_object.make_spectrum()
    summed_contr=np.nanmean(model_object.contr_em_orders,axis=0) # average over all orders
    contribution_plot=summed_contr/np.max(summed_contr)*(xmax-xmin)+xmin
    ax.plot(contribution_plot,retrieval_object.model_object.pressure,linestyle='dashed',
            lw=1.5,alpha=0.8,color=retrieval_object.color1)
    lines.append(Line2D([0], [0], color=retrieval_object.color1, alpha=0.8,
                        linewidth=1.5, linestyle='--',label='Emission contr.'))
    
    ax.set(xlabel='Temperature [K]', ylabel='Pressure [bar]',yscale='log',
        ylim=(np.nanmax(retrieval_object.model_object.pressure),
        np.nanmin(retrieval_object.model_object.pressure)),xlim=(xmin,xmax))
            
    ax.legend(handles=lines,fontsize=fs)
    ax.tick_params(labelsize=fs)
    ax.set_xlabel('Temperature [K]', fontsize=fs)
    ax.set_ylabel('Pressure [bar]', fontsize=fs)

    if 'ax' not in kwargs: # save as separate plot
        fig.tight_layout()
        name = 'PT_profile' if retrieval_object.callback_label=='final_' else f'{retrieval_object.callback_label}PT_profile'
        fig.savefig(f'{retrieval_object.output_dir}/{name}.pdf')
        plt.close()

def cornerplot(retrieval_object,getfig=False,figsize=(20,20),fs=12,only_params=None):
    
    plot_posterior=retrieval_object.posterior # posterior that we plot here, might get clipped
    medians,_,_=retrieval_object.get_quantiles(retrieval_object.posterior)
    labels=list(retrieval_object.parameters.param_mathtext.values())
    indices=np.linspace(0,len(retrieval_object.parameters.params)-1,len(retrieval_object.parameters.params),dtype=int)

    if only_params is not None: # keys of specified parameters to plot
        indices=[]
        for key in only_params:
            idx=list(retrieval_object.parameters.params).index(key)
            indices.append(idx)
        plot_posterior=np.array([retrieval_object.posterior[:,i] for i in indices]).T
        labels=np.array([labels[i] for i in indices])
        medians=np.array([medians[i] for i in indices])

    fig = plt.figure(figsize=figsize) # fix size to avoid memory issues
    fig = corner.corner(plot_posterior, 
                        labels=labels, 
                        title_kwargs={'fontsize':fs},
                        label_kwargs={'fontsize':fs*0.8},
                        color=retrieval_object.color1,
                        linewidths=0.5,
                        fill_contours=True,
                        quantiles=[0.16,0.5,0.84],
                        title_quantiles=[0.16,0.5,0.84],
                        show_titles=True,
                        hist_kwargs={'density': False,
                                'fill': True,
                                'alpha': 0.5,
                                'edgecolor': 'k',
                                'linewidth': 1.0},
                        fig=fig,
                        quiet=True)
    
    # split title to avoid overlap with plots
    titles = [axi.title.get_text() for axi in fig.axes]
    for i, title in enumerate(titles):
        if len(title) > 30: # change 30 to 1 if you want all titles to be split
            title_split = title.split('=')
            titles[i] = title_split[0] + '\n ' + title_split[1]
        fig.axes[i].title.set_text(titles[i])

    plt.subplots_adjust(wspace=0,hspace=0)

    if getfig==False:
        name= f'cornerplot' if retrieval_object.callback_label=='final_' else f'{retrieval_object.callback_label}cornerplot'
        fig.savefig(f'{retrieval_object.output_dir}/{name}.pdf',bbox_inches="tight",dpi=200)
        plt.close()
    else:
        ax = np.array(fig.axes)
        return fig, ax

def make_all_plots(retrieval_object,only_params=None):
    plot_spectrum_split(retrieval_object)
    plot_spectrum_inset(retrieval_object)
    plot_pt(retrieval_object)
    summary_plot(retrieval_object)
    # if only_params==None -> make cornerplot with all parameters, but will be huge
    cornerplot(retrieval_object,only_params=only_params)
    
def summary_plot(retrieval_object):

    fs=13
    if retrieval_object.chemistry in ['equchem','quequchem']:
        only_params=['rv','vsini','log_g','C/O','Fe/H','log_C12_13_ratio','log_O16_18_ratio','log_O16_17_ratio']
    if retrieval_object.chemistry=='freechem':
        only_params=['rv','vsini','log_g']
        # plot 6 most abundant species
        abunds=[] # abundances
        species=retrieval_object.chem_species
        for spec in species:
            abunds.append(retrieval_object.params_dict[spec])
        abunds, species = zip(*sorted(zip(abunds, species)))
        only_species=species[-6:][::-1] # get largest 6
        only_params+=only_species

    fig, ax = cornerplot(retrieval_object,getfig=True,only_params=only_params,figsize=(17,17),fs=fs)
    l, b, w, h = [0.37,0.84,0.6,0.15] # left, bottom, width, height
    ax_spec = fig.add_axes([l,b,w,h])
    ax_res = fig.add_axes([l,b-0.03,w,h-0.12])
    plot_spectrum_inset(retrieval_object,ax=(ax_spec,ax_res),inset=False,fs=fs)

    l, b, w, h = [0.68,0.47,0.29,0.29] # left, bottom, width, height
    ax_PT = fig.add_axes([l,b,w,h])
    plot_pt(retrieval_object,ax=ax_PT,fs=fs)
    name = 'summary' if retrieval_object.callback_label=='final_' else f'{retrieval_object.callback_label}summary'
    fig.savefig(f'{retrieval_object.output_dir}/{name}.pdf',
                bbox_inches="tight",dpi=200)
    plt.close()

    