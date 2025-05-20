import icarogw
import matplotlib.pyplot as plt
import numpy as np
import h5py
import copy

from icarogw.posterior_samples import posterior_samples
from icarogw.injections import injections_at_source
from icarogw.priors.mass import mass_prior
from icarogw.priors.redshift import redshift_prior
from icarogw.cosmologies import flatLCDM
from icarogw.analyses.cosmo_pop_rate_marginalized import hierarchical_analysis

H0_array=np.linspace(20,140,1000) # H0 at which you want to study

mass_hp = dic_param = {'alpha':3.78,'beta':0.81,'mmin':4.98,'mmax':112.5,'mu_g':32.27,
                    'sigma_g':3.88,'lambda_peak':0.03,'delta_m':4.8,'b':50}
mp = mass_prior(name='BBH-powerlaw-gaussian',hyper_params_dict=mass_hp)
mp_list = [mp for i in range(len(H0_array))] # Defines a list of population priors for masses, all equal
cosmo_list = [flatLCDM(H0=H0,Omega_m=0.3) for H0 in H0_array] # Defines a list of cosmology, here we are changing H0
zp_pl = [redshift_prior(cosmo,'madau',{'gamma':4.59,'kappa':2.86,'zp':2.47}) for cosmo in cosmo_list] # Defines a uniform in comoving volume prior

injdata = h5py.File('/home/ansonchen/gwcosmo_O4a_cosmo/injections_O4a_SNR11_m_1000_IFAR_minus1.h5','r')
# injdata = h5py.File('/home/ansonchen/cosmic_dipole_gw/code/combined_pdet_SNR_11.0.h5','r')

inj = icarogw.injections.injections_at_detector(m1d=np.array(injdata['m1d']),
                                    m2d=np.array(injdata['m2d']),
                                    dl=np.array(injdata['dl']),
                                    prior_vals=np.array(injdata['pini']),
                                    snr_det=np.array(injdata['snr']),
                                    snr_cut=0,
                                    ifar=np.inf+0*np.array(injdata['m1d']),
                                    ifar_cut=0,
                                    ntotal=np.array(injdata['ntotal']),
                                    Tobs=np.array(injdata['Tobs']))
injdata.close()

injections = copy.deepcopy(inj) # Nobs is set in self.injection in the init
injections.update_cut(snr_cut=11,ifar_cut=0) # We are going to update the injections with the new SNR cut to 12

# Read the posterior samples
# index = np.loadtxt('index_alldet.txt')

posterior_dict = {}
for i in range(0, 206):
#for i in index:
    #if i<201:
    #i = int(i)
    filename = '/home/ansonchen/cosmic_dipole_gw/code/Fisher_sampler_O4_inc_new/Fisher_samples_%d.dat'%i
            
    pos_samples = posterior_samples(filename)
    posterior_dict[i]=pos_samples
    
analysis = hierarchical_analysis(posterior_dict,injections) # Initialize the analysis
single_posterior = analysis.run_analysis_on_lists(mp_list,zp_pl) # Run the analysis on the list of population models

 # Just combining posteriors with this script. You might want to add a smoothing factor when multiplying posteriors
combined = np.zeros(len(H0_array))
for i in list(single_posterior.keys()): #range(len(single_posterior)):
    combined+=single_posterior[i]
    combined-=combined.max()
    # print(i, single_posterior[i])
    single_posterior[i] = np.exp(single_posterior[i])
    single_posterior[i]/= np.trapz(single_posterior[i],H0_array)
    np.savetxt('posterior_H0_gridded_O4_inc_vcprior_icarogw/posterior_H0_gridded_%d.txt'%i, [H0_array,single_posterior[i]])
    plt.plot(H0_array,single_posterior[i], color='k',alpha=0.2)
combined=np.exp(combined)
combined/=np.trapz(combined,H0_array)

plt.axvline(70,linestyle='--')
plt.plot(H0_array,combined/3,color='r',label='Combined')
plt.xlim(min(H0_array),max(H0_array))
plt.ylim(0)
plt.legend()
plt.savefig('/home/ansonchen/cosmic_dipole_gw/code/icarogw_O4_inc_vcprior.png')
