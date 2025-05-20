import numpy as np
import GWDALI as dali
import pickle
import pycbc.detector as pycbc_detector

rad = np.pi/180 ; deg = 1./rad

FreeParams = ['m1','m2','DL','iota','RA','Dec'] #,'psi','t_coal','phi_coal']

# Cosmic Explorer:
# det4 = {"name":"CE","lon":-119.4,"lat":46.5,"rot":126.0,"shape":90} #H1
# det5 = {"name":"CE","lon":-90.8,"lat":30.6,"rot":197.7,"shape":90} #L1
# Einstein Telescope:
# det1 = {"name":"ET","lon":10,"lat":43,"rot":0,"shape":60}
# det2 = {"name":"ET","lon":10,"lat":43,"rot":120,"shape":60}
# det3 = {"name":"ET","lon":10,"lat":43,"rot":-120,"shape":60}

det_h1 = {"name":"aLIGO","lon":-119.4,"lat":46.5,"rot":126.0,"shape":90,"abv":'H1'} #H1
det_l1 = {"name":"aLIGO","lon":-90.8,"lat":30.6,"rot":197.7,"shape":90,"abv":'L1'} #L1
det_v1 = {"name":"aVirgo","lon":10.5,"lat":43.6,"rot":70.6,"shape":90,"abv":'V1'} #V1
det_k1 = {"name":"KAGRA","lon":137.3,"lat":36.4,"rot":29.6,"shape":90,"abv":'K1'} #K1

det_list = {'H1':det_h1, 'L1':det_l1, 'V1':det_v1, 'K1':det_k1}

#------------------------------------------------------
# Setting Injections (Single detection)
#------------------------------------------------------
# z = 0.1 # Redshift

# fr = open("/home/ansonchen/cosmic_dipole_gw/code/O4_2det_inj.p", "rb")
fr = open("/home/ansonchen/cosmic_dipole_gw/O4_test/GW_injections_O4.p", "rb")
inj = pickle.load(fr)
fr.close()

# params = {}
# params['m1']  = 15 #*(1+z) # mass of the first object [solar mass]
# params['m2']  = 13 #*(1+z) # mass of the second object [solar mass]
# # params['z']   = z
# params['RA']       = 20 #np.random.uniform(-180,180)
# params['Dec']      = 30 #(np.pi/2-np.arccos(np.random.uniform(-1,1)))*deg
# params['DL']       = 2 # Gpc
# params['iota']     = 0.7 #np.random.uniform(0,np.pi)      # Inclination angle (rad)
# params['psi']      = 0.5 #np.random.uniform(0,np.pi) # Polarization angle (rad)
# params['t_coal']   = 0  # Coalescence time
# params['phi_coal'] = 0  # Coalescence phase
# # Spins:
# params['sx1'] = 0
# params['sy1'] = 0
# params['sz1'] = 0
# params['sx2'] = 0
# params['sy2'] = 0
# params['sz2'] = 0

#----------------------------------------------------------------------
# "approximant" options:
#       [Leading_Order, TaylorF2_py, ...] or any lal approximant
#----------------------------------------------------------------------
# "dali_method" options:
#       [Fisher, Fisher_Sampling, Doublet, Triplet, Standard]
#----------------------------------------------------------------------

#phi = (ra-detectors[0].gmst_estimate(gps_time))/np.pi*180

for j in range(0, len(inj['injections_parameters']['m1d'])):

    # phi = (inj['injections_parameters']['ras'][j]-pycbc_detector.gmst_accurate(inj['injections_parameters']['geocent_time'][j]))

    injection_parameters = dict(m1=inj['injections_parameters']['m1d'][j], m2=inj['injections_parameters']['m2d'][j],
                            RA=inj['injections_parameters']['ras'][j], Dec=inj['injections_parameters']['decs'][j], psi=inj['injections_parameters']['psis'][j], 
                            DL=inj['injections_parameters']['dls'][j], iota=inj['injections_parameters']['incs'][j], t_gps=inj['injections_parameters']['geocent_time'][j], t_coal=0, phi_coal=0,
                            sx1=0,sy1=0,sz1=0,sx2=0,sy2=0,sz2=0)

    dets = []
    for det in inj['injections_parameters']['dets'][j]:
        dets.append(det_list[det])

    res = dali.GWDALI(Detection_Dict = injection_parameters,
        FreeParams     = FreeParams,
        detectors      = dets, # Einstein Telescope + 2 Cosmic Explorer
        fmin  = 20., 
        fmax  = 2048, 
        approximant    = 'IMRPhenomXPHM',
        dali_method    = 'Fisher_Sampling',
        sampler_method = 'nestle', # Same as Bilby sampling method
        save_fisher    = False,
        save_cov       = False,
        plot_corner    = False,
        save_samples   = False,
        hide_info      = True,
        index          = 1,
        rcond          = 1.e-4,
        npoints=500) # points for "nested sampling" or steps/walkers for "MCMC"

    Samples = res['Samples']
    # Fisher  = res['Fisher']
    CovFish = res['CovFisher']
    # Cov     = res['CovFisher']
    # Rec = res['Recovery']
    # Err     = res['Error']
    SNR     = res['SNR']

    # print(SNR)
    snr_tot_sq = 0
    for snr in SNR:
        snr_tot_sq += snr**2
    print(j, np.sqrt(snr_tot_sq))
    # print(Err)

    np.savetxt('Fisher_sampler_O4_inc_new/Fisher_samples_'+str(j)+'.dat', Samples, header='mass_1\tmass_2\tluminosity_distance\tiota\tra\tdec')
    np.savetxt('covariance_O4_inc/cov_'+str(j)+'.txt', CovFish)
