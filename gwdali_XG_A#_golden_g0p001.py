import numpy as np
import GWDALI as dali
import pickle
import pycbc.detector as pycbc_detector
from astropy.coordinates import SkyCoord
from astropy import units as u

rad = np.pi/180 ; deg = 1./rad

def dipole_cal(g, ra, dec, l=264, b=48):
    dipole_gal = SkyCoord(l*u.deg, b*u.deg, frame='galactic')
    dipole_ICRS = dipole_gal.transform_to('icrs')

    phi_dipole = dipole_ICRS.ra.value*np.pi/180
    theta_dipole = np.pi/2-dipole_ICRS.dec.value*np.pi/180

    nx_dipole = np.cos(phi_dipole)*np.sin(theta_dipole)
    ny_dipole = np.sin(phi_dipole)*np.sin(theta_dipole)
    nz_dipole = np.cos(theta_dipole)

    phi = ra
    theta = np.pi/2-np.array(dec)

    nx = np.cos(phi)*np.sin(theta)
    ny = np.sin(phi)*np.sin(theta)
    nz = np.cos(theta)

    dipole = g*(nx_dipole*nx + ny_dipole*ny + nz_dipole*nz)

    return dipole

FreeParams = ['m1','m2','DL','iota','RA','Dec']#,'psi','t_coal','phi_coal']

# Cosmic Explorer:
# det_h1 = {"name":"CE","lon":-119.4,"lat":46.5,"rot":126.0,"shape":90,"abv":'H1'} #H1
det_l1 = {"name":"CE","lon":-90.8,"lat":30.6,"rot":197.7,"shape":90,"abv":'L1'} #L1
det_v1 = {"name":"ET","lon":10.5,"lat":43.6,"rot":70.6,"shape":90,"abv":'V1'} #V1
# Einstein Telescope:
# det1 = {"name":"ET","lon":10,"lat":43,"rot":0,"shape":60}
# det2 = {"name":"ET","lon":10,"lat":43,"rot":120,"shape":60}
# det3 = {"name":"ET","lon":10,"lat":43,"rot":-120,"shape":60}

det_h1 = {"name":"a_sharp","lon":-119.4,"lat":46.5,"rot":126.0,"shape":90,"abv":'H1'} #H1
# det_l1 = {"name":"aLIGO","lon":-90.8,"lat":30.6,"rot":197.7,"shape":90,"abv":'L1'} #L1
# det_v1 = {"name":"aVirgo","lon":10.5,"lat":43.6,"rot":70.6,"shape":90,"abv":'V1'} #V1
# det_k1 = {"name":"KAGRA","lon":137.3,"lat":36.4,"rot":29.6,"shape":90,"abv":'K1'} #K1

det_list = {'H1':det_h1, 'L1':det_l1, 'V1':det_v1}

#------------------------------------------------------
# Setting Injections (Single detection)
#------------------------------------------------------
# z = 0.1 # Redshift

# BBH
fr = open("/home/ansonchen/cosmic_dipole_gw/XG_A#_golden/seed_1/BBH/GW_injections_XG_A#.p", "rb")
inj = pickle.load(fr)
fr.close()

g = 0.001

for j in range(0,len(inj['injections_parameters']['m1d'])):

    j=int(j)

    dipole = dipole_cal(g, inj['injections_parameters']['ras'][j], inj['injections_parameters']['decs'][j])

    injection_parameters = dict(m1=inj['injections_parameters']['m1d'][j]*(1+dipole), m2=inj['injections_parameters']['m2d'][j]*(1+dipole),
                            RA=inj['injections_parameters']['ras'][j], Dec=inj['injections_parameters']['decs'][j], psi=inj['injections_parameters']['psis'][j], 
                            DL=inj['injections_parameters']['dls'][j]*(1+dipole), iota=inj['injections_parameters']['incs'][j], t_gps=inj['injections_parameters']['geocent_time'][j], t_coal=0, phi_coal=0,
                            sx1=0,sy1=0,sz1=0,sx2=0,sy2=0,sz2=0)

    dets = []
    for det in inj['injections_parameters']['dets'][j]:
        dets.append(det_list[det])

    res = dali.GWDALI(Detection_Dict = injection_parameters,
        FreeParams     = FreeParams,
        detectors      = dets, # Einstein Telescope + 2 Cosmic Explorer
        fmin  = 20., 
        fmax  = 2048., 
        approximant    = 'IMRPhenomXPHM',
        dali_method    = 'Fisher',
        sampler_method = 'nestle', # Same as Bilby sampling method
        save_fisher    = False,
        save_cov       = False,
        plot_corner    = False,
        save_samples   = False,
        hide_info      = True,
        index          = 1,
        rcond          = 1.e-4,
        npoints=300) # points for "nested sampling" or steps/walkers for "MCMC"

    # Samples = res['Samples']
    # Fisher  = res['Fisher']
    CovFish = res['CovFisher']
    # Cov     = res['Covariance']
    # Rec = res['Recovery']
    # Err     = res['Error']
    SNR     = res['SNR']

    # print(SNR)
    snr_tot_sq = 0
    for snr in SNR:
        snr_tot_sq += snr**2
    print(j, np.sqrt(snr_tot_sq))
    # print(Err)

    # np.savetxt('Fisher_sampler_XG_golden_seed_1/BBH_Fisher_samples_'+str(j)+'.dat', Samples, header='mass_1\tmass_2\tluminosity_distance\tiota\tra\tdec')
    np.savetxt('covariance_XG_A#_golden_seed_1_g0p001/BBH_cov_'+str(j)+'.txt', CovFish)

# NSBH
fr = open("/home/ansonchen/cosmic_dipole_gw/XG_A#_golden/seed_1/NSBH/GW_injections_XG_A#.p", "rb")
inj = pickle.load(fr)
fr.close()

for j in range(0,len(inj['injections_parameters']['m1d'])):

    j=int(j)

    dipole = dipole_cal(g, inj['injections_parameters']['ras'][j], inj['injections_parameters']['decs'][j])

    injection_parameters = dict(m1=inj['injections_parameters']['m1d'][j]*(1+dipole), m2=inj['injections_parameters']['m2d'][j]*(1+dipole),
                            RA=inj['injections_parameters']['ras'][j], Dec=inj['injections_parameters']['decs'][j], psi=inj['injections_parameters']['psis'][j], 
                            DL=inj['injections_parameters']['dls'][j]*(1+dipole), iota=inj['injections_parameters']['incs'][j], t_gps=inj['injections_parameters']['geocent_time'][j], t_coal=0, phi_coal=0,
                            sx1=0,sy1=0,sz1=0,sx2=0,sy2=0,sz2=0)

    dets = []
    for det in inj['injections_parameters']['dets'][j]:
        dets.append(det_list[det])

    res = dali.GWDALI(Detection_Dict = injection_parameters,
        FreeParams     = FreeParams,
        detectors      = dets, # Einstein Telescope + 2 Cosmic Explorer
        fmin  = 20., 
        fmax  = 2048., 
        approximant    = 'IMRPhenomXPHM',
        dali_method    = 'Fisher',
        sampler_method = 'nestle', # Same as Bilby sampling method
        save_fisher    = False,
        save_cov       = False,
        plot_corner    = False,
        save_samples   = False,
        hide_info      = True,
        index          = 1,
        rcond          = 1.e-4,
        npoints=300) # points for "nested sampling" or steps/walkers for "MCMC"

    # Samples = res['Samples']
    # Fisher  = res['Fisher']
    CovFish = res['CovFisher']
    # Cov     = res['Covariance']
    # Rec = res['Recovery']
    # Err     = res['Error']
    SNR     = res['SNR']

    # print(SNR)
    snr_tot_sq = 0
    for snr in SNR:
        snr_tot_sq += snr**2
    print(j, np.sqrt(snr_tot_sq))
    # print(Err)

    # np.savetxt('Fisher_sampler_XG_golden_seed_1/NSBH_Fisher_samples_'+str(j)+'.dat', Samples, header='mass_1\tmass_2\tluminosity_distance\tiota\tra\tdec')
    np.savetxt('covariance_XG_A#_golden_seed_1_g0p001/NSBH_cov_'+str(j)+'.txt', CovFish)

# BNS
fr = open("/home/ansonchen/cosmic_dipole_gw/XG_A#_golden/seed_1/BNS/GW_injections_XG_A#.p", "rb")
inj = pickle.load(fr)
fr.close()

for j in range(0,len(inj['injections_parameters']['m1d'])):

    j=int(j)

    dipole = dipole_cal(g, inj['injections_parameters']['ras'][j], inj['injections_parameters']['decs'][j])

    injection_parameters = dict(m1=inj['injections_parameters']['m1d'][j]*(1+dipole), m2=inj['injections_parameters']['m2d'][j]*(1+dipole),
                            RA=inj['injections_parameters']['ras'][j], Dec=inj['injections_parameters']['decs'][j], psi=inj['injections_parameters']['psis'][j], 
                            DL=inj['injections_parameters']['dls'][j]*(1+dipole), iota=inj['injections_parameters']['incs'][j], t_gps=inj['injections_parameters']['geocent_time'][j], t_coal=0, phi_coal=0,
                            sx1=0,sy1=0,sz1=0,sx2=0,sy2=0,sz2=0)

    dets = []
    for det in inj['injections_parameters']['dets'][j]:
        dets.append(det_list[det])

    res = dali.GWDALI(Detection_Dict = injection_parameters,
        FreeParams     = FreeParams,
        detectors      = dets, # Einstein Telescope + 2 Cosmic Explorer
        fmin  = 20., 
        fmax  = 2048., 
        approximant    = 'IMRPhenomXPHM',
        dali_method    = 'Fisher',
        sampler_method = 'nestle', # Same as Bilby sampling method
        save_fisher    = False,
        save_cov       = False,
        plot_corner    = False,
        save_samples   = False,
        hide_info      = True,
        index          = 1,
        rcond          = 1.e-4,
        npoints=300) # points for "nested sampling" or steps/walkers for "MCMC"

    # Samples = res['Samples']
    # Fisher  = res['Fisher']
    CovFish = res['CovFisher']
    # Cov     = res['Covariance']
    # Rec = res['Recovery']
    # Err     = res['Error']
    SNR     = res['SNR']

    # print(SNR)
    snr_tot_sq = 0
    for snr in SNR:
        snr_tot_sq += snr**2
    print(j, np.sqrt(snr_tot_sq))
    # print(Err)

    # np.savetxt('Fisher_sampler_XG_golden_seed_1/BNS_Fisher_samples_'+str(j)+'.dat', Samples, header='mass_1\tmass_2\tluminosity_distance\tiota\tra\tdec')
    np.savetxt('covariance_XG_A#_golden_seed_1_g0p001/BNS_cov_'+str(j)+'.txt', CovFish)
