import numpy as np
from scipy.integrate import simps
import camb
import pandas as pd
from camb import model,initialpower
from time import time


start = time()


#=======================================================================================================================
# CUSTOM FUNCTIONS
#=======================================================================================================================


def Dz(z,gamma_0,gamma_a):
    gamma = gamma_0 + gamma_a*z/(1.+z)
    z_integral = np.linspace(0.,z,500)
    integrand_dz = (results.get_Omega('cdm',z=z_integral)+results.get_Omega('baryon',z=z_integral)+results.get_Omega('neutrino',z=z_integral))**gamma/(1.+z_integral)
    return np.exp(-simps(integrand_dz,z_integral))


def lens_ker(chis,zs,H_z):
    
    W_k = (3.*Om0*(H0**2)*chis*(1.+zs)*(chistar-chis))/(2.*chistar*c*H_z)	
    return W_k


def gal_ker(dist,zs,alpha,bias,chis,H_z):
    
    W_g = np.zeros(shape=(1,len(zs)))
    	
    dist = dist/simps(dist,zs)
    W_g[0,:] = bias*dist
    
    '''
    for i in range(len(zs)):
	    
	    zs_intgrl = zs[i:]
	    ch_intgrl = chis[i:]
	    H_z_intgrl = H_z[i:]
	    dist_intgrl = dist[i:]
    
	    integrand = (1.-ch_intgrl[0]/ch_intgrl)*(alpha-1.)*dist_intgrl
	    val_intgrl = simps(y=integrand,x=zs_intgrl)

	    W_g[1,i] = (3.*Om0*(H0**2)*chis[i]*(1.+zs[i]))*val_intgrl/(2.*c*H_z[i])
	 '''
    
    return W_g[0,:] #(W_g[0,:]+W_g[1,:])
    
    
def cib_bias_z(zs):
    b0, b1, b2 = 0.94, 0.66, 0.04
    return b0 + b1*zs + b2*zs**2
    
    
def cib_kernel(dist,zs):
    
    W_cib = np.zeros(shape=(len(zs)))
    	
    dist = dist/simps(dist,zs)
    W_cib = bias*dist
    
    return W_cib
    

#=======================================================================================================================
# FILE PATH NAME DECLARATIONS
#=======================================================================================================================


infile_dIdz = '/home/dekak/Desktop/CIB_delensing/data/interpolated_b_dIdz.csv'
infile_dist = '/home/dekak/Desktop/CMBS4xLSST/outputs/photoz_dist_tomographic_bins.txt'
outdir_theor = '/home/dekak/Desktop/CIB_delensing/theory_cls/'
outdir_eff_bias = '/home/dekak/Desktop/CIB_delensing/theory_cls/'


#=======================================================================================================================
# VARIABLE DECLARATIONS
#=======================================================================================================================


nside = 2048

lmin = 0
lmax = 7000

kmin = 1e-4
kmax = 20.0
c = 3.e5

bias = 1.3 #*np.array([0.95503496, 0.89696663, 0.90705486, 0.9100578, 0.87373673, 0.90124658, 0.89613075, 0.92179589])
alpha = 1.
bias_evol = True

gamma_0 = 6./11.
gamma_a = 0.

zbins_start = 1
zbins_end = 8

zmin_kk = 0
zmax_kk = 100
spacing_kk = 1000


theor_gg = True
theor_gg_cross = True
theor_kk = True
theor_kg = True
theor_cib_auto = True
theor_k_cib = True
theor_g_cib = True
theor_shot_noise = True


#=======================================================================================================================
# INITIALIZATION
#=======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
# Making matter power spectra at redshift z = 0
# ----------------------------------------------------------------------------------------------------------------------


pars = camb.set_params(WantTransfer=True,NonLinear='NonLinear_none',num_nu_massless = 2.046,
					    nu_mass_degeneracies=[0.0],share_delta_neff=True,nu_mass_fractions=[1.0],
					    nu_mass_numbers=[1],MassiveNuMethod='Nu_trunc',H0=67.32117,ombh2=0.0223828,omch2=0.1201075,
					    mnu=0.06451439,omk=0,tau=0.05430842)

pars.Transfer.high_precision=True
pars.Transfer.kmax=1.3464234
pars.InitPower.set_params(parameterization=1,ns=0.9660499,As=2.100549e-09)
pars.InitPower.At=0.0
pars.Reion.helium_delta_redshift = 0.5
pars.Reion.helium_redshiftstart = 6.0
pars.set_for_lmax(lmax=lmax,lens_potential_accuracy=4)
#pars.NonLinearModel.set_params(halofit_version='original')
#pars.NonLinearModel.Min_kh_nonlinear = 0.15

results = camb.get_results(pars)
chistar = results.conformal_time(0)-results.tau_maxvis

H0 = results.hubble_parameter(z=0.0)
Om0 = results.get_Omega('cdm',z=0)+results.get_Omega('baryon',z=0)+results.get_Omega('neutrino',z=0)

print ('Value of Hubble cosntant:',H0)
print ('Value of Density Parameter:',Om0)
print (r'Value of sigma_8:',results.get_sigma8_0())


ls = np.arange(lmin,lmax+1,dtype=np.float32)
meanz = np.zeros(shape=zbins_end-zbins_start+1)
gal_bias = np.zeros(shape=zbins_end-zbins_start+1)


gal_dist = np.loadtxt(infile_dist)
dz = (gal_dist[2:,0]-gal_dist[:-2,0])/2.
chis = results.comoving_radial_distance(gal_dist[:,0])
H_z = results.hubble_parameter(gal_dist[:,0])


D_z = np.zeros_like(gal_dist[:,0])
for x in range(len(gal_dist[:,0])):
    D_z[x] = Dz(gal_dist[x,0],gamma_0,gamma_a)

PK = camb.get_matter_power_interpolator(pars,nonlinear=True,hubble_units=False,k_hunit=False,kmax=kmax,
                                            var1=model.Transfer_tot,var2=model.Transfer_tot,zmax=gal_dist[-1,0])


#=======================================================================================================================
# COMPUTING KERNELS
#=======================================================================================================================

'''
for ii in range(zbins_start,zbins_end+1):
    print(ii)
    meanz[ii-zbins_start] = simps(gal_dist[:,0]*(gal_dist[:,ii-zbins_start+1]*chis)**2/H_z,gal_dist[:,0])/simps((gal_dist[:,ii-zbins_start+1]*chis)**2/H_z,gal_dist[:,0])
    gal_bias[ii-zbins_start] = bias/Dz(meanz[ii-zbins_start],gamma_0,gamma_a)

np.savetxt(outdir_eff_bias+'/effective_bias.txt',np.array([meanz,gal_bias]).T)
del meanz,gal_bias
'''

#=======================================================================================================================
# COMPUTING KERNELS
#=======================================================================================================================

'''
for ii in range(zbins_start,zbins_end+1):

    if theor_gg or theor_kg or theor_gg_cross is True:
	    Wg = gal_ker(gal_dist[1:-1,ii-zbins_start+1],gal_dist[1:-1,0],alpha,bias,chis[1:-1],H_z[1:-1])
    
    if theor_kg is True:
	    Wk = lens_ker(chis[1:-1],gal_dist[1:-1,0],H_z[1:-1])
	    
    if theor_gg_cross is True:	
        if ii < zbins_end - 1:
            Wg2 = gal_ker(gal_dist[1:-1,ii-zbins_start+2],gal_dist[1:-1,0],alpha,bias,chis[1:-1],H_z[1:-1])
            Wg3 = gal_ker(gal_dist[1:-1,ii-zbins_start+3],gal_dist[1:-1,0],alpha,bias,chis[1:-1],H_z[1:-1])
        elif ii < zbins_end :
            Wg2 = gal_ker(gal_dist[1:-1,ii-zbins_start+2],gal_dist[1:-1,0],alpha,bias,chis[1:-1],H_z[1:-1])

#=======================================================================================================================
# COMPUTING CLS
#=======================================================================================================================


#---------------------------------------------------------------------------------------------------------------
# COMPUTING GG SPECTRA
#---------------------------------------------------------------------------------------------------------------


    if theor_gg is True:

	    outfile_cl = outdir_theor+'/cl_gg_lmax_'+str(lmax)+'_bin_'+str(ii)+'.dat'
	    cl = np.zeros(shape=(lmax-lmin+1))
	    
	    for l in range(lmin,lmax+1):
		    k = (l+0.5)/chis[1:-1]
		    w = np.ones(k.shape)
		    w[k<kmin] = 0.
		    w[k>kmax] = 0.
	    
		    if bias_evol is True:
			    cl[l] = np.dot(dz,w*H_z[1:-1]*Wg**2*PK.P(0,k,grid=False)/chis[1:-1]**2/c)
		    if bias_evol is False:
			    cl[l] = np.dot(dz,w*H_z[1:-1]*Wg**2*PK.P(0,k,grid=False)*(D_z[1:-1])**2/chis[1:-1]**2/c)
		    
		    del k,w
		    
	    np.savetxt(outfile_cl,cl)
		    
	    del outfile_cl,cl,l

#---------------------------------------------------------------------------------------------------------------
# COMPUTING GG cross SPECTRA
#---------------------------------------------------------------------------------------------------------------

    if theor_gg_cross is True:
        if ii < zbins_end - 1 :
            
            outfile_cl = outdir_theor+'/cl_gg_lmax_'+str(lmax)+'_bin_'+str(ii)+str(ii+1)+'.dat'
            cl = np.zeros(shape=(lmax-lmin+1))
	        
            for l in range(lmin,lmax+1):
                k = (l+0.5)/chis[1:-1]
                w = np.ones(k.shape)
                w[k<kmin] = 0.
                w[k>kmax] = 0.

                if bias_evol is True:
                    cl[l] = np.dot(dz,w*H_z[1:-1]*Wg*Wg2*PK.P(0,k,grid=False)/chis[1:-1]**2/c)
                if bias_evol is False:
                    cl[l] = np.dot(dz,w*H_z[1:-1]*Wg*Wg2*PK.P(0,k,grid=False)*(D_z[1:-1])**2/chis[1:-1]**2/c)

                del k,w
		        
            np.savetxt(outfile_cl,cl)
		        
            del outfile_cl,cl,l

            outfile_cl = outdir_theor+'/cl_gg_lmax_'+str(lmax)+'_bin_'+str(ii)+str(ii+2)+'.dat'
            cl = np.zeros(shape=(lmax-lmin+1))

            for l in range(lmin,lmax+1):
                k = (l+0.5)/chis[1:-1]
                w = np.ones(k.shape)
                w[k<kmin] = 0.
                w[k>kmax] = 0.

                if bias_evol is True:
                    cl[l] = np.dot(dz,w*H_z[1:-1]*Wg*Wg3*PK.P(0,k,grid=False)/chis[1:-1]**2/c)
                if bias_evol is False:
                    cl[l] = np.dot(dz,w*H_z[1:-1]*Wg*Wg3*PK.P(0,k,grid=False)*(D_z[1:-1])**2/chis[1:-1]**2/c)

                del k,w

            np.savetxt(outfile_cl,cl)

            del outfile_cl,cl,l
	        
        elif ii < zbins_end :
            outfile_cl = outdir_theor+'/cl_gg_lmax_'+str(lmax)+'_bin_'+str(ii)+str(ii+1)+'.dat'
            cl = np.zeros(shape=(lmax-lmin+1))

            for l in range(lmin,lmax+1):
                k = (l+0.5)/chis[1:-1]
                w = np.ones(k.shape)
                w[k<kmin] = 0.
                w[k>kmax] = 0.

                if bias_evol is True:
                    cl[l] = np.dot(dz,w*H_z[1:-1]*Wg*Wg3*PK.P(0,k,grid=False)/chis[1:-1]**2/c)
                if bias_evol is False:
                    cl[l] = np.dot(dz,w*H_z[1:-1]*Wg*Wg3*PK.P(0,k,grid=False)*(D_z[1:-1])**2/chis[1:-1]**2/c)

                del k,w
	            
            np.savetxt(outfile_cl,cl)
                
            del outfile_cl,cl,l
	        
	        

#---------------------------------------------------------------------------------------------------------------
# COMPUTING KG SPECTRA
#---------------------------------------------------------------------------------------------------------------

    
    if theor_kg is True:

	    outfile_cl = outdir_theor+'/cl_kg_lmax_'+str(lmax)+'_bin_'+str(ii)+'.dat'
	    cl = np.zeros(shape=(lmax-lmin+1))
	    
	    for l in range(lmin,lmax+1):
		    k = (l+0.5)/chis[1:-1]
		    w = np.ones(k.shape)
		    w[k<kmin] = 0.
		    w[k>kmax] = 0.
	    
		    if bias_evol is True:
			    cl[l] = np.dot(dz,w*H_z[1:-1]*Wg*Wk*PK.P(0,k,grid=False)*(D_z[1:-1])/chis[1:-1]**2/c)
		    if bias_evol is False:
			    cl[l] = np.dot(dz,w*H_z[1:-1]*Wg*Wk*PK.P(0,k,grid=False)*(D_z[1:-1])**2/chis[1:-1]**2/c)
		    
		    del k,w
		    
	    np.savetxt(outfile_cl,cl)
		    
	    del outfile_cl,cl,l
    
    
    if theor_gg or theor_kg is True:del Wg
    if theor_kg is True: del Wk


del dz,chis,H_z,D_z

#---------------------------------------------------------------------------------------------------------------
# COMPUTING KK SPECTRA
#---------------------------------------------------------------------------------------------------------------


if theor_kk is True:
    
    zs = np.linspace(zmin_kk,zmax_kk,np.int64((zmax_kk-zmin_kk)*spacing_kk))
    dz = (zs[2:]-zs[:-2])/2.
    zs = zs[1:-1]
    chis = results.comoving_radial_distance(zs)
    H_z = results.hubble_parameter(zs)
    D_z = np.zeros_like(zs)
    
    
    for x in range(len(zs)):
	    D_z[x] = Dz(zs[x],gamma_0,gamma_a)
    
    Wk = lens_ker(chis,zs,H_z)
    np.savetxt('/home/dekak/Desktop/CMBS4xLSST/zphoto_cls/cmb_kk_window_function.dat',np.array([zs,Wk]).T)
    cl = np.zeros(shape=(len(ls)))
    
    for i,l in enumerate(ls):
	    k = (l+0.5)/chis
	    w = np.ones(k.shape)
	    w[k<kmin] = 0.
	    w[k>kmax] = 0.
	    cl[i] = np.dot(dz,w*H_z*Wk**2*PK.P(0,k,grid=False)*D_z**2/chis**2/c)
	    
	    del k,w


    outfile_cl = outdir_theor+'/cl_kk_lmax_'+str(lmax)+'.dat'
    np.savetxt(outfile_cl,cl)
    
    del zs,dz,chis,H_z,D_z,Wk,cl,outfile_cl,l
'''    

#=======================================================================================================================
#  CIB REDSHIFT DISTRIBUTION
#=======================================================================================================================


dIdz_dist = pd.read_csv(infile_dIdz, comment='#')

z_cib = dIdz_dist['z'].values
b_dIdz = dIdz_dist['b_dIdz'].values

## create cib distribution bins
cib_dist = np.zeros((len(z_cib), zbins_end + 2))
cib_dist[:,0] = z_cib
for ii in range(zbins_start, zbins_end+1):
    cib_dist[1:len(gal_dist[1:-1,0])+1,ii] = np.where(gal_dist[1:-1,ii] > 0, 1, 0)
cib_dist[1:-1, zbins_end+1] = np.where(z_cib[1:-1] > gal_dist[-1,0], 1, 0)

new_b_dIdz = np.where(cib_dist[:,0]<0.01, 0, b_dIdz)   # setting low z to zero


cib_bias = cib_bias_z(cib_dist[:,0])
dIdz = b_dIdz / cib_bias

## create galaxy distribution bins
gal_dist_highz = np.zeros((len(z_cib), zbins_end + 2))
gal_dist_highz[:,0] = z_cib
for ii in range(zbins_start, zbins_end+1):
    gal_dist_highz[1:len(gal_dist[1:-1,0])+1,ii] = np.where(gal_dist[1:-1,ii] > 0, gal_dist[1:-1,ii], 0)
gal_dist_highz[1:-1, zbins_end+1] = 0.

dz = (cib_dist[2:,0]-cib_dist[:-2,0])/2.
chis = results.comoving_radial_distance(cib_dist[:,0])
H_z = results.hubble_parameter(cib_dist[:,0])

D_z = np.zeros_like(cib_dist[:,0])
for x in range(len(cib_dist[:,0])):
    D_z[x] = Dz(cib_dist[x,0],gamma_0,gamma_a)


PK = camb.get_matter_power_interpolator(pars,nonlinear=True,hubble_units=False,k_hunit=False,kmax=kmax,
                                            var1=model.Transfer_tot,var2=model.Transfer_tot,zmax=cib_dist[-1,0])    


#=======================================================================================================================
# COMPUTING CIB CLS
#=======================================================================================================================
    
    
for ii in range(zbins_start,zbins_end+2):
    
    if theor_cib_auto or theor_k_cib or theor_g_cib is True:
        Wcib = new_b_dIdz[1:-1] * cib_dist[1:-1,ii-zbins_start+1]       
        
    if theor_k_cib is True:
        Wk = lens_ker(chis[1:-1],cib_dist[1:-1,0],H_z[1:-1])
	    
    if theor_g_cib is True:	
        Wg = gal_ker(gal_dist_highz[1:-1,ii-zbins_start+1],gal_dist_highz[1:-1,0],alpha,bias,chis[1:-1],H_z[1:-1])

#---------------------------------------------------------------------------------------------------------------
# COMPUTING GG SPECTRA
#---------------------------------------------------------------------------------------------------------------
    
    if theor_cib_auto is True:
    
        outfile_cl = outdir_theor+'/cl_cibcib_lmax_'+str(lmax)+'_bin_'+str(ii)+'.dat'
        cl = np.zeros(shape=(lmax-lmin+1))
	    
        for l in range(lmin,lmax+1):
            k = (l+0.5)/chis[1:-1]
            w = np.ones(k.shape)
            w[k<kmin] = 0.
            w[k>kmax] = 0.
	    
            if bias_evol is True:
                cl[l] = np.dot(dz,w*H_z[1:-1]*Wcib**2*PK.P(0,k,grid=False)*(D_z[1:-1])**2/chis[1:-1]**2/c)
            
            del k,w
		    
        np.savetxt(outfile_cl,cl)
            
        del outfile_cl,cl,l

#---------------------------------------------------------------------------------------------------------------
# COMPUTING GG cross SPECTRA
#---------------------------------------------------------------------------------------------------------------

    if theor_g_cib is True:
    
        outfile_cl = outdir_theor+'/cl_g_cib_lmax_'+str(lmax)+'_bin_'+str(ii)+'.dat'
        cl = np.zeros(shape=(lmax-lmin+1))
        
        for l in range(lmin,lmax+1):
            k = (l+0.5)/chis[1:-1]
            w = np.ones(k.shape)
            w[k<kmin] = 0.
            w[k>kmax] = 0.

            if bias_evol is True:
                cl[l] = np.dot(dz,w*H_z[1:-1]*Wg*Wcib*PK.P(0,k,grid=False)*(D_z[1:-1])**2/chis[1:-1]**2/c)
            
            del k,w
	        
        np.savetxt(outfile_cl,cl)
	        
        del outfile_cl,cl,l

#---------------------------------------------------------------------------------------------------------------
# COMPUTING KG SPECTRA
#---------------------------------------------------------------------------------------------------------------

    
    if theor_k_cib is True:

	    outfile_cl = outdir_theor+'/cl_k_cib_lmax_'+str(lmax)+'_bin_'+str(ii)+'.dat'
	    cl = np.zeros(shape=(lmax-lmin+1))
	    
	    for l in range(lmin,lmax+1):
		    k = (l+0.5)/chis[1:-1]
		    w = np.ones(k.shape)
		    w[k<kmin] = 0.
		    w[k>kmax] = 0.
	    
		    if bias_evol is True:
			    cl[l] = np.dot(dz,w*H_z[1:-1]*Wcib*Wk*PK.P(0,k,grid=False)*(D_z[1:-1])**2/chis[1:-1]**2/c)
			    
		    del k,w
		    
	    np.savetxt(outfile_cl,cl)
		    
	    del outfile_cl,cl,l
    
    
    if theor_gg or theor_kg is True:del Wg
    if theor_kg is True: del Wk
    
    
    if theor_shot_noise is True:
        outfile_cl = 'noise_cls/nl_cib_lmax_'+str(lmax)+'_bin_'+str(ii)+'.dat'
        cl = np.zeros(shape=(lmax-lmin+1))
        
        norm_Wcib = Wcib / simps(b_dIdz[1:-1] * cib_dist[1:-1,ii-zbins_start+1], cib_dist[1:-1,0])
        shot_noise_545 = 1690 * 1e-12  # convert from Jy^2 to MJy^2
        for l in range(lmin, lmax+1):
            cl[l] = np.dot(dz, norm_Wcib*shot_noise_545)
    
        np.savetxt(outfile_cl,cl)
        del outfile_cl, cl, norm_Wcib, shot_noise_545        
        
del dz,chis,H_z,D_z
  

#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------		
#---------------------------------------------------------------------------------------------------------------


end = time()
print('Time elasped for theoretical calculations:',(end-start),'seconds')


#=====================================================================================================================
# DEALLOCATION
#=====================================================================================================================


del infile_dist,outdir_theor
del nside,lmin,lmax,kmin,kmax,c,bias,alpha,bias_evol,gamma_0,gamma_a,zbins_start,zbins_end,zmin_kk,zmax_kk,spacing_kk
del theor_gg,theor_kk,theor_kg,pars,results,chistar,H0,Om0,ls,gal_dist,PK, gal_dist_highz, cib_dist, dIdz_dist
del start,end
