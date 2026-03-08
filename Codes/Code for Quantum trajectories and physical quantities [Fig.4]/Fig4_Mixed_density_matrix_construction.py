import numpy as np
import math as ma
import time
import os
from qutip import Qobj, tensor, identity
from qutip import *
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
from matplotlib.colors import TwoSlopeNorm

plt.rcParams["font.family"] = "freeserif"


#######################################################################################

def Observables(state):
    # Observable at steady state:----------------------------

    Sx1_avg = expect(sx1, state)/S; Sx2_avg = expect(sx2, state)/S;
    Sy1_avg = expect(sy1, state)/S; Sy2_avg = expect(sy2, state)/S;
    Sz1_avg = expect(sz1, state)/S; Sz2_avg = expect(sz2, state)/S;

    sq_Sx1_avg = expect(sx1*sx1, state)/S**2;   sq_Sx2_avg = expect(sx2*sx2, state)/S**2; 
    sq_Sy1_avg = expect(sy1*sy1, state)/S**2;   sq_Sy2_avg = expect(sy2*sy2, state)/S**2;
    sq_Sz1_avg = expect(sz1*sz1, state)/S**2;   sq_Sz2_avg = expect(sz2*sz2, state)/S**2;

    delta_Sx1 = sq_Sx1_avg-Sx1_avg**2;       delta_Sx2 = sq_Sx2_avg-Sx2_avg**2
    delta_Sy1 = sq_Sy1_avg-Sy1_avg**2;       delta_Sy2 = sq_Sy2_avg-Sy2_avg**2
    delta_Sz1 = sq_Sz1_avg-Sz1_avg**2;       delta_Sz2 = sq_Sz2_avg-Sz2_avg**2


    Sx1_Sx2_avg = expect(sx1*sx2, state)/S**2;   
    Sy1_Sy2_avg = expect(sy1*sy2, state)/S**2;  
    Sz1_Sz2_avg = expect(sz1*sz2, state)/S**2;  

    sq_Sx_minus_avg = expect( (sx1-sx2)*(sx1-sx2)/4.0, state)/S**2;     sq_Sx_plus_avg = expect( (sx1+sx2)*(sx1+sx2)/4.0, state)/S**2; 
    sq_Sy_minus_avg = expect( (sy1-sy2)*(sy1-sy2)/4.0, state)/S**2;     sq_Sy_plus_avg = expect( (sy1+sy2)*(sy1+sy2)/4.0, state)/S**2;
    sq_Sz_minus_avg = expect( (sz1-sz2)*(sz1-sz2)/4.0, state)/S**2;     sq_Sz_plus_avg = expect( (sz1+sz2)*(sz1+sz2)/4.0, state)/S**2;

    delta_Sx_minus = sq_Sx_minus_avg-(Sx1_avg-Sx2_avg)**2/4.0;      delta_Sx_plus = sq_Sx_plus_avg-(Sx1_avg+Sx2_avg)**2/4.0
    delta_Sy_minus = sq_Sy_minus_avg-(Sy1_avg-Sy2_avg)**2/4.0;      delta_Sy_plus = sq_Sy_plus_avg-(Sy1_avg+Sy2_avg)**2/4.0
    delta_Sz_minus = sq_Sz_minus_avg-(Sz1_avg-Sz2_avg)**2/4.0;      delta_Sz_plus = sq_Sz_plus_avg-(Sz1_avg+Sz2_avg)**2/4.0

    return Sz1_avg,Sz2_avg,  Sy1_avg,Sy2_avg,  Sx1_avg,Sx2_avg,  delta_Sx1,delta_Sy1,delta_Sz1,  delta_Sx2,delta_Sy2,delta_Sz2,    Sx1_Sx2_avg,Sy1_Sy2_avg,Sz1_Sz2_avg,    delta_Sx_minus,delta_Sx_plus,  delta_Sy_minus,delta_Sy_plus, delta_Sz_minus,delta_Sz_plus

def FFT(T,A):
    N = len(A);  # sampling rate
    y = A
    Y = np.fft.fft(y)/N # fft computing and normalization
    Y = abs(Y[range(int(N/2.0))])
    sample_interval = T[2]-T[1]
    sample_frequency = 1.0/ sample_interval
    values = np.arange(int(N/2))
    timePeriod = N/sample_frequency
    frequencies = values/timePeriod
    return frequencies,Y

#===================== Phase coherence and fluctuation =======================

def phase_states(S, phi0):
    d = int(2 * S + 1)  # Dimension of Hilbert space
    phi_vals = np.zeros(d)
    phi_kets = []
    phi_dms = []

    # Define spin-z operator and its eigenstates
    #========================================================================
    # The atomic number difference (nL-nR) = Sz and the relative phase
    # are conjugate so to evaluate the phase state, we need Fourier transform
    # of Sz eigenstates;
    #========================================================================
    Sz = spin_Jz(S)
    eigvals, eigkets = Sz.eigenstates()

    # Loop over phase points
    for m in range(d):
        # Phase value: from -π to π (can change to 0 to 2π if preferred)
        phi_m = phi0 + (2 * np.pi * m) / d
        phi_vals[m] = phi_m

        # Initialize the phase state as zero ket
        phase_ket = 0 * eigkets[0]

        # Construct the phase state
        for n in range(-S,S+1, 1):
            coeff = np.exp(1j * n * phi_m)
            phase_ket += coeff * eigkets[n+S]/ np.sqrt(float(d))

        phase_ket = phase_ket.unit()

        phi_kets.append(phase_ket)

    # Average z value as cosine of phi
    z_avg = np.cos(phi_vals)

    return phi_vals, z_avg, phi_kets

def compute_phase_observables(p_rho, phi_kets, phi_vals):
    delta = 0.001
    # Compute phase distribution: p(ϕ_m) = Tr[ρ_S · ϕ_op[m]]

    phase_distribution = []
    for m in range(len(phi_vals)):
        dist_m = expect(p_rho, phi_kets[m])
        phase_distribution.append( dist_m.real )

    phase_distribution = np.array(phase_distribution)
    phase_distribution /= np.sum(phase_distribution)

    #phase_distribution = np.real([expect(phi_op_list[m], p_rho) for m in range(len(phi_vals))])
    #phase_distribution /= np.sum(phase_distribution)

    # ⟨ϕ⟩ = ∑ ϕ_m * p(ϕ_m)
    phi_avg = np.sum(phi_vals * phase_distribution)
    sq_phi_avg = np.sum(phi_vals**2 * phase_distribution)

    exp_phi_avg = np.sum(np.exp(1j*delta*phi_vals) * phase_distribution)

    # ⟨cos(ϕ)⟩ = ∑ cos(ϕ_m) * p(ϕ_m)
    cos_phi = np.cos(phi_vals)
    cos_phi_avg = np.sum(cos_phi * phase_distribution)

    # Phase fluctuation:----------------------
    phase_fluctuation = sq_phi_avg-phi_avg**2

    
    return phi_avg, phase_fluctuation, cos_phi_avg, exp_phi_avg, phase_distribution


def exp_complex(a):
	im = complex(0,1)
	return np.cos(a)+im*np.sin(a)

def dag(A):
    return np.conjugate(A.T)

def Expectation(A,psi):
    return np.matmul(dag(psi), np.matmul(A,psi))

def coherentspinstate(theta,phi,S):
	im = complex(0,1)
	dim = 2*S+1
	state = np.zeros(dim,dtype=complex)

	m = -S
	for i in range(dim):
		v = np.zeros(int(dim))
		v[int(dim)-i-1] = 1.0
		state = state + ma.sqrt(ma.factorial(2*S)/(ma.factorial(S-m)*ma.factorial(S+m)))*(ma.sin(theta/2.0)**(S-m))*(ma.cos(theta/2.0)**(S+m))*exp_complex((S-m)*phi)*v
		m=m+1
	state = state/np.linalg.norm(state)
	return state

def Husimi_spin(fig, ax,S,rho):
    Theta = np.linspace(np.pi,0, 200)
    Phi = np.linspace(0, 2*np.pi, 200)

    husimi = np.zeros(shape=(200,200))
    i = 0;
    for theta in Theta:
        j = 0
        for phi in Phi:
            psi = coherentspinstate(theta,phi,S)#spin_coherent(S, theta, phi)
            #psi = np.array(psi)
            A = Expectation(rho,psi)
            #print(theta,phi)
            husimi[i,j] = A#[0,0]
            j += 1
        i += 1

    cf  = ax.pcolormesh(np.cos(Theta),Phi/np.pi,husimi.T,cmap='turbo',shading = "gouraud", vmin=0)
    ax.tick_params(direction='out', length=6, width=2, colors='k',labelsize=40.0)
    cb = fig.colorbar(cf, ax=ax, ticks=ticker.MultipleLocator(0.1))
    cb.ax.tick_params(labelsize=45)

    file2 = open("Spin_Husimi_distribution_OCT-S={}-gamma={}-V={}-wz1={}--t={}.dat".format(S,gamma, V, wz1, t_val ),"w+")
    np.savetxt(file2,np.array(husimi),fmt='%.6f')
    file2.close()

    i, j = np.unravel_index(np.argmax(husimi), husimi.shape)

    Sz_max = np.cos(Theta[i]); Phi_max = Phi[j];

    return Sz_max, Phi_max


'''@@@@@@@@@@@^^^^^&&&&&&&&&&*****%%%%%%%%%%%%%%%!!!!!!!!!!!!%%%%%%%%%%%%%%%*****&&&&&&&&&&^^^^^@@@@@@@@@@@'''
#=============================================  MAIN SIMULATION ==============================================
'''@@@@@@@@@@@^^^^^&&&&&&&&&&*****%%%%%%%%%%%%%%%!!!!!!!!!!!!%%%%%%%%%%%%%%%*****&&&&&&&&&&^^^^^@@@@@@@@@@@'''

# ---------------- PARAMETERS ----------------
S = 2
dim = (2*S + 1)**2

wz1 = 0.5
wz2 = 0.5
V = 1.7
gamma = 0.2

# Simulation time array (same as used in generating wavevectors)
t0 = 0.0
tmax = 200.0
dt = 0.2
t = np.arange(t0, tmax, dt);  t = [0, 10, 199]

# Calculation of spin operators for both spins :-------------------------------------------------------------------
sm = spin_Jm(S); sp = spin_Jp(S); sx = spin_Jx(S); sy = spin_Jy(S); sz = spin_Jz(S);  I = identity(2*S+1);
sz1 = tensor(sz,I);  sz2 = tensor(I,sz); sp1 = tensor(sp,I);  sp2 = tensor(I,sp); sm1 = tensor(sm,I);  sm2 = tensor(I,sm);
sx1 = tensor(sx,I);  sx2 = tensor(I,sx); sy1 = tensor(sy,I);  sy2 = tensor(I,sy);
#------------------------------------------------------------------------------------------------------------------

outdir = f"WaveVectors_S={S}_V={V}_gamma={gamma}_wz={wz1}"

# ---------------- LOOP OVER TIME FILES ----------------
file1 = open('Mixed_density_matrix_Over_Quantum_trajectories_Open_Coupled_top--S={}-V={}--gamma={}--wz={}--Physical_quantities-initial_Chaotic_state_Final.dat'.format(S,V,gamma, wz1),"w+")

Delta_Sx_minus = []; Delta_Sy_minus = []; Delta_Sz_minus = []
Delta_Sx_plus = []; Delta_Sy_plus = []; Delta_Sz_plus = []; Entanglement_entropy = []; Phase_Fluctuation = [];  Relative_entropy_coherence = []
Purity = []

for t_val in t:
    # Construct filename using physical time
    fname = os.path.join(outdir, f"psi_t={t_val:.1f}.npy")

    # Load all trajectories at this time
    psi_t_array = np.load(fname)  # shape: (Ntraj, dim)
    Ntraj = psi_t_array.shape[0]

    rho_np = (psi_t_array.conj().T @ psi_t_array) / Ntraj   # BLAS-optimized
    rho = Qobj(rho_np,dims=[[2*S+1, 2*S+1], [2*S+1, 2*S+1]],copy=False)

    del psi_t_array

    #@$@$@$@$@$@$@$@$@$@$@$@*******&&&&&&&&&((%%%%%%%%%%%%%%%%%%%%%))&&&&&&&&&*******@$@$@$@$@$@$@$@$@$@$@$@#
    '''                                        PHYSICAL QUANTITIES                                        '''
    #@$@$@$@$@$@$@$@$@$@$@$@*******&&&&&&&&&((%%%%%%%%%%%%%%%%%%%%%))&&&&&&&&&*******@$@$@$@$@$@$@$@$@$@$@$@#
    Sz1_avg,Sz2_avg,  Sy1_avg,Sy2_avg,  Sx1_avg,Sx2_avg,  delta_Sx1,delta_Sy1,delta_Sz1,  delta_Sx2,delta_Sy2,delta_Sz2,    Sx1_Sx2_avg,Sy1_Sy2_avg,Sz1_Sz2_avg,    delta_Sx_minus,delta_Sx_plus,  delta_Sy_minus,delta_Sy_plus, delta_Sz_minus,delta_Sz_plus = Observables(rho)
    #&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%: YWYWYWYWYWYWYWYWYWYWYWYWY  :%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
    '''                                        SYNCHRONIZATION                                           '''
    #&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%: YWYWYWYWYWYWYWYWYWYWYWYWY  :%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

    Delta_Sx_minus.append(delta_Sx_minus);  Delta_Sx_plus.append(delta_Sx_plus)
    Delta_Sy_minus.append(delta_Sy_minus);  Delta_Sy_plus.append(delta_Sy_plus)
    Delta_Sz_minus.append(delta_Sz_minus);  Delta_Sz_plus.append(delta_Sz_plus)

    #@$@$@$@$@$@$@$@$@$@$@$@*******&&&&&&&&&((%%%%%%%%%%%%%%%%%%%%%))&&&&&&&&&*******@$@$@$@$@$@$@$@$@$@$@$@#
    '''                                   ENTANGLEMENT ENTROPY, PURITY                                    '''
    #@$@$@$@$@$@$@$@$@$@$@$@*******&&&&&&&&&((%%%%%%%%%%%%%%%%%%%%%))&&&&&&&&&*******@$@$@$@$@$@$@$@$@$@$@$@#
    # We are calculating mixed density matrix over quantum trajectories so total entropy is nonzero:----
    Entropy_total = entropy_vn(rho); Purity_total = (rho*rho).tr()
    p_rho_spin1 = rho.ptrace([0]);  Entropy_spin1 = entropy_vn(p_rho_spin1);  Purity_spin1 = (p_rho_spin1*p_rho_spin1).tr()
    p_rho_spin2 = rho.ptrace([1]);  Entropy_spin2 = entropy_vn(p_rho_spin2);  Purity_spin2 = (p_rho_spin2*p_rho_spin2).tr()

    Entanglement_entropy.append(Entropy_spin1);  Purity.append(Purity_spin1)

    # Relative entropy of coherence:---------------------------------------------------------------------
    rho_diag = qutip.Qobj(np.diag(rho.diag()), dims=rho.dims)
    S_diag = qutip.entropy_vn(rho_diag)

    Relative_entropy_coherence.append( S_diag - Entropy_total )

    #@$@$@$@$@$@$@$@$@$@$@$@*******&&&&&&&&&((%%%%%%%%%%%%%%%%%%%%%))&&&&&&&&&*******@$@$@$@$@$@$@$@$@$@$@$@#
    '''                                  PHASE COHERENCE AND FLUCTUAION                                   '''
    #@$@$@$@$@$@$@$@$@$@$@$@*******&&&&&&&&&((%%%%%%%%%%%%%%%%%%%%%))&&&&&&&&&*******@$@$@$@$@$@$@$@$@$@$@$@#
    phi_vals, z_avg, phi_kets = phase_states(S,0.0)
    phi_avg, phase_fluctuation, cos_phi_avg, exp_phi_avg, phase_distribution = compute_phase_observables(p_rho_spin1, phi_kets, phi_vals)

    index = np.argmax(phase_distribution);  max_phi = phi_vals[index]
    min_phi = max_phi-np.pi

    phi_vals, z_avg, phi_kets = phase_states(S,min_phi)
    phi_avg, phase_fluctuation, cos_phi_avg, exp_phi_avg, phase_distribution = compute_phase_observables(p_rho_spin1, phi_kets, phi_vals)

    print(np.round(t_val,2),'\t',phase_fluctuation,'\t',Purity_total,'\t',Purity_spin1,'\t',Entropy_spin1,'\t',S_diag - Entropy_total)

    Phase_Fluctuation.append(phase_fluctuation)
    #==================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================
    # Storing the quantities in a data file:--------------------------------
    print(t_val,'\t', Sx1_avg.real,'\t',Sy1_avg.real,'\t',Sz1_avg.real,'\t',Sx2_avg.real,'\t',Sy2_avg.real,'\t',Sz2_avg.real,'\t',      delta_Sx1.real,'\t',delta_Sy1.real,'\t',delta_Sz1.real,'\t',    delta_Sx2.real,'\t',delta_Sy2.real,'\t',delta_Sz2.real,'\t',    Sx1_Sx2_avg.real,'\t',Sy1_Sy2_avg.real,'\t',Sz1_Sz2_avg.real,'\t',      delta_Sx_minus.real,'\t',delta_Sy_minus.real,'\t',delta_Sz_minus.real,'\t',     delta_Sx_plus.real,'\t',delta_Sy_plus.real,'\t',delta_Sz_plus.real,'\t',    Entropy_total.real,'\t',Purity_total.real,'\t',Entropy_spin1.real,'\t',Purity_spin1.real,'\t',Entropy_spin2.real,'\t',Purity_spin2.real,'\t',phi_avg.real,'\t',phase_fluctuation.real,'\t',cos_phi_avg.real,'\t',S_diag - Entropy_total, file=file1)
    #==================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================

    #@$@$@$@$@$@$@$@$@$@$@$@*******&&&&&&&&&((%%%%%%%%%%%%%%%%%%%%%))&&&&&&&&&*******@$@$@$@$@$@$@$@$@$@$@$@#
    '''                                  STRUCTURE OF REDUCED DENSITY MATRICS                             '''
    #@$@$@$@$@$@$@$@$@$@$@$@*******&&&&&&&&&((%%%%%%%%%%%%%%%%%%%%%))&&&&&&&&&*******@$@$@$@$@$@$@$@$@$@$@$@#
    A = abs(p_rho_spin1.full().real)

    file2 = open("Spin_Reduced_density_matrix_OCT-S={}-gamma={}-V={}-wz1={}--t={}.dat".format(S,gamma, V, wz1, t_val ),"w+")
    np.savetxt(file2,np.array(A),fmt='%.6f')
    file2.close()

    fig1, ax1 = plt.subplots()
    im = ax1.imshow(A, cmap='YlGnBu_r', origin='lower', aspect='auto', vmin=0)#, norm=norm)
    cbar = fig1.colorbar(im, ax=ax1,  orientation='horizontal', location='top', ticks=ticker.MultipleLocator(0.1))
    cbar.ax.tick_params(labelsize=50)
    fig1.savefig('Fig4_(e)_t={}.png'.format(t_val))


    #@$@$@$@$@$@$@$@$@$@$@$@*******&&&&&&&&&((%%%%%%%%%%%%%%%%%%%%%))&&&&&&&&&*******@$@$@$@$@$@$@$@$@$@$@$@#
    '''                                        HUSIMI DISTRIBUTION                                        '''
    #@$@$@$@$@$@$@$@$@$@$@$@*******&&&&&&&&&((%%%%%%%%%%%%%%%%%%%%%))&&&&&&&&&*******@$@$@$@$@$@$@$@$@$@$@$@#
    fig2, ax2 = plt.subplots()
    Husimi_spin(fig2, ax2,S,p_rho_spin2.full())
    fig2.savefig('Fig4_(g)_t={}.png'.format(t_val))

    #@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@


#plt.show()