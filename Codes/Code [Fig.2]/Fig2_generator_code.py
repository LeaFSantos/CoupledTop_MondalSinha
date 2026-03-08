import numpy as np
import math as ma
import cmath
import scipy as sc
import matplotlib.pyplot as plt
from numpy import linalg as LA
import time
import random
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
import matplotlib.ticker as ticker
import qutip
from qutip import *
from scipy.integrate import odeint

plt.rcParams["font.family"] = "freeserif"

fig4, ax4 = plt.subplots()
fig5, ax5 = plt.subplots()
#######################################################################################

def Dissipative_CT(S,V):
      H = -sx1-sx2+(V/S)*sz1*sz2 + wz1 * sz1 + wz2 * sz2
      return H

#def dag(A):
#    return np.conjugate(A.T)

#def Expectation(A,psi):
#    return np.matmul(dag(psi), np.matmul(A,psi))


#===================================================================================================
'''                           PHASE OPERATOR AND CORRESPONDING QUANTITIES                        '''
#===================================================================================================

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

    #=== Phase variable in proper range ==========
    if i == 1 or i == 3:
        phi_avg = np.mod(phi_avg.real, 2*np.pi)


    sq_phi_avg = np.sum(phi_vals**2 * phase_distribution)

    exp_phi_avg = np.sum(np.exp(1j*delta*phi_vals) * phase_distribution)

    # ⟨cos(ϕ)⟩ = ∑ cos(ϕ_m) * p(ϕ_m)
    cos_phi = np.cos(phi_vals)
    cos_phi_avg = np.sum(cos_phi * phase_distribution)

    # Phase fluctuation:----------------------
    phase_fluctuation = sq_phi_avg-phi_avg**2

    
    return phi_avg, phase_fluctuation, cos_phi_avg, exp_phi_avg, phase_distribution

def Phase_Fluctuation(t,Psi):
    Psi = Qobj(Psi, dims=[[ 2*S+1, 2*S+1], [1, 1]])
    p_rho = Psi.ptrace([0]); 

    phi_vals, z_avg, phi_kets = phase_states(S,-np.pi)
    phi_avg, phase_fluctuation, cos_phi_avg, exp_phi_avg, phase_distribution = compute_phase_observables(p_rho, phi_kets, phi_vals)

    if phase_fluctuation  > (np.pi**2)/3.0:
        phi_vals, z_avg, phi_kets = phase_states(S,0)
        phi_avg, phase_fluctuation, cos_phi_avg, exp_phi_avg, phase_distribution = compute_phase_observables(p_rho, phi_kets, phi_vals)
    
    return [phi_avg, phase_fluctuation, cos_phi_avg, exp_phi_avg]

#===================================================================================================
'''                     STOCHASTIC WAVEFUCNTION METHOD [QUANTUM TRAJECTORIES]                    '''
#===================================================================================================

def Dynamics(preference,t0,dt,tmax,theta1,phi1,theta2,phi2):
    psi1 = spin_coherent(S, theta1, phi1); psi2 = spin_coherent(S, theta2, phi2)
    psi0 = tensor(psi1,psi2)

    S_x_minus = 0.5*(sx1-sx2); S_y_minus = 0.5*(sy1-sy2);   S_z_minus = 0.5*(sz1-sz2)

    operators = [sx1,sy1,sz1,sx2,sy2,sz2,  sx1*sx1,sy1*sy1,sz1*sz1, S_x_minus*S_x_minus, S_y_minus*S_y_minus, S_z_minus*S_z_minus, Phase_Fluctuation]

    tlist = np.arange(t0, tmax, dt)
  
    mcdata = qutip.mcsolve(H, psi0, tlist, Lindblad_list, operators, ntraj=100,progress_bar=True)
    Sx1_avg = mcdata.expect[0]/S; Sx2_avg = mcdata.expect[3]/S
    Sy1_avg = mcdata.expect[1]/S; Sy2_avg = mcdata.expect[4]/S
    Sz1_avg = mcdata.expect[2]/S; Sz2_avg = mcdata.expect[5]/S
    sq_Sx1_avg = mcdata.expect[6]/S**2;
    sq_Sy1_avg = mcdata.expect[7]/S**2;
    sq_Sz1_avg = mcdata.expect[8]/S**2;

    sq_Sx1_minus = mcdata.expect[9]/S**2;
    sq_Sy1_minus = mcdata.expect[10]/S**2;
    sq_Sz1_minus = mcdata.expect[11]/S**2;

    phi_avg = mcdata.expect[12][:,0]
    phase_fluctuation = mcdata.expect[12][:,1]
    cos_phi_avg= mcdata.expect[12][:,2]
    exp_phi_avg = mcdata.expect[12][:,3]

    return tlist, Sx1_avg,Sy1_avg,Sz1_avg, Sx2_avg,Sy2_avg,Sz2_avg, sq_Sx1_avg, sq_Sy1_avg, sq_Sz1_avg, sq_Sx1_minus,sq_Sy1_minus,sq_Sz1_minus,  phi_avg, phase_fluctuation, cos_phi_avg, exp_phi_avg

#===================================================================================================
'''                                      CLASSICAL DYNAMICS                                      '''
#===================================================================================================
def f(state, t): 
    sx1, sy1, sz1, sx2, sy2, sz2 = state
    
    dsx1dt =   -V * sy1 * sz2 +gamma * sx1 * sz1
    dsy1dt = sz1 + V * sx1 * sz2 +gamma * sy1 * sz1
    dsz1dt = -sy1 -gamma * (sx1 ** 2 + sy1 ** 2)
   
    dsx2dt =  -V * sy2 * sz1 +gamma * sx2 * sz2
    dsy2dt = sz2 +  V * sx2 * sz1 +gamma * sy2 * sz2
    dsz2dt = -sy2 -gamma * (sx2 ** 2 + sy2 ** 2)

    return  dsx1dt, dsy1dt, dsz1dt, dsx2dt, dsy2dt, dsz2dt

def motion(state0,t0,tmax,dt, f):
    t = np.arange(t0,tmax, dt)
    states = odeint(f, state0, t,rtol=10**-12, atol=10**-12)
    return states

def Initial_state(sz1,phi1,sz2,phi2):
    theta1 = np.arccos(sz1); theta2 = np.arccos(sz2);
    sx1 = np.sin(theta1)*np.cos(phi1); sy1 = np.sin(theta1)*np.sin(phi1)
    sx2 = np.sin(theta2)*np.cos(phi2); sy2 = np.sin(theta2)*np.sin(phi2)
    state1 = np.array([sx1,sy1,sz1, sx2,sy2,sz2])
    #----------------------------------
    sx1 = np.sin(theta1)*np.cos(phi1+0.01); sy1 = np.sin(theta1)*np.sin(phi1+0.01)
    sx2 = np.sin(theta2)*np.cos(phi2); sy2 = np.sin(theta2)*np.sin(phi2)
    state2 = np.array([sx1,sy1,sz1, sx2,sy2,sz2])
    return state1,state2

#@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@ FOURIER SPECTRUM @*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@

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



def Figure_Configuration(ax, x_min, x_max, y_min, y_max,  x_step, y_step, x_size, y_size, LS):
    ax.set_xlim(x_min,x_max)
    ax.set_ylim(y_min, y_max)

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3.5)

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(direction='in', which='both',length=10, width=3,labelsize=LS,pad=14)
    ax.tick_params(direction='in', which='major',length=20, width=3,labelsize=LS,pad=14)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')

    ax.xaxis.set_major_locator(ticker.MultipleLocator(x_step))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(y_step))
    ax.set_axisbelow(False)
    

    figure = plt.gcf()
    figure.set_size_inches(x_size,y_size)
    plt.tight_layout()

'''@@@@@@@@@@@^^^^^&&&&&&&&&&*****%%%%%%%%%%%%%%%!!!!!!!!!!!!%%%%%%%%%%%%%%%*****&&&&&&&&&&^^^^^@@@@@@@@@@@'''
#=============================================  MAIN SIMULATION ==============================================
'''@@@@@@@@@@@^^^^^&&&&&&&&&&*****%%%%%%%%%%%%%%%!!!!!!!!!!!!%%%%%%%%%%%%%%%*****&&&&&&&&&&^^^^^@@@@@@@@@@@'''

# System parameter:--------------------------------------
S = 50; dim = (2*S+1)**2; 

wz1 = 0.0; wz2 = 0.0; V = 0.5; gamma = 0.2; 

# Time parameter:--------------------------------------
t0 = 0.0; tmax = 50; dt = 0.1; t = np.arange(t0, tmax, dt)

# Calculation of spin operators for both spins :---------
sm = spin_Jm(S); sp = spin_Jp(S); sx = spin_Jx(S); sy = spin_Jy(S); sz = spin_Jz(S);  I = identity(2*S+1);
sz1 = tensor(sz,I);  sz2 = tensor(I,sz); sp1 = tensor(sp,I);  sp2 = tensor(I,sp); sm1 = tensor(sm,I);  sm2 = tensor(I,sm);
sx1 = tensor(sx,I);  sx2 = tensor(I,sx); sy1 = tensor(sy,I);  sy2 = tensor(I,sy);

# Hamiltonian;-------------------------------------------
H = Dissipative_CT(S,V)

# Collapse operators (Lindblad):-------------------------
L1 = np.sqrt(gamma/S)*sm1; L2 = np.sqrt(gamma/S)*sm2;  Lindblad_list = [L1, L2]



Color = ['red','blue','darkorchid','seagreen','gold']

#UVUVUVUVUVUVUV&&&&&&&&&&*****@@@@@@@@@@@==$$$$$$$$$$WYWYWYWYWYWYW$$$$$$$$$$==@@@@@@@@@@@*****&&&&&&&&&&UVUVUVUVUVUVUV
'''DYNAMICS OF SPIN COMPONENTS AND PHASE FLUCTUATION [averaged over few quantum trajectories] FOR JUDICIOUSLY SELECTED 
   INITIAL CONDITIONS [for loop with index i] ON PHASE PORTRAIT. 
'''
#UVUVUVUVUVUVUV&&&&&&&&&&*****@@@@@@@@@@@==$$$$$$$$$$WYWYWYWYWYWYW$$$$$$$$$$==@@@@@@@@@@@*****&&&&&&&&&&UVUVUVUVUVUVUV

for i in range(5):
    print(i)
    #@*@*@*@*@*@*@*@*@*@*@*@*@ IN THE "OSCILLATORY REGIME" [gamma=0.2, V=0.5] @*@*@*@*@*@*@*@*@*@*@*@*@
    if i == 0:
        theta1 = np.arccos(0.0+0.1); theta2 = theta1  
        phi1 = -np.arcsin(gamma); phi2 = phi1
        phi_vals, z_avg, phi_kets = phase_states(S,-np.pi)
        number = "I"
    elif i == 1:
        theta1 = np.arccos(0.0+0.1); theta2 = theta1  
        phi1 = -np.arcsin(gamma)+np.pi; phi2 = phi1
        phi_vals, z_avg, phi_kets = phase_states(S,0.0)
        number = "II"
    elif i == 2:
        theta1 = np.arccos(0.0+0.2); theta2 = theta1 
        phi1 = -np.arcsin(gamma)+0.7; phi2 = phi1
        phi_vals, z_avg, phi_kets = phase_states(S,-np.pi)
        number = "III"
    elif i == 3:
        theta1 = np.arccos(0.0+0.2); theta2 = theta1 
        phi1 = -np.arcsin(gamma)+np.pi-0.7; phi2 = phi1
        phi_vals, z_avg, phi_kets = phase_states(S,0.0)
        number = "IV"
    elif i == 4:
        theta1 = np.arccos(0.0+0.2); theta2 = theta1
        phi1 = -np.arcsin(gamma)+np.pi-1.5; phi2 = phi1
        phi_vals, z_avg, phi_kets = phase_states(S,0.0)
        number = "V"

    #@@@@@@@@@@@^^^^^&&&&&&&&&&*****!!!!!!!!!!!!>=<>=<>=<>=<>=<>=<!!!!!!!!!!!!*****&&&&&&&&&&^^^^^@@@@@@@@@@@
    '''                                         CLASSICAL DYNAMICS                                        '''
    #@@@@@@@@@@@^^^^^&&&&&&&&&&*****!!!!!!!!!!!!>=<>=<>=<>=<>=<>=<!!!!!!!!!!!!*****&&&&&&&&&&^^^^^@@@@@@@@@@@

    state1,state2 = Initial_state(np.cos(theta1),phi1,np.cos(theta2),phi2)
    t_value = np.arange(t0,tmax,dt)
    States1 = motion(state1,t0,tmax,dt,f);
    Sz1 = States1[:,2]; Sx1 = States1[:,0]; Sy1 = States1[:,1];  Phi = np.angle(Sx1+1j*Sy1)

    for kk in range(len(Phi)):
        Phi[kk] = np.mod(Phi[kk], 2*np.pi)

    #UVUVUVUVUVUVUV&&&&&&&&&&*****@@@@@@@@@@@==$$$$$$$$$$$$$$$$$$$$==@@@@@@@@@@@*****&&&&&&&&&&UVUVUVUVUVUVUV
    '''                                  QUANTUM DYNAMICS OF TRAJECTORIES                                '''
    #UVUVUVUVUVUVUV&&&&&&&&&&*****@@@@@@@@@@@==$$$$$$$$$$$$$$$$$$$$==@@@@@@@@@@@*****&&&&&&&&&&UVUVUVUVUVUVUV

    tlist, Sx1_avg,Sy1_avg,Sz1_avg, Sx2_avg,Sy2_avg,Sz2_avg, sq_Sx1_avg, sq_Sy1_avg, sq_Sz1_avg, sq_Sx1_minus,sq_Sy1_minus,sq_Sz1_minus,  phi_avg, phase_fluctuation, cos_phi_avg, exp_phi_avg = Dynamics("Monte_carlo",t0,dt,tmax,theta1,phi1,theta2,phi2)

    for kk in range(len(phi_avg)):
        phi_avg[kk] = np.mod(phi_avg.real[kk], 2*np.pi)

    #@@@@@@@@@@@^^^^^&&&&&&&&&&*****!!!!!!!!!!!!>=<>=<>=<>=<>=<>=<!!!!!!!!!!!!*****&&&&&&&&&&^^^^^@@@@@@@@@@@
    '''                                         DATA STORING                                              '''
    #@@@@@@@@@@@^^^^^&&&&&&&&&&*****!!!!!!!!!!!!>=<>=<>=<>=<>=<>=<!!!!!!!!!!!!*****&&&&&&&&&&^^^^^@@@@@@@@@@@
    #file1 = open('Single_Quantum_trajectory_Open_Coupled_top--S={}-V={}--gamma={}--wz={}--Physical_quantities_phase-Dynamics-initial_condition_V_trajectory_no_{}.dat'.format(S,V,gamma,wz1, jj),"w+")

    file1 = open('Average_Quantum_trajectory_Open_Coupled_top--S={}-V={}--gamma={}--wz={}--Physical_quantities_phase-Dynamics-initial_condition_{}_Ensemble=100.dat'.format(S,V,gamma,wz1, number),"w+")

    for j in range(len(tlist)):
        print(tlist[j],'\t',Sx1[j].real,'\t',Sy1[j].real,'\t',Sz1[j].real,'\t',Phi[j].real,'\t',Sx1_avg[j].real,'\t',Sy1_avg[j].real,'\t',Sz1_avg[j].real,'\t',Sx2_avg[j].real,'\t',Sy2_avg[j].real,'\t',Sz2_avg[j].real,'\t',sq_Sx1_avg[j].real,'\t',sq_Sy1_avg[j].real,'\t',sq_Sz1_avg[j].real,'\t',sq_Sx1_minus[j].real,'\t',sq_Sy1_minus[j].real,'\t',sq_Sz1_minus[j].real,'\t',phi_avg[j].real,'\t',phase_fluctuation[j].real,'\t',cos_phi_avg[j].real, file=file1)

    file1.close()

    #@*@*@*@*@*@*@*@*@*@*@*@*@*@*@* PLOTTING DATA @*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@
    if i == 0:
        fig4, ax4 = plt.subplots()
    Figure_Configuration(ax4, -1,1, 0, 2,  0.5, 0.5,  10, 8.5, 55.0)
    ax4.plot(Sz1,Phi/np.pi,'o',color=Color[i],ms=8)

    if i == 0:
        fig5, ax5 = plt.subplots()
    Figure_Configuration(ax5, -1,1, 0, 2,  0.5, 0.5,  10, 8.5, 55.0)
    ax5.plot(Sz1_avg[:200],phi_avg[:200]/np.pi,'o',color=Color[i],ms=8)
    
    if i == 2:
        fig1, ax1 = plt.subplots()
        Figure_Configuration(ax1, 0, 50, -1, 1,  10, 1,  10, 5.5, 55.0)
        ax1.plot(tlist,0.5*(Sz1_avg+Sz2_avg),'-',color='r',lw=4)
        ax1.plot(tlist,Sz1,'k--',lw=4)
        fig1.savefig('Fig2_(a).png')


        fig2, ax2 = plt.subplots()
        Figure_Configuration(ax2, 0, 50, -1, 1,  10, 1,  10, 3.5, 55.0)
        delta_Sz_minus = np.sqrt(sq_Sz1_minus-(0.5*(Sz1_avg-Sz2_avg))**2  )
        ax2.errorbar(tlist,0.5*(Sz1_avg-Sz2_avg),delta_Sz_minus,color='r',lw=4, errorevery=20)
        fig2.savefig('Fig2_(b).png')
        ax2.clear(); 

        # Frequency analysis of a single trajectory:---------------------------------
        fig3, ax3 = plt.subplots()
        Figure_Configuration(ax3, 0, 2.5, -0.01, 0.15,  0.5, 0.05,  10, 8.5, 60.0)
        Sz_plus = 0.5*(Sz1_avg+Sz2_avg)
        frequencies,Y = FFT(tlist,Sz_plus-np.mean(Sz_plus))
        ax3.plot(2*np.pi*frequencies,Y,'o-',color='r',ms=12,lw=3)
        fig3.savefig('Fig2_(c).png')
        ax3.clear(); 


fig4.savefig('Fig2_(d).png')
ax4.clear();  

fig5.savefig('Fig2_(e).png')
ax5.clear();  