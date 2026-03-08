import numpy as np
import os
import qutip
from qutip import *
import matplotlib.pyplot as plt

#@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*
''' THIS CODE GENERATES ALL QUANTUM STATES FOR DIFFERENT STOCHASTIC TRAJECTORIES AND DIFFERENT TIMES. USE THE DATA FILES GENERATED
    FROM THIS FILE IN THE CODE "Mixed_density_matrix_construction.py" TO FURTHER COMPUTE THE PHYSICAL QUANTITIES, DENSITY MATRIX ETCS.'''
#@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*


#######################################################################################
# Hamiltonian
#######################################################################################

def Dissipative_CT(S, V):
    H = -sx1 - sx2 + (V/S)*sz1*sz2 + wz1*sz1 + wz2*sz2
    return H

#######################################################################################
# Monte-Carlo dynamics: RETURN WAVE VECTORS ONLY
#######################################################################################

def Dynamics_wavevectors(H, t_values, theta1, phi1, theta2, phi2):
    psi1 = spin_coherent(S, theta1, phi1)
    psi2 = spin_coherent(S, theta2, phi2)
    psi0 = tensor(psi1, psi2)

    opts = qutip.Options(
        keep_runs_results=True,
        store_states=True,
        nsteps=10000
    )

    result = qutip.mcsolve(
        H, psi0, t_values,
        c_ops=Lindblad_list,
        ntraj=1,              # ONE trajectory
        options=opts,
        progress_bar=False
    )

    # list of kets: result.states[0][ti]
    return result.states[0]

#######################################################################################
# PARAMETERS
#######################################################################################

S = 2
dim = (2*S + 1)**2

'''Regular regime parameters:=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*'''
#wz1 = 0.0; wz2 = 0.0
#V = 0.5; gamma = 0.2

'''Transient chaos regime parameters:=*=*=*=*=*=*=*=*=*=*=*=*'''
#wz1 = 0.0; wz2 = 0.0
#V = 1.7; gamma = 0.2

'''Steady state chaos regime parameters:=*=*=*=*=*=*=*=*=*=*='''
wz1 = 0.5; wz2 = 0.5
V = 1.7; gamma = 0.2

#@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@*@

t0 = 0.0
tmax = 200
dt = 0.2
t = np.arange(t0, tmax, dt);
Nt = len(t)

#######################################################################################
# Initial conditions
#######################################################################################

def Initial_conditions_from_Symmetric_class(i):
    if i == 0:
        theta1 = np.arccos(0.1); theta2 = theta1
        phi1 = -np.arcsin(gamma); phi2 = phi1
    elif i == 1:
        theta1 = np.arccos(0.1); theta2 = theta1
        phi1 = -np.arcsin(gamma) + np.pi; phi2 = phi1
    elif i == 2:
        theta1 = np.arccos(0.2); theta2 = theta1
        phi1 = -np.arcsin(gamma) + 0.7; phi2 = phi1
    elif i == 3:
        theta1 = np.arccos(0.2); theta2 = theta1
        phi1 = -np.arcsin(gamma) + np.pi - 0.7; phi2 = phi1
    elif i == 4:
        theta1 = np.arccos(0.2); theta2 = theta1
        phi1 = -np.arcsin(gamma) + np.pi - 1.5; phi2 = phi1

    return theta1, theta2, phi1, phi2



theta1= 1.2480525464086845 ;theta2= 2.483326615395263 ;phi1= 1.0040948093191977 ;phi2= -0.5780531993227633  #---------> Current initial state

print("theta1=",theta1,';theta2=',theta2,  ";phi1=",phi1,';phi2=',phi2,)

psi1 = spin_coherent(S, theta1, phi1);  psi2 = spin_coherent(S, theta2, phi2);  psi0 = tensor(psi1, psi2)

#######################################################################################
# Spin operators
#######################################################################################

sm = spin_Jm(S)
sp = spin_Jp(S)
sx = spin_Jx(S)
sy = spin_Jy(S)
sz = spin_Jz(S)
I  = identity(2*S+1)

sz1 = tensor(sz, I)
sz2 = tensor(I, sz)
sm1 = tensor(sm, I)
sm2 = tensor(I, sm)
sx1 = tensor(sx, I)
sx2 = tensor(I, sx)

#######################################################################################
# Lindblad operators
#######################################################################################

L1 = np.sqrt(gamma/S) * sm1
L2 = np.sqrt(gamma/S) * sm2
Lindblad_list = [L1,L2]

#######################################################################################
# Hamiltonian
#######################################################################################

H = Dissipative_CT(S, V)

#######################################################################################
# OUTPUT DIRECTORY (TIME-CHUNKED)
#######################################################################################

outdir = f"WaveVectors_S={S}_V={V}_gamma={gamma}_wz={wz1}"
os.makedirs(outdir, exist_ok=True)

#######################################################################################
# MAIN LOOP OVER TRAJECTORIES
#######################################################################################


Ntraj = 5000    # number of trajectories

dim = psi0.full().size

# create memory-mapped files once
for t_phys in t:
    fname = f"{outdir}/psi_t={t_phys:.1f}.npy"
    np.lib.format.open_memmap(
        fname,
        mode='w+',
        dtype=np.complex64,
        shape=(Ntraj, dim)
    )

for traj in range(Ntraj):
    print(f"Trajectory {traj}")

    States = Dynamics_wavevectors(H, t, theta1, phi1, theta2, phi2)

    for ti, psi in enumerate(States):
        fname = f"{outdir}/psi_t={t[ti]:.1f}.npy"

        psi_mm = np.lib.format.open_memmap( fname, mode='r+', dtype=np.complex64)

        psi_mm[traj] = psi.full().ravel()  

    del States
