import time
import random
import numpy as np
import scipy as sc
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from multiprocessing import Process
import multiprocessing
from numpy import linalg as LA

plt.rcParams["font.family"] = "freeserif"

#fig, ax = plt.subplots(1,2)
#ax1=ax[0]; ax2=ax[1]

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# Definitions:
'''def f(state, t): 
    sx1, sy1, sz1, sx2, sy2, sz2 = state
    
    dsx1dt = -wzz * sy1*sz2 - w1z*sy1 
    dsy1dt = w1z*sx1 - w1x*sz1 - wxx*sz1*sx2 + wzz*sx1*sz2 
    dsz1dt = w1x*sy1 + wxx*sy1*sx2 
   
    dsx2dt = -wzz * sy2*sz1 - w2z*sy2 
    dsy2dt = w2z*sx2 - w2x*sz2 - wxx*sz2*sx1 + wzz*sx2*sz1 
    dsz2dt = w2x*sy2 + wxx*sy2*sx1
   
    return  dsx1dt, dsy1dt, dsz1dt, dsx2dt, dsy2dt, dsz2dt'''

def f(state, t): 
    sx1, sy1, sz1, sx2, sy2, sz2 = state
    
    dsx1dt = -wzz * sy1*sz2 - w1z*sy1 + gamma*sx1*sz1
    dsy1dt = w1z*sx1 - w1x*sz1 - wxx*sz1*sx2 + wzz*sx1*sz2 + gamma*sy1*sz1
    dsz1dt = w1x*sy1 + wxx*sy1*sx2 -gamma*(sx1**2+sy1**2)
   
    dsx2dt = -wzz * sy2*sz1 - w2z*sy2 + gamma*sx2*sz2
    dsy2dt = w2z*sx2 - w2x*sz2 - wxx*sz2*sx1 + wzz*sx2*sz1 + gamma*sy2*sz2
    dsz2dt = w2x*sy2 + wxx*sy2*sx1 -gamma*(sx2**2+sy2**2)
   
    return  dsx1dt, dsy1dt, dsz1dt, dsx2dt, dsy2dt, dsz2dt

def motion(state0,t0,tmax,f):
    t = np.arange(t0, tmax, dt)
    states = odeint(f, state0, t,rtol=10**-12, atol=10**-12)
    return states

def PHI1(phi1,S_x1,S_y1):
    for j in range (0,len(phi1)):
        if S_x1[j] < 0.0 and S_y1[j] < 0.0:
           phi1[j] = phi1[j]-np.pi
        elif S_x1[j] < 0.0 and S_y1[j] > 0.0:
           phi1[j] = phi1[j]+np.pi
    return phi1

def PHI(phi1,S_x1,S_y1):
    for j in range (0,len(phi1)):
        if phi1[j]< -np.pi/2:
           phi1[j] = 2*np.pi+phi1[j]
        '''elif phi1[j]> 1.5*np.pi:
           phi1[j] = -2*np.pi+phi1[j]'''
    return phi1

def SKIP(S_z1,phi1,skip):
        z1 = []
        ph1 = []
        for j in range(0,len(phi1),skip+1):
            z1.append(S_z1[j])
            ph1.append(phi1[j])
        ph1 = np.array(ph1)
        return z1,ph1

def plot_visible(azimuth, elev,x_coord_2,y_coord_2,z_coord_2,color,ms,Line):
    # plot empty plot, with points (without a line)
    points, = ax1.plot([],[],[],Line, markersize=ms, lw=4, color=color)
    ax1.view_init(elev, azimuth )
    #transform viewing angle to normal vector in data coordinates
    a = azimuth*np.pi/180. -np.pi
    e = elev*np.pi/180. - np.pi/2.
    X = [ np.sin(e) * np.cos(a),np.sin(e) * np.sin(a),np.cos(e)]
    # concatenate coordinates
    Z = np.c_[x_coord_2, y_coord_2, z_coord_2]
    # calculate dot product
    # the points where this is positive are to be shown
    cond = (np.dot(Z,X) >= 0)
    # filter points by the above condition
    x_c = x_coord_2[cond]
    y_c = y_coord_2[cond]
    z_c = z_coord_2[cond]
    # set the new data points
    points.set_data(x_c, y_c)
    points.set_3d_properties(z_c, zdir="z")
    fig.canvas.draw_idle()
    fig.tight_layout()

def Wireframe(azimuth, elev):
    PHI = np.arange(0,2*np.pi,0.1*np.pi); THETA = np.arange(0,np.pi,0.05*np.pi)
    for i in range(len(PHI)):
        phi = PHI[i]; theta = np.linspace(0, np.pi+0.001, 10000)
        x = np.sin(theta)*np.cos(phi); y = np.sin(theta)*np.sin(phi); z = np.cos(theta)
        plot_visible(azimuth, elev,x,y,z,'silver',1.5,'o')
        #ax1.plot(x,y,z,'b-')

        theta = THETA[i]; phi = np.linspace(0, 2*np.pi+0.001, 10000)
        x = np.sin(theta)*np.cos(phi); y = np.sin(theta)*np.sin(phi); z = np.array([np.cos(theta)]*len(phi))
        plot_visible(azimuth, elev,x,y,z,'silver',1.5,'o')
        #ax1.plot(x,y,z,'r-')

def Initial_state(sz1,phi1,sz2,phi2):
    theta1 = np.arccos(sz1); theta2 = np.arccos(sz2); 
    sx1 = np.sin(theta1)*np.cos(phi1); sy1 = np.sin(theta1)*np.sin(phi1);  sz1 = np.cos(theta1)
    sx2 = np.sin(theta2)*np.cos(phi2); sy2 = np.sin(theta2)*np.sin(phi2);  sz2 = np.cos(theta2)
    state1 = np.array([sx1,sy1,sz1, sx2,sy2,sz2])
    return state1

def State_picker():
    # Plotting parameters:=======================================
    u1 = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100); x = np.outer(np.cos(u1),np.sin(v)); y = np.outer(np.sin(u1), np.sin(v)); z = np.outer(np.ones(np.size(u1)),np.cos(v))

    elev,azimuth = 35, 15
    Wireframe(elev,azimuth)

    ax1.set_zlim3d(-1, 1); ax1.set_ylim3d(-1.2, 1.2); ax1.set_xlim3d(-1.2, 1.2)

    ax1.set_xlabel('$X$'); ax1.set_ylabel('$Y$'); ax1.set_zlabel('$Z'); ax1.set_axis_off()

    fig.set_size_inches(20,8)

    ax2.set_xlim(-1,1); ax2.set_ylim(-1,1)
    ###########################################################################################
    j = 0; i = 0
    Color = ['blueviolet','deeppink', 'springgreen','orangered','deepskyblue','turquoise','gold','cyan']

    #while True:
    for j in range(N):
        pts = []
        pts = np.array(plt.ginput(1, timeout=-1))
        #---------------------------------------------------
        sz1 = pts[0][0]
        phi1 = pts[0][1]*np.pi
        theta1 = np.arccos(sz1)
        #+++++++++++++++++++++++++++++

        #@$@$@$@$@$@$@$@$@$@$=====%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&=====@$@$@$@$@$@$@$@$@$@$
        '''                          INITIAL PHASE SPACE POINT                        '''
        #@$@$@$@$@$@$@$@$@$@$=====%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&=====@$@$@$@$@$@$@$@$@$@$

        #sz1, phi1, sz2, phi2 = Initial_Phase_space_points(abs(wzz), j)

        sz1 = np.random.uniform(-1,1); sz2 = np.random.uniform(-0.5,0.5); 
        phi1 = np.random.uniform(-np.pi,np.pi); phi2 = np.random.uniform(-np.pi,np.pi);

        theta2 = np.arccos(sz2)
        state1 = Initial_state(sz1,phi1,sz2,phi2)

        #@$@$@$@$@$@$@$@$@$@$=====%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&=====@$@$@$@$@$@$@$@$@$@$
        '''                    STROBOSCOPIC TIME EVOLUTION EVOLUTION                 '''
        #@$@$@$@$@$@$@$@$@$@$=====%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&=====@$@$@$@$@$@$@$@$@$@$

        states0 = motion(state1,t0,tmax,f)
        Sx1 = states0[:,0]; Sy1 = states0[:,1]; Sz1 = states0[:,2];


        Phi1 = np.angle(Sx1+1j*Sy1)

        ax2.plot(Sz1, Phi1/np.pi,'o',markersize = 2,lw=3)

        #@$@$@$@$@$@$@$@$@$@$=====%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&=====@$@$@$@$@$@$@$@$@$@$
        '''                         PLOTTING OVER BLOCH SPHERE                        '''
        #@$@$@$@$@$@$@$@$@$@$=====%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&=====@$@$@$@$@$@$@$@$@$@$
        ax1.plot(Sx1,Sy1,Sz1,'-',color=Color[j],lw=2, markersize=5, alpha = 1); 
        print(j%N)

        #plot_visible(35,10, Sx1[:],Sy1[:],Sz1[:],Color[j],3,'o')   #-148,-20


        ax1.view_init(10,35) 
        fig.tight_layout()
           
 

###########################################################################################
# control parameters:--
w1x = -1.0
w2x = -1.0
w1z = -0.0
w2z = -0.0

wxx = -0.0; wzz =  1.7;        gamma = 0.2

N = 3

t0 = 0.0; tmax = 100.0; dt = 0.1;

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#=========================================================================================
fig=plt.figure();
ax1 = fig.add_subplot(projection='3d');  fig, ax2 = plt.subplots()

A = State_picker()
#------------------------------------------------------------------------------------------
plt.show()




