#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from emg_regression.utils.data_processing import Data_processing
from emg_regression.approximators.lstm import load_model, predict

# Load model
time_model = '02_26'
params_path = 'data/params_'+time_model+'.sav'
model_path  = 'data/lstm_'+time_model+'.pt'
model, params = load_model(params_path,model_path)
mu, std, max_emg =  params["mu"],  params["std"], params["maxs"]
window_size, offset =  params["window_size"], params["offset"]
dt = 0.01  # Time step

# Calculate the sigmoid functions over the time vector
sigmoid  = lambda a,t,T,sd,freq : 1 / (1 + np.exp(-a * t))
gauss    = lambda a,t,T,sd,freq : a * np.exp(-0.5 * ((t - T) / sd) ** 2)
sin      = lambda a,t,T,sd,freq: np.array([a * np.sin(2.0 * np.pi * freq * t)]).T
cos      = lambda a,t,T,sd,freq: a + np.array([a  * np.cos(2.0 * np.pi * freq * t)]).T
linear   = lambda a,t,T,sd,freq: np.array([a * t]).T
constant = lambda a,t,T,sd,freq: a * np.ones((len(t),1))
steps    = lambda a,t,T,sd,freq: np.repeat(np.linspace(0,a,3),len(t)//3)


generate_signals = lambda f,a,t,T,sd,freq : np.column_stack((f(a[0],t,T,sd,freq), f(a[1],t,T,sd,freq), f(a[2],t,T,sd,freq), f(a[3],t,T,sd,freq)))
inital_second = np.ones((int(1/dt),4))
t_ = lambda dt, t0, Tf: np.arange(t0,t0+Tf,dt)

# Plot the sigmoid functions
f = gauss
m1, m2, m3, m4 = max_emg*1.5
a0 = 0.0
count = 0

patterns = [[m1,a0,a0,a0],[a0,m2,a0,a0],[a0,a0,m3,a0],[a0,a0,a0,m4]]
# patterns = [[m1,m2,a0,a0],[a0,a0,m3,m4], [m1,a0,m3,a0], [a0,m2,a0,m4]]

# durations = [0.5,1,2,6,15]
durations = [10]
for T in durations:
    # T = 1.0 # minimum 1 second of signal to predict
    freq = (1/T) # *0.5
    sd = T/3

    # t1 = np.arange(0, 1, dt)
    # t2 = np.arange(1, T+1, dt)
    # t0 = np.append(t1,t2)
    t_init, t_mov = t_(dt,t0=0,Tf=1), t_(dt,t0=1,Tf=T)
    t0 = t_(dt,t0=0,Tf=2+T)
    x_init = np.ones((len(t_init),model.dim_input))

    fig, axes = plt.subplots(len(patterns),3,figsize=(10,6))

    for i, pattern in enumerate(patterns):
        # print(pattern)
        [a1,a2,a3,a4] = pattern
        
        x_gen = generate_signals(f,pattern,t_mov,T,sd,freq)
        x_gen = np.vstack((x_init*x_gen[0],x_gen))
        x_gen = np.vstack((x_gen,x_init*x_gen[-1]))

        emg_proc, ang_pred, tproc = predict(model=model,y=x_gen,window_size=window_size,offset=offset,
                                            time=t0)
        ang_pred *= 180/np.pi

        ax = axes[i,0]
        ax.plot(tproc, emg_proc, label=['RES','LES','REO','LEO'])
        ax.grid()

        ax = axes[i,1]
        ax.plot(tproc,ang_pred[:,0],label=r'$\theta$',c='darkblue')
        ax.plot(tproc,ang_pred[:,1],label=r'$\phi$',c='orange')
        ax.grid()   

        ax = axes[i,2]
        ax.plot(ang_pred[:,0],ang_pred[:,1],'k')
        ax.scatter(ang_pred[0,0],ang_pred[0,1],c='b')
        ax.scatter(ang_pred[-1,0],ang_pred[-1,1],c='r')    
        ax.set_xlim(-40,40)
        ax.set_ylim(-40,40)    
        ax.set_xlabel(r'$\theta$')
        ax.set_ylabel(r'$\phi$')
        ax.grid()

    axes[-1,0].legend()
    axes[-1,1].legend()
    plt.suptitle(f"T = {T} seconds")
    plt.tight_layout()
    # plt.savefig('figures/'+'fig_'+str(count)+'.png',format='png', dpi=300, bbox_inches='tight', facecolor='w')
    # count+=1
plt.show()


""" -------------------------------------------------------------- 
Input: trunk motion trajectory -> Output: EMG activations sequence
"""

# Load inverted model (receives positions, returns EMG activity)
time_model = '17_54'
params_path = 'data/params-1_'+time_model+'.sav'
model_path  = 'data/lstm-1_'+time_model+'.pt'
model_inv, params_inv = load_model(params_path,model_path)
mu, std =  params_inv["mu"],  params_inv["std"]
window_size, offset = params_inv["window_size"], params_inv["offset"]

# explore patterns that each trajectory will generate, and check time dependecy for simple line
p0 = 0
p1 = 60 * np.pi/180 #radians
p2 = 45 * np.pi/180 #radians

# different motions
frontal_flex = lambda t, p0, p1, Tf: np.array([0*t,p0+(p1 - p0)*t/Tf]).T
side_flex    = lambda t, p0, p1, Tf: np.array([p0+(p1 - p0)*t/Tf,0.1+0*t]).T

# Same initial and end position, but different velocities
Tf = 10 # Tf: seconds to reach p1
t_init, t_mov = t_(dt,t0=0,Tf=1), t_(dt,t0=1,Tf=Tf)
t0 = t_(dt,t0=0,Tf=1+Tf)
y1 = np.ones((len(t_init),model_inv.dim_input))

y_front = frontal_flex(t_mov,p0,p1,Tf)
y_back  = frontal_flex(t_mov,p0,-p1,Tf)
y_right = side_flex(t_mov,p0,p2,Tf)
y_left  = side_flex(t_mov,p0,-p2,Tf)

motions = [y_front,y_back,y_right,y_left]

fig, axes = plt.subplots(3,len(motions),figsize=(10,6))
colors  = ["blue","orange","g","r"]

for i, y_motion in enumerate(motions):
    y = np.vstack((y1*y_motion[0,:], y_motion))

    ax = axes[0,i] if len(motions)>1 else axes[0]
    ax.plot(t0,y)
    ax.grid()
    ax = axes[1,i] if len(motions)>1 else axes[1]
    ax.plot(y[:,0],y[:,1])
    ax.scatter(y[0,0],y[0,1],c='b')
    ax.scatter(y[-1,0],y[-1,1],c='r')
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.grid()

    # compare EMGs generated from these 2:
    angles_proc, emg_pred, tproc = predict(model=model_inv,y=y,window_size=window_size,offset=offset,time=t0)
    
    # EMG varying with angle is the same, even with different velocities?
    # slower movements, we have access to predictions for smaller angles
    ax = axes[2,i] if len(motions)>1 else axes[2]
    for j in range(model_inv.dim_output):
        ax.plot(tproc,emg_pred[:,j],color=colors[j])
        ax.set_xlabel('Time')
    ax.grid()
    axes[2,3].legend(['RES','LES','REO','LEO'], bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
plt.show()
