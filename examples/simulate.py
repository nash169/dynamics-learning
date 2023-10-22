#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from emg_regression.models.pendulum import Pendulum
from emg_regression.models.spherical_pendulum import SphericalPendulum
from emg_regression.utils.integrator import Integrator
from emg_regression.interfaces.my_interface import Interface
from emg_regression.utils.visualization import vis_3D_motion

# model
# model = Pendulum(length=0.5)
model = SphericalPendulum(length=0.5)

# integrator
integrator = Integrator(step=0.01)

# simulate
freq = 0.5
T = 10

# 2D case
t = np.array([0.0])
x = np.array([[0.0, 0.0, 0.0, 0.0]]) # state vec
f1 = lambda x : np.array([0.8 * np.sin(2 * np.pi * freq * x), 0.0])
f2 = lambda x : np.array([0.0, 0.2 * np.sin(2 * np.pi * freq * x)])
f3 = lambda x : np.array([-0.3 * np.sin(2 * np.pi * freq * x), 0.0])

# f  = lambda x : np.array([1 * np.sin(2 * np.pi * freq * x), 0.5 * np.sin(2 * np.pi * freq * x)])
# f1 = lambda x : np.array([0.8 * np.sin(2 * np.pi * freq * x), 0.1 * np.sin(2 * np.pi * freq * x)])
# f2 = lambda x : np.array([0.1 * np.sin(2 * np.pi * freq * x), 0.2 * np.sin(2 * np.pi * freq * x)])

u = np.array([f1(t[-1])])
a = np.array([integrator(model, x[-1,:], u[-1])[1]])

while (t[-1] < T):
    f = f1 if t[-1] < T/4 else (f3 if t[-1] > T/3 else f2)
    u = np.append(u, f(t[-1])[np.newaxis,:],axis=0)
    x_, a_ = integrator(model, x[-1,:], u[-1])
    x = np.append(x, x_[np.newaxis,:], axis=0)
    a = np.append(a, a_[np.newaxis,:], axis=0)
    t = np.append(t, t[-1] + integrator.step)

# np.save('data/train_x', u)
# np.save('data/train_y', x)

# np.save('data/test_x', u)
# np.save('data/test_y', x)

fig = plt.figure(figsize=(5,8))
ax = fig.add_subplot(511)
ax.plot(t, u) # input
ax.set_ylabel("Input")
ax = fig.add_subplot(512)
ax.plot(t, a) 
ax.set_ylabel("Acceleration")
ax = fig.add_subplot(513)
ax.plot(t, x[:,2:]*180/np.pi) 
ax.set_ylabel("Velocity")
ax = fig.add_subplot(514)
ax.plot(t, x[:,:2]*180/np.pi)
ax.set_ylabel("Theta (deg)")

ax = fig.add_subplot(515)
ax.plot(x[:,0]*180/np.pi,x[:,1]*180/np.pi)
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$\phi$')
plt.tight_layout()
plt.show()


# Projection on visual interface
H_ = np.array([-np.pi/3, np.pi/3, -np.pi/3, np.pi/3])
interface = Interface(length=0.5,H=H_,home_loc='center')

p = interface(x[:,:2])

fig = plt.figure(figsize=(5,8))
ax = fig.add_subplot(211)
ax.plot(x[:,0]*180/np.pi,x[:,1]*180/np.pi)
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$\phi$')
ax = fig.add_subplot(212)
ax.plot(p[:,0],p[:,1])
ax.set_xlim([0,interface.d1])
ax.set_ylim([interface.d2,0])
 
plt.show()

# # vis_3D_motion(t,u,theta=x[:,0],phi=x[:,1],L = model.length,frame_rate=200)
# # vis_2D_motion(t,u,theta=x[:,0],L = model.length,frame_rate=200)
