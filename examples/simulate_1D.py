#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from emg_regression.models.pendulum import Pendulum
from emg_regression.utils.integrator import Integrator
from emg_regression.utils.visualization import vis_2D_motion

# model
model = Pendulum(length=0.5)

# integrator
integrator = Integrator(step=0.01)

# simulate
freq = 0.5
T = 5

# 2D case
t = np.array([0.0])
x = np.array([[0.0, 0.0]]) # state vec
f = lambda x : 1*np.sin(2*np.pi*freq*x)

u = np.array([f(t[-1])])
a = np.array([integrator(model, x[-1,:], u[-1])[1]])


while (t[-1] < T):
    u = np.append(u, f(t[-1]))
    x_, a_ = integrator(model, x[-1,:], u[-1])
    x = np.append(x, x_[np.newaxis,:], axis=0)
    a = np.append(a, a_[np.newaxis,:], axis=0)
    t = np.append(t, t[-1] + integrator.step)

# np.save('data/train_x', u)
# np.save('data/train_y', x[:,0])

fig = plt.figure(figsize=(5,8))
ax = fig.add_subplot(411)
ax.plot(t, u) # input
ax.set_ylabel("Input")
ax = fig.add_subplot(412)
ax.plot(t, a) 
ax.set_ylabel("Acceleration")
ax = fig.add_subplot(413)
ax.plot(t, x[:,1]*180/np.pi) 
ax.set_ylabel("Velocity")
ax = fig.add_subplot(414)
ax.plot(t, x[:,0]*180/np.pi)
ax.set_ylabel("Theta (deg)")
plt.tight_layout()
plt.show()


# vis_3D_motion(t,u,theta=x[:,0],phi=x[:,1],L = model.length,frame_rate=200)
vis_2D_motion(t,u,theta=x[:,0],L = model.length,frame_rate=200)
