#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from emg_regression.models.pendulum import Pendulum
from emg_regression.utils.integrator import Integrator

# model
model = Pendulum(length=2.0)

# integrator
integrator = Integrator(step=0.01)

# simulate
t = 0.0
T = 3.0
x = np.array([[np.pi/4, 0.0]])
f = lambda x : np.sin(1.5*x)
u = np.array([0.0])


while (t <= T):
    u = np.append(u, f(t))
    x = np.append(x, integrator(model, x[-1,:], u[-1])[np.newaxis,:], axis=0)
    t += integrator.step

np.save('data/train_x', u)
np.save('data/train_y', x[:,0])

fig = plt.figure()
ax = fig.add_subplot(311)
ax.plot(np.arange(0,T,integrator.step), x[:-2,0])
ax = fig.add_subplot(312)
ax.plot(np.arange(0,T,integrator.step), x[:-2,1])
ax = fig.add_subplot(313)
ax.plot(np.arange(0,T,integrator.step), u[:-2])
plt.show()
