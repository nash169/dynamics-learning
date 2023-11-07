#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from emg_regression.models.pendulum import Pendulum
from emg_regression.models.spherical_pendulum import SphericalPendulum
from emg_regression.utils.integrator import Integrator
from emg_regression.interfaces.my_interface import Interface
from emg_regression.utils.visualization import vis_3D_motion
import random
import pickle

dt = 0.01
length = 0.5
gravity = 9.81

# model
model = SphericalPendulum(length=length)

# integrator
integrator = Integrator(step=dt)

# simulate
T = 3
t = np.arange(0,T,dt)

# Define system input force (control and gravity compensation)

# generate input signal randomly (control forces for theta and phi)
u_c = lambda t,a1,a2,a3,b1,b2,b3: a1 * np.sin(a2*t + a3) + b1 * np.cos(b2*t + b3)

def generate_input(min_val0, max_val0, min_val1, max_val1):
    a1, a2, a3, b1, b2, b3 = [random.uniform(min_val0, max_val0) for _ in range(6)]
    c1, c2, c3, d1, d2, d3 = [random.uniform(min_val1, max_val1) for _ in range(6)]
    u_theta = u_c(t,a1,a2,a3,b1,b2,b3)
    u_phi   = u_c(t,c1,c2,c3,d1,d2,d3)
    u_control = np.array([u_theta,u_phi]).T
    return u_control

def random_initial_state():
    margin = np.pi/20
    theta_0 = random.choice([-1,1]) * np.random.uniform(margin, np.pi)
    phi_0   = random.choice([-1,1]) * np.random.uniform(margin, np.pi/2 - margin)
    return [theta_0, phi_0, 0.0, 0.0]


# gravity compensation
u_gravity = lambda phi: np.array([0, -(gravity/length)*np.sin(phi)])

# Create various trajectories with same time duration
min_utheta, max_utheta = -.5, .5
min_uphi, max_uphi     = -.5, .5

nb_traj = 100
u_traj = np.zeros((nb_traj,len(t),2))
x_traj = np.zeros((nb_traj,len(t),4))

uc_traj = np.zeros((nb_traj,len(t),2))
ug_traj = np.zeros((nb_traj,len(t),2))

# change x_init
x_init = [0.0, 0.0, 0.0, 0.0]

for i in range(nb_traj):
    u_control = generate_input(min_utheta, max_utheta, min_uphi, max_uphi)

    # Define initial conditions and integrate
    k  = 0
    t_ = dt
    # x_init = random_initial_state()
    x = np.array([x_init]) # state vec: theta, phi, theta_dot, phi_dot
    u = np.array([u_control[k] + u_gravity(x[-1][1])])
    a = np.array([integrator(model, x[-1], u[-1])[1]])
    
    u_g = np.array([u_gravity(x[-1][1])])

    while t_+dt < T:
        x_k = x[-1]
        u_k = u_control[k] + u_gravity(x_k[1])
        x_, a_ = integrator(model, x_k, u_k)

        u_g = np.vstack((u_g, u_gravity(x_k[1])[np.newaxis,:]))

        u = np.vstack((u, u_k[np.newaxis,:]))
        x = np.vstack((x, x_[np.newaxis,:]))
        a = np.vstack((a, a_[np.newaxis,:]))
        t_ += dt
        k += 1

    u_traj[i,:,:] = u
    x_traj[i,:,:] = x
    ug_traj[i,:,:] = u_g
    uc_traj[i,:,:] = u_control

# Show all trajectories
fig, axes = plt.subplots(4,5,figsize=(14,5))
for i, ax in enumerate(axes.flat):
    x_deg = x_traj[i][:,:2]*180/np.pi
    ax.plot(x_deg[:,0],x_deg[:,1])
    ax.scatter(x_deg[0,0] ,x_deg[0,1] ,c='blue')
    ax.scatter(x_deg[-1,0],x_deg[-1,1],c='red')
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$\phi$')
    ax.set_xlim([-200,200])
    ax.set_ylim([-200,200])
    ax.tick_params(axis='x', labelsize=9)  # Adjust the fontsize as needed for x-ticks
    ax.tick_params(axis='y', labelsize=9)  # Adjust the fontsize as needed for y-ticks
plt.tight_layout()
plt.show()

# Save trajectories
train_data = {'x':x_traj, 'u':u_traj, 'uc':uc_traj, 'ug':ug_traj, 't':t}

# data_path = './data/traj_center_100_5s.pkl'
# f = open(data_path, "wb")
# pickle.dump(train_data, f)
# f.close()

# Show

# vis = 1
# if vis:
#     # Input forces
#     fig = plt.figure(figsize=(8,2.5))
#     ax = fig.add_subplot(131)
#     ax.plot(t, u_control) # input
#     ax.set_title('Control input')
#     ax = fig.add_subplot(132)
#     ax.plot(t, ug) # input
#     ax.set_title('Control gravity')
#     ax = fig.add_subplot(133)
#     ax.plot(t, u) # input
#     ax.set_title('Input')
#     plt.tight_layout()
#     plt.show()

#     # System states
#     fig = plt.figure(figsize=(8,2.5))
#     ax = fig.add_subplot(131)
#     ax.plot(t, a) # input
#     ax.set_title("Acceleration")
#     ax = fig.add_subplot(132)
#     ax.plot(t, x[:,2:]*180/np.pi) # input
#     ax.set_title("Velocity")
#     ax = fig.add_subplot(133)
#     ax.plot(t,  x[:,:2]*180/np.pi) # input
#     ax.set_title("Theta (deg)")
#     plt.tight_layout()
#     plt.show()

#     # Produced motion path
#     # Projection on visual interface
#     H_ = np.array([-np.pi/3, np.pi/3, -np.pi/3, np.pi/3])
#     interface = Interface(length=length,H=H_,home_loc='center')
#     p = interface(x[:,:2])
#     x_deg = x*180/np.pi

#     fig = plt.figure(figsize=(8,2.5))
#     ax = fig.add_subplot(121)
#     ax.plot(x_deg[:,0],x_deg[:,1])
#     ax.scatter(x_deg[0,0] ,x_deg[0,1] ,c='blue')
#     ax.scatter(x_deg[-1,0],x_deg[-1,1],c='red')
#     ax.set_xlabel(r'$\theta$')
#     ax.set_ylabel(r'$\phi$')

#     ax = fig.add_subplot(122)
#     ax.plot(p[:,0],p[:,1])
#     ax.scatter(p[0,0] ,p[0,1] ,c='blue')
#     ax.scatter(p[-1,0],p[-1,1],c='red')
#     ax.set_xlim([0,interface.d1])
#     ax.set_ylim([interface.d2,0])
#     plt.tight_layout()
#     plt.show()


# # 3D coordinates
# theta, phi = x[:,0], x[:,1]
# margin = 0.1

# # Create a 3D plot
# fig = plt.figure(figsize=(3,4))
# ax = fig.add_subplot(111, projection='3d')

# for i in range(len(x)):

#     # Calculate the pendulum position in cartesian coordinates
#     e1 = length * np.cos(theta[i]) * np.sin(phi[i])
#     e2 = - length * np.sin (theta[i]) * np.sin(phi[i])
#     e3 = length * np.cos(phi[i])

#     # Plot the trajectory
#     if i == 0:
#         ax.scatter(e1, e2, e3,color='blue',s=50)
#         ax.plot([0,e1], [0,e2], [0,e3],color='grey')
#     if i == len(x)-1:
#         ax.scatter(e1, e2, e3,color='red',s=50)
#         ax.plot([0,e1], [0,e2], [0,e3],color='grey')
#     else:
#         ax.scatter(e1, e2, e3,color='k',s=5,alpha=0.3)

# # Set plot limits for visualization
# ax.set_xlim(-length-margin, length+margin)
# ax.set_ylim(-length-margin, length+margin)
# ax.set_zlim(0-margin,length+margin)

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('3D Trajectory')
# # ax.legend()
# plt.tight_layout()
# plt.show()






# # vis_3D_motion(t,u,theta=x[:,0],phi=x[:,1],L = model.length,frame_rate=200)
# # vis_2D_motion(t,u,theta=x[:,0],L = model.length,frame_rate=200)
