#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

def vis_2D_motion(t,u,theta,L,frame_rate):
    theta = theta+np.pi/2
    margin = 0.1
    thetas_deg = np.round(np.array(theta)*180/np.pi,2)

    # Function to update the animation frame
    def update(i):
        ax1.cla()
        ax2.cla()

        ax1.plot(t[:i], u[:i], 'b-', linewidth=2)
        ax1.set_xlabel('Time (s)')
        ax1.set_xlim([t[0],t[-1]])
        ax1.set_ylim([u.min()-margin,u.max()+margin])
        ax1.set_ylabel('Applied Forces (Nm)')
        ax1.set_title('Applied Forces vs. Time')
        ax1.annotate(f'F: {u[i]:.2f} N', 
                     xy=(0.8, 0.85), 
                     xycoords='axes fraction',
                     fontsize=10, color='k')
        
        # Calculate the pendulum position over time
        x = L * np.cos(theta[i])
        y = L * np.sin (theta[i])
        # Draw the pendulum
        ax2.plot([0, x], [0, y], 'r', linewidth=2)
        ax2.plot(x, y, 'bo', markersize=10)
        ax2.axvline(x=0,color='k',linewidth=0.5,linestyle='--')
        ax2.axhline(y=0,color='k')
        
        # Set plot limits for visualization
        ax2.set_xlim(-L-margin, L+margin)
        ax2.set_ylim(0-margin,L+margin)
        ax2.set_aspect('equal')

        # Display theta in the top-right corner of the pendulum subplot
        ax2.annotate(r'$\theta$: {:.1f}°'.format(thetas_deg[i]),xy=(0.8, 0.85),xycoords='axes fraction',
                     fontsize=10,color='k')
        ax2.set_title(f'Time: {t[i]:.2f} seconds')

    # Create an animation
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
    ani = FuncAnimation(fig, update, frames=len(theta), repeat=False, interval=1000/frame_rate)
    plt.show()


def vis_3D_motion(t,u,theta,phi,L,frame_rate):
    margin = 0.1

    def update(i):
        ax1.cla()
        ax2.cla()

        # Plot external forces applied to pendulum
        ax1.plot(t[:i], u[:i,0], label='u1',linewidth=2)
        ax1.plot(t[:i], u[:i,1], label='u2',linewidth=2)
        ax1.set_xlabel('Time (s)')
        ax1.set_xlim([t[0],t[-1]])
        ax1.set_ylim([u.min()-margin,u.max()+margin])
        ax1.set_ylabel('Applied Forces (Nm)')
        ax1.set_title(f'Time: {t[i]:.2f} seconds')
        ax1.legend() 

        # Calculate the pendulum position in cartesian coordinates
        x = L * np.cos(theta[i]) * np.sin(phi[i])
        y = - L * np.sin (theta[i]) * np.sin(phi[i])
        z = L * np.cos(phi[i])

        # Draw the pendulum
        ax2.plot([0, x], [0, y], [0, z], 'r', linewidth=2)
        ax2.scatter(x, y, z, color='b', s=50)

        # Set plot limits for visualization
        ax2.set_xlim(-L-margin, L+margin)
        ax2.set_ylim(-L-margin,L+margin)
        ax2.set_zlim(0-margin,L+margin)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('z')

        # Add legend with angle values
        legend = [Line2D([0], [0], color='w', label=r'$\theta$: {:.1f}°'.format(theta[i] * 180 / np.pi)),
                           Line2D([0], [0], color='w', label=r'$\phi$: {:.1f}°'.format(phi[i] * 180 / np.pi))]
        ax2.legend(handles=legend, loc='upper left')


    # Create an animation
    fig = plt.figure(figsize=(8,8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])
    ax1, ax2 = plt.subplot(gs[0]), plt.subplot(gs[1], projection='3d')
    ani = FuncAnimation(fig, update, frames=len(theta), repeat=False, interval=1000/frame_rate)
    plt.show()
