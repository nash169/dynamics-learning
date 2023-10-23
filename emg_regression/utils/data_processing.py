#!/usr/bin/env python
import numpy as np
from scipy import signal
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

class Data_processing():
    def __init__(self, data, degrees, downsample_factor):
        self.data = data

        self.ds_factor = downsample_factor if downsample_factor is not None else 1
        self.fs = 2000/self.ds_factor
        self.fc = 1
        self.degrees = degrees

        self.muscles = ['ES right','ES left', 'OEA right','OEA left']
        self.colors  = ["blue","orange","g","r"]
        
    def load_data(self):
        data = self.data
        self.emg_chs = [0,1,2,3]
        self.nb_ch   = len(self.emg_chs)

        self.t            = np.array(data['time'])         [::self.ds_factor]
        self.emgdata      = np.array(data['emgdata'])      [::self.ds_factor,self.emg_chs]
        self.emgfeatures  = np.array(data['emgfeatures'])  [::self.ds_factor,[0,2,4,6]] # get RMS only
        self.eul          = np.array(data['eul'])          [::self.ds_factor] 
        self.quat         = np.array(data['quat'])         [::self.ds_factor]
        self.linacc       = np.array(data['linacc'])       [::self.ds_factor]
        self.angvel       = np.array(data['angvel'])       [::self.ds_factor]
        self.torso_angles = np.array(data['torso_angles']) [::self.ds_factor]
        self.desCmd       = np.array(data['desCmd'])       [::self.ds_factor]
        self.cursorPos    = np.array(data['cursorPos'])    [::self.ds_factor]
        self.targetPos    = np.array(data['targetPos'])    [::self.ds_factor]
        self.homePos      = np.array(data['homePos'])      [::self.ds_factor]
        self.control_mode = data['control_modality']  

        self.t = self.t - self.t[0]
        self.get_torso_ang(eul0=self.eul[0,:])
        self.get_torso_angvel()

        if not self.degrees: 
            self.torso_angles = self.torso_angles * np.pi/180
            self.torso_vel = self.torso_vel * np.pi/180

    def get_torso_ang(self,eul0):
        """ Returns torso right flex angle and frontal flex angles"""
        R_init  = R.from_euler('xyz', eul0, degrees=True).as_matrix()
        R_trunk = R.from_euler('xyz', self.eul, degrees=True).as_matrix()
        R_diff  = np.matmul(R_init.T, R_trunk)
        eul_torso = R.from_matrix(R_diff).as_euler('xyz', degrees=True)
        self.torso_angles = np.array([eul_torso[:, 2], -eul_torso[:, 1]]).T

    def get_torso_angvel(self):
        """ Returns torso right flex angle and frontal flex angular velocity (rad/s)"""
        ang0 = self.angvel[0,:]
        R_init  = R.from_euler('xyz', ang0, degrees=True).as_matrix()
        R_trunk = R.from_euler('xyz', self.angvel, degrees=True).as_matrix()
        R_diff  = np.matmul(R_init.T, R_trunk)
        vel_torso = R.from_matrix(R_diff).as_euler('xyz', degrees=True)
        torso_angle_vel = np.array([vel_torso[:, 2], -vel_torso[:, 1]]).T

        # Filter
        Bf, Af = signal.butter(2, self.fc/(self.fs/2), 'low') 
        self.torso_vel = signal.filtfilt(Bf,Af,torso_angle_vel,axis=0)*180/np.pi #deg        

    def pre_process(self,vis,filt=1):
        """ Filters features and normalizes them. Data visualization."""
        
        # Filter the features
        Bf, Af = signal.butter(2, self.fc/(self.fs/2), 'low') 
        self.features_filt = signal.filtfilt(Bf,Af,self.emgfeatures,axis=0)
        self.features = self.features_filt if filt else self.emgfeatures

        # Normalize EMG data
        self.features_mean, self.features_std = self.features.mean(0), self.features.std(0)
        self.features_norm = (self.features -self.features_mean)/ self.features_std
        self.features_norm_by_max = self.features/ np.max(self.features,0)

        # Normalize output
        self.output_mean, self.output_std = self.torso_angles.mean(0), self.torso_angles.std(0)
        self.output_norm = (self.torso_angles -self.output_mean)/ self.output_std

        if vis:
            fig = plt.figure(figsize=(6,6))
            ax = fig.add_subplot(411)
            for i in range(self.emgfeatures.shape[1]): 
                ax.plot(self.t, self.emgfeatures[:, i]/np.max(self.emgfeatures[:, i]), lw=1,label=self.muscles[i], color=self.colors[i])
            ax.set_title('Raw EMG features (RMS)',fontsize=7,weight='bold')

            ax = fig.add_subplot(412)
            for i in range(self.features_norm.shape[1]):
                ax.plot(self.t, self.features_norm[:, i],label=self.muscles[i], color=self.colors[i])
            ax.set_title('Filtered and normalized EMG features',fontsize=7,weight='bold')

            ax = fig.add_subplot(413)
            ax.plot(self.t,self.torso_angles, label=['left flex', 'front flex'])
            ax.set_title('Trunk angles and position commands',fontsize=7,weight='bold')

            ax = fig.add_subplot(414)
            ax.plot(self.t,self.torso_vel, label=['left flex', 'front flex'])
            ax.set_title('Trunk angular velocity (lat, front)',fontsize=7,weight='bold')
            plt.show()

            plt.subplots(figsize=(4,2))
            plt.plot(self.cursorPos[:,0],self.cursorPos[:,1])
            plt.show()