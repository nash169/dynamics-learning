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
        self.twindow_s = 0.1

        self.muscles = ['ES right','ES left', 'OEA right','OEA left']
        self.colors  = ["blue","orange","g","r"]
        
    def load_data(self):
        data = self.data
        self.emg_chs = [0,1,2,3]
        self.nb_ch   = len(self.emg_chs)

        self.t            = np.array(data['time'])         [::self.ds_factor]
        self.emgdata      = np.array(data['emgdata'])      [::self.ds_factor,self.emg_chs]
        self.emgfeatures  = np.array(data['emgfeatures'])  [::self.ds_factor] # get RMS only
        self.eul          = np.array(data['eul'])          [::self.ds_factor] 
        self.quat         = np.array(data['quat'])         [::self.ds_factor]
        self.linacc       = np.array(data['linacc'])       [::self.ds_factor]
        self.angvel       = np.array(data['angvel'])       [::self.ds_factor]
        self.torso_angles = np.array(data['torso_angles']) [::self.ds_factor]
        self.desCmd       = np.array(data['desCmd'])       [::self.ds_factor]
        # self.trialID      = np.array(data['trialID'])      [::self.ds_factor]
        # self.cursorPos    = np.array(data['cursorPos'])    [::self.ds_factor]
        # self.targetPos    = np.array(data['targetPos'])    [::self.ds_factor]
        # self.homePos      = np.array(data['homePos'])      [::self.ds_factor]
        self.control_mode = data['control_modality']  

        self.t = self.t - self.t[0]
        # self.get_torso_ang(eul0=self.eul[0,:])
        # self.get_torso_angvel()

        # if not self.degrees: 
        #     self.torso_angles = self.torso_angles * np.pi/180
        #     self.torso_vel = self.torso_vel * np.pi/180

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

    def get_emgfeatures_online(self,fc=[50,400]):
        emgdata = self.emgdata
        fs = 2000
        twindow = int(0.1 * fs)
        buffer  = int(0.03 * fs)
        nfc1, nfc2 = fc[0]/(fs/2), fc[1]/(fs/2)
        A, B = signal.butter(7, [nfc1, nfc2], 'band') 
        rms = lambda x: np.sqrt(np.mean(x**2,0))
        features = np.array([]).reshape((0,emgdata.shape[1]))
        for i in range(0,len(emgdata),twindow):
            try:
                emg_tw = emgdata[i:i+twindow,:]
            except:
                emg_tw = emgdata[i:,:]
            data_to_filt = emg_tw if i==0 else np.vstack((buffer_tw,emg_tw))
            emg_tw_filt = signal.lfilter(A,B,data_to_filt,axis=0)
            emg_tw_filt = emg_tw_filt if i==0 else emg_tw_filt[buffer:]
            features = np.vstack((features,rms(emg_tw_filt)))
            buffer_tw = emg_tw[-buffer:]
        self.emg_rms = features

    def get_emgfeatures(self,emgdata,y,t):
        twindow = int(self.twindow_s * self.fs)
        self.twindow = twindow
        rms = lambda x: np.sqrt(np.mean(x**2,0))
        nzc = lambda x: ((x[:-1] * x[1:]) < 0).sum(0)

        features = np.array([]).reshape((0,emgdata.shape[1]*2))
        target = np.array([]).reshape((0,y.shape[1]))
        time = np.array([])
        for i in range(0,len(emgdata),twindow):
            try:
                emg_tw = emgdata[i:i+twindow,:]
                angles_tw = y[i:i+twindow,:]
                t_tw = t[i]
            except:
                emg_tw = emgdata[i:,:]
                angles_tw = y[i:,:]
            features_tw = np.append(rms(emg_tw),nzc(emg_tw))
            features = np.vstack((features,features_tw))
            target = np.vstack((target,angles_tw.mean(0)))
            time = np.append(time,t_tw)
        
        self.features, self.target, self.time = features, target, time
        self.fs_features = 1/self.twindow_s
        return features,target,time

    def filter_features(self):
        fc = 1
        bf,af = signal.butter(4, fc/(self.fs_features/2), 'low') 
        self.features_filt = signal.filtfilt(bf,af, self.features,axis=0)

        # bf,af = signal.butter(4, fc/(data.fs_features/2), 'low') 
        # features_filt = signal.filtfilt(bf,af, features,axis=0)
        return self.filter_features

    def process_emg(self,vis):
        x  = self.emgdata
        fs = self.fs
        fc = [50,400]
        A, B = signal.butter(8, [fc[0]/(fs/2), fc[1]/(fs/2)], 'band')
        sigBP = signal.filtfilt(A,B,x,axis=0)
        emg_rms,self.angles,self.t_f = self.get_emgfeatures(sigBP,self.torso_angles,self.t)

        fs_f = 10 # 0.1s - 1 sample 
        Af,Bf = signal.butter(4, 1/(fs_f/2), 'low')
        self.emg_rms = signal.lfilter(Af,Bf,emg_rms,axis=0)
        self.fs_features = fs_f

        if vis:
            fig = plt.figure(figsize=(16,9))
            ax = fig.add_subplot(211)
            ax.plot(self.t_f,self.emg_rms)
            ax = fig.add_subplot(212)
            ax.plot(self.t_f,self.angles)
            plt.show()


    def pre_process_features(self,vis,filt=1):
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
            plt.plot(self.torso_angles[:,0],self.torso_angles[:,1])
            plt.show()