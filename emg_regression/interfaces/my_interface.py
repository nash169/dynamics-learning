import numpy as np
import pyautogui

class Interface():
    def __init__(self, length, H, home_loc):
        super(Interface, self).__init__()

        self._H = H # [theta_min,theta_max,phi_min,phi_max]
        self._length = length

        # Screen dimensions
        self.d1, self.d2 = pyautogui.size()
        self.cursor_radius = 25
        self.screen_margin = self.cursor_radius + 40

        # Start position on the screen
        self.home_loc = home_loc

    def __call__(self, state):
        """ Receive trunk state (2 angles) and projects them on xy plane (transverse plane)"""

        theta = state[0] if len(state.shape) == 1 else state[:,0]
        phi   = state[1] if len(state.shape) == 1 else state[:,1]

        # From spherical to cartesian coordinates
        # x =   self.length * np.sin(phi) * np.cos(theta)
        # y = - self.length * np.sin(phi) * np.sin(theta)

        # normalize by maximum/ minimum ROM
        qx = np.where(theta < 0, theta/self.H[0], theta/np.abs(self.H[1]))
        qy = np.where(phi < 0, phi/self.H[2], phi/np.abs(self.H[3]))

        # scale to screen
        px = qx * self.d1//2 + self.p_home[0]
        py = qy * self.d2//2 + self.p_home[1] if self.home_loc == 'center' else qy * self.d2 +  self.p_home[1]

        # bound to screen limits?

        return np.array([px,py]).T

    @property
    def p_home(self):
        if self.home_loc == "center":
            return self.d1//2, self.d2//2
        if self.home_loc == "bottom":
            return self.d1//2, self.d2 - self.screen_margin

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        self._length = value

    @property
    def H(self):
        return self._H

    @H.setter
    def H(self, value):
        self._H = value