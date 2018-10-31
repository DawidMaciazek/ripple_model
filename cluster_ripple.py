import numpy as np
from scipy import signal
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

class model:
    def __init__(self):
        # discretization setup
        self.dx = 1.0 # [ nm ]
        self.dt = 1.0 # [ s ]

        # setup cluster effective roughness (gauss)
        sigma = 5 # [ nm ]
        sigma2 = np.square(sigma)
        gauss_cutoff = 0.2

        gauss_r = np.sqrt(-2*sigma2*np.log(gauss_cutoff))

        odd_axis = int(np.ceil(gauss_r/self.dx))*2 + 1
        even_axis = int(np.ceil((gauss_r-self.dx*0.5)))*2 + 2

        odd_edge = odd_axis // 2
        odd_range = np.linspace(-odd_edge, odd_edge, odd_axis)

        even_edge = even_axis // 2 - 0.5 * self.dx
        even_range = np.linspace(-even_edge, even_edge, even_axis)

        gauss_x, gauss_y = np.meshgrid(even_range, odd_range)
        # assuming symetrical crater effect
        self.mask_x = np.exp(-(np.square(gauss_x)+np.square(gauss_y))/(2*sigma2))
        self.mask_x = self.mask_x/np.sum(self.mask_x)
        self.mask_y = self.mask_x.T
        if True:
            plt.imshow(self.mask_x)
            plt.show()

        # INITIALIZE
        self.theta_deg = 45.0
        self.theta =  np.radians(self.theta_deg)
        self.surf_slope = np.tan(-self.theta)

        self.xy_slope = self.surf_slope*np.sqrt(2)/2.0

        # calculate plane size
        node_number = 90
        self.range_x = np.arange(node_number)*self.dx
        self.box_size = node_number*self.dx

        self.X, self.Y  = np.meshgrid(self.range_x, self.range_x)
        self.Z = self.X*self.xy_slope+self.Y*self.xy_slope

        self.Z += np.random.random(self.Z.shape)
        #self.Z += 4*np.sin(self.Y*2*np.pi/25.0)

        self.slope_correction = -(self.box_size*self.xy_slope)

        self.Z_arr = []
        self.write_freq = 1

    def init_functions(self, infile):
        fun_values = np.loadtxt(infile)
        thetas = np.radians(fun_values[:,0])
        sputtering = fun_values[:,1]
        redist = fun_values[:,2]

        self.sputtering_fit = interp1d(thetas, sputtering, kind='cubic')
        self.redist_fit = interp1d(thetas, redist, kind='cubic')

    def single_step(self):
        roll_x = np.roll(self.Z, 1, axis=1)
        print(self.slope_correction)
        roll_x[:,0] += self.slope_correction
        slopes_x = roll_x - self.Z
        plt.imshow(slopes_x)
        plt.show()

        roll_y = np.roll(self.Z, 1, axis=0)
        roll_y[0,:] += self.slope_correction
        slopes_y = roll_y - self.Z

        slopes_x_conv = signal.convolve2d(slopes_x, self.mask_x,
                                          boundary='wrap', mode='same')
        slopes_y_conv = signal.convolve2d(slopes_y, self.mask_y,
                                          boundary='wrap', mode='same')

        normal_magnitude = np.sqrt(np.square(slopes_x_conv)+np.square(slopes_y_conv)+1.0)
        thetas = np.arccos(1.0/normal_magnitude)

        omegas = np.arctan2(slopes_y_conv,slopes_x_conv)

        sputtering = self.sputtering_fit(thetas)

        if True:
            plt.imshow(sputtering)
            plt.show()




m = model()
plt.imshow(m.Z)
plt.show()
m.init_functions('3000ar.txt')
m.single_step()

