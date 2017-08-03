import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib.widgets import Slider
import sys

class MTools:
    @staticmethod
    def gauss2d(x, x0, xs, y, y0, ys):
        return np.exp(-( (np.power(x-x0,2)/(2*np.power(xs,2))) +
                         (np.power(y-y0,2)/(2*np.power(ys,2))) ))


class SurfEnergyDepositionModel:
    def __init__(self, **kwargs):
        self.sample_len_u = kwargs.get('sample_len', 100)
        self.dist_scale = 1.0/self.sample_len_u

        self.nodes = kwargs.get('nodes', 50)

        self.xy_spacing = np.linspace(0, 1.0, self.nodes, endpoint=False)
        self.xy_d = self.xy_spacing[1]
        self.Z = np.zeros((self.nodes, self.nodes), dtype=float)

        self.X = np.tile(self.xy_spacing, (self.nodes, 1))
        self.Y = self.X.T

        # ion erosion
        self.erosion_u = kwargs.get('erosion', 0.000001)  # [A(z)/A^2(xy)*s]
        self.erosion = (self.erosion_u*self.xy_spacing)/self.dist_scale

        self.theta_deg = kwargs.get('theta', 60)
        self.theta = np.radians(self.theta_deg)

        self.decay_u = kwargs.get('decay', 0.05)  # [1/A]
        self.decay = self.decay_u/self.dist_scale

        self.shift_u = kwargs.get('shift', 30)  # [A]
        self.shift = self.shift_u*self.dist_scale

        self.xsigma_u = kwargs.get('xsigma', 30)  # [A]
        self.xsigma = self.xsigma_u*self.dist_scale

        self.xion_range = 4*self.xsigma*self.xsigma*np.sin(self.theta)
        if self.xion_range >= 1.0:
            raise ValueError("Ion range bigger than sample len")

        self.xion_indexes = np.sum(self.xy_spacing < self.xion_range)

        self.ysigma_u = kwargs.get('ysigma', 5)  # [A]
        self.ysigma = self.ysigma_u*self.dist_scale

        self.yion_range = self.ysigma*3
        if self.yion_range >= 0.5:
            raise ValueError("Ion range y bigger than sample len")

        self.yion_indexes = np.sum(self.xy_spacing < self.yion_range)
        self.yion_last = self.yion_indexes-1

        self.x_gauss = np.exp(
            np.power(-(self.xy_spacing[:self.xion_indexes]-self.shift)
                     *(1.0/np.sin(self.theta)), 2)
            / (2.0*np.power(self.xsigma, 2)))

        self.y_gauss = np.exp(-np.power(self.xy_spacing[:self.yion_indexes], 2)
                              / (2.0*np.power(self.ysigma, 2)))

        self.y_gauss = np.append((self.y_gauss[::-1])[:-1], self.y_gauss)
        # normalise to 1
        self.y_gauss *= 1.0/np.sum(self.y_gauss)

        #self.y_gauss

        self.rms_u = kwargs.get("rms", 5) # [A]
        self.rms = self.rms_u*self.dist_scale
        self.rms_cut = kwargs.get("rmscut", 0.05)
        self.rms_slope = -np.log(self.rms_cut/(1-self.rms_cut))/self.rms

        # diffusion
        self.diffusion_u = kwargs.get('diffusion', 10)
        self.diffusion = (self.diffusion_u*np.power(self.dist_scale,2))/np.power(self.xy_d, 2)

        self.Z_history = []
        self.Z_initialized = False

        print self.xion_indexes, self.yion_indexes

    def add_gauss(self, amp, x0, xs, y0, ys):
        self.Z += amp*MTools.gauss2d(self.X, x0*self.dist_scale, xs*self.dist_scale,
                                 self.Y, y0*self.dist_scale, ys*self.dist_scale)

    def add_sin(self, amp, nx, ny):
        self.Z += amp*self.dist_scale*np.sin(self.X*(2*nx*np.pi)+self.Y*(2*ny*np.pi))

    def show(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_wireframe(self.X, self.Y, self.Z)
        plt.show()

    def show_history(self):
        #fig, ax = plt.subplots()
        #plt.subplots_adjust(bottom=0.25)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')


        ax.plot_wireframe(self.X, self.Y, self.Z_history[0])

        axslider = plt.axes([0.15, 0.1, 0.65, 0.05])
        slider = Slider(axslider, "Tmp", 0, len(self.Z_history)-1, valinit=0)

        def update(val):
            selection = self.Z_history[int(val)]
            ax.clear()
            ax.plot_wireframe(self.X, self.Y, selection)

        slider.on_changed(update)
        plt.show()



    def run_single(self):
        if self.Z_initialized == False:
            self.Z_initialized = True
            self.Z_history.append(self.Z.copy())

        # ion erosion
        Z_pad = np.pad(self.Z, [[(self.yion_indexes-1)*2, (self.yion_indexes-1)*2], [self.xion_indexes-1, self.xion_indexes-1]], 'wrap')
        Z_erosion = np.zeros(Z_pad.shape, dtype=float)
        # print Z_pad.shape, self.xion_indexes, self.yion_indexes
        # loop over y dim
        #self.x_gauss
        for i_y in xrange((self.yion_indexes-1), self.nodes+3*(self.yion_indexes-1)):
            for i_x in xrange(0, self.nodes+self.xion_indexes-1):
                for j_y in xrange(-self.yion_indexes+1, self.yion_indexes):
                    Z_erosion[i_y+j_y][i_x:(i_x+self.xion_indexes)] += (
                    self.x_gauss*self.y_gauss[j_y+self.yion_last] )
        self.Z -= Z_erosion[2*(self.yion_indexes-1):self.nodes+2*(self.yion_indexes-1), self.xion_indexes-1:self.nodes+self.xion_indexes-1]


        # diffusion part
        # along x axis
        Z_pad = np.pad(self.Z, 1, 'wrap')
        x_diff = np.diff(Z_pad, 2, 1)[1:-1]
        y_diff = np.diff(Z_pad, 2, 0)[:,1:-1]

        self.Z += self.diffusion*(x_diff + y_diff)

        self.Z_history.append(self.Z.copy())

model = SurfEnergyDepositionModel(diffusion=1.0, nodes=20)
#model.add_gauss(0.1,50,7, 44,3)
model.add_sin(100,2,0)

for i in range(10):
    model.run_single()
model.show_history()