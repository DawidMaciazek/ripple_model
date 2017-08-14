import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib.widgets import Slider
from matplotlib import cm

# TO DO:
# add gauss dist integration normalization
# >> for flat surface assumption >>

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
        print "SPACING", self.xy_spacing[1]
        self.xy_d = self.xy_spacing[1]
        self.Z = np.zeros((self.nodes, self.nodes), dtype=float)

        self.X = np.tile(self.xy_spacing, (self.nodes, 1))
        self.Y = self.X.T

        # ion erosion
        self.erosion_u = kwargs.get('erosion', 0.1)  # [A(z)/A^2(xy)*s]
        self.erosion = (self.erosion_u*self.xy_spacing[1])/self.dist_scale
        print self.erosion

        self.theta_deg = kwargs.get('theta', 60)
        self.theta = np.radians(self.theta_deg)

        self.decay_u = kwargs.get('decay', 0.05)  # [1/A]
        self.decay = self.decay_u/self.dist_scale

        self.shift_u = kwargs.get('shift', 30)  # [A]
        self.shift = self.shift_u*self.dist_scale

        self.xsigma_u = kwargs.get('xsigma', 30)  # [A]
        self.xsigma = self.xsigma_u*self.dist_scale

        self.xion_range = 4*self.xsigma*np.sin(self.theta)
        if self.xion_range >= 1.0:
            raise ValueError("Ion range bigger than sample len")

        self.xion_indexes = np.sum(self.xy_spacing < self.xion_range)

        self.ysigma_u = kwargs.get('ysigma', 12)  # [A]
        self.ysigma = self.ysigma_u*self.dist_scale

        self.yion_range = self.ysigma*3
        if self.yion_range >= 0.5:
            raise ValueError("Ion range y bigger than sample len")
        print "Ion range: x={}, y={}".format(self.xion_range/self.dist_scale, self.yion_range/self.dist_scale)

        self.yion_indexes = np.sum(self.xy_spacing < self.yion_range)
        self.yion_last = self.yion_indexes-1

        self.x_gauss = np.exp(
            -np.power((self.xy_spacing[:self.xion_indexes]-self.shift)
                     *(1.0/np.sin(self.theta)), 2)
            / (2.0*np.power(self.xsigma, 2)))
        print self.xy_spacing[:self.xion_indexes]
        print self.shift, self.xsigma ,np.sin(self.theta)
        plt.plot(self.x_gauss)
        plt.show()
        self.z_const_diff = self.xy_spacing[:self.xion_indexes]/np.tan(self.theta)

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
        self.diffusion = self.diffusion_u*np.power(self.dist_scale,2)
        self.diffusion = self.diffusion/np.power(self.xy_d, 2)
        print "DIFFUSION: {}".format(self.diffusion)

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

        ax.plot_surface(self.X, self.Y, self.Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        plt.show()

    def show_history(self, mode='img'):
        #fig, ax = plt.subplots()
        #plt.subplots_adjust(bottom=0.25)
        fig = plt.figure()

        if mode == 'img':
            ax = fig.add_subplot(111)
            axslider = plt.axes([0.15, 0.1, 0.65, 0.05])
            slider = Slider(axslider, "Tmp", 0, len(self.Z_history)-1, valinit=0)

            ax.imshow(self.Z_history[0])

            def update(val):
                ax.clear()
                ax.imshow(self.Z_history[int(val)])
        else:
            ax = fig.add_subplot(111, projection='3d')
            axslider = plt.axes([0.15, 0.1, 0.65, 0.05])
            slider = Slider(axslider, "Tmp", 0, len(self.Z_history)-1, valinit=0)

            ax.plot_surface(self.X, self.Y, self.Z_history[0], cmap=cm.coolwarm, linewidth=0, antialiased=False)
            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')
            #ax.plot_wireframe(self.X, self.Y, self.Z_history[0])

            def update(val):
                ax.clear()
                ax.set_xlabel('X axis')
                ax.set_ylabel('Y axis')
                ax.plot_surface(self.X/self.dist_scale, self.Y/self.dist_scale, self.Z_history[int(val)]/self.dist_scale, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        slider.on_changed(update)
        plt.show()

    def run_single(self, show=False):
        if self.Z_initialized == False:
            self.Z_initialized = True
            self.Z_history.append(self.Z.copy())

        # ion erosion
        Z_pad = np.pad(self.Z, [[(self.yion_indexes-1)*2, (self.yion_indexes-1)*2], [self.xion_indexes-1, self.xion_indexes-1]], 'wrap')
        Z_erosion = np.zeros(Z_pad.shape, dtype=float)
        random_noise = np.abs(np.random.normal(0, 1, Z_erosion.shape))
        # print Z_pad.shape, self.xion_indexes, self.yion_indexes
        # loop over y dim
        #self.x_gauss
        for i_y in xrange((self.yion_indexes-1), self.nodes+3*(self.yion_indexes-1)):
            for i_x in xrange(0, self.nodes+self.xion_indexes-1):
                for j_y in xrange(-self.yion_indexes+1, self.yion_indexes):
                    z_diff = self.z_const_diff + (Z_pad[i_y+j_y,i_x:(i_x+self.xion_indexes)] - Z_pad[i_y,i_x])
                    Z_erosion[i_y+j_y][i_x:(i_x+self.xion_indexes)] += self.erosion*(
                    self.x_gauss*self.y_gauss[j_y+self.yion_last] ) * (
                        np.exp(-self.decay * z_diff)/(1.0+np.exp(-self.rms_slope*(z_diff-self.rms))))# * random_noise[i_y][i_x]
                    '''
                    if show:
                        # erosion result
                        plt.plot(self.xy_spacing[i_x:(i_x+self.xion_indexes)],
                                 Z_erosion[i_y+j_y][i_x:(i_x+self.xion_indexes)]/max(Z_erosion[i_y+j_y][i_x:(i_x+self.xion_indexes)]), 'ro')

                        # surface at the moment
                        plt.plot(self.xy_spacing[i_x:(i_x+self.xion_indexes)] ,
                                 Z_pad[i_y+j_y][i_x:(i_x+self.xion_indexes)]/max(np.abs(Z_pad[i_y+j_y][i_x:(i_x+self.xion_indexes)])), 'g')
                        # diff
                        plt.plot(self.xy_spacing[i_x:(i_x+self.xion_indexes)], z_diff/max(np.abs(Z_pad[i_y+j_y][i_x:(i_x+self.xion_indexes)])), 'b')
                        # trajectory
                        plt.plot(self.xy_spacing[i_x:(i_x+self.xion_indexes)], self.z_const_diff/max(np.abs(Z_pad[i_y+j_y][i_x:(i_x+self.xion_indexes)])), 'r')

                        # surface gauss energy
                        plt.plot(self.xy_spacing[i_x:(i_x+self.xion_indexes)],
                                 self.x_gauss/max(self.x_gauss), 'y')

                        plt.show()
                    '''

                '''
                plt.imshow(Z_pad)
                plt.show()
                plt.imshow(Z_erosion)
                plt.show()
                '''

            plt.imshow(Z_erosion[0:2*self.yion_indexes])
            plt.show()
                #
            #
        #
        if show:
            ax = plt.subplot2grid((1,2), (0, 0))
            ax.imshow(Z_pad[2*(self.yion_indexes-1):self.nodes+2*(self.yion_indexes-1), self.xion_indexes-1:self.nodes+self.xion_indexes-1])

            ax2 = plt.subplot2grid((1,2), (0,1))
            ax2.imshow(Z_erosion [2*(self.yion_indexes-1):self.nodes+2*(self.yion_indexes-1), self.xion_indexes-1:self.nodes+self.xion_indexes-1])
            plt.show()

        self.Z -= Z_erosion[2*(self.yion_indexes-1):self.nodes+2*(self.yion_indexes-1), self.xion_indexes-1:self.nodes+self.xion_indexes-1]

        # diffusion part
        # along x axis
        Z_pad = np.pad(self.Z, 1, 'wrap')
        x_diff = np.diff(Z_pad, 2, 1)[1:-1]
        y_diff = np.diff(Z_pad, 2, 0)[:,1:-1]

        '''
        ax = plt.subplot2grid((2,2), (0,0))
        ax.imshow(Z_pad[:3,:3])

        ax1 = plt.subplot2grid((2,2), (0,1))
        ax1.imshow(Z_pad[:3,-3:])

        ax2 = plt.subplot2grid((2,2), (1,0))
        ax2.imshow(Z_pad[-3:,:3])

        ax3 = plt.subplot2grid((2,2), (1,1))
        ax3.imshow(Z_pad[-3:,-3:])

        plt.show()
        '''

        self.Z += self.diffusion*(x_diff + y_diff)

        self.Z_history.append(self.Z.copy())

model = SurfEnergyDepositionModel(diffusion=01.1100, nodes=60, erosion=0.0010, theta=85,sample_len=300, ysigma=8, rms=1.1)
#model.add_sin(0.1,1,6)
model.add_sin(1,0,1)
#model.add_sin(0.5,0,7)

#model.show()

#model.add_gauss(1.00, 200, 20, 200, 20)

import time

t = time.time()
w = int(raw_input("run for:"))
while w>0:
    for i in range(w):
        print "{}  ({} s)".format(i, time.time()-t)
        t = time.time()
        model.run_single()
        if w == 1:
            model.run_single(True)
    #model.erosion = 0
    model.show_history()#mode='surf')
    model.show_history(mode='surf')
    w = int(raw_input("run for:"))

