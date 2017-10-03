import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib.widgets import Slider
from matplotlib import cm

import warnings
warnings.filterwarnings('error')

def yamamura(theta, theta_opt, f):
    return np.power(np.cos(theta), -f)*np.exp(f * (1-np.power(np.cos(theta), -1)) * np.cos(theta_opt)  )

def gauss2d(x, x0, xs, y, y0, ys):
    return np.exp(-( (np.power(x-x0,2)/(2*np.power(xs,2))) +
                        (np.power(y-y0,2)/(2*np.power(ys,2))) ))

def rotation_matrix(r_vec, ra):
    r_vec = np.array(r_vec)
    r_vec = r_vec/np.linalg.norm(r_vec)
    rmatrix = [[np.cos(ra) + r_vec[0]*r_vec[0]*(1-np.cos(ra)),
                r_vec[0]*r_vec[1]*(1-np.cos(ra)) - r_vec[2]*np.sin(ra),
                r_vec[0]*r_vec[2]*(1-np.cos(ra)) + r_vec[1]*np.sin(ra)],
               [r_vec[1]*r_vec[0]*(1-np.cos(ra)) + r_vec[2]*np.sin(ra),
                np.cos(ra) + r_vec[1]*r_vec[1]*(1-np.cos(ra)),
                r_vec[1]*r_vec[2]*(1-np.cos(ra)) - r_vec[0]*np.sin(ra)],
               [r_vec[2]*r_vec[0]*(1-np.cos(ra)) - r_vec[1]*np.sin(ra),
                r_vec[2]*r_vec[1]*(1-np.cos(ra)) + r_vec[0]*np.sin(ra),
                np.cos(ra) + r_vec[2]*r_vec[2]*(1-np.cos(ra))]]

    return np.array(rmatrix)


class model:
    def __init__(self, **kwargs):
        self.counter = 0
        self.sample_len = kwargs.get('sample_len', 500)
        self.nodes_num = kwargs.get('nodes_num', 500)

        self.f = kwargs.get('f', 2.4)

        self.theta = kwargs.get('theta', 60.0)
        self.theta = np.radians(self.theta)
        self.sample_slope = np.tan(-self.theta)
        self.slope_corr = self.sample_len*self.sample_slope

        self.theta_opt = kwargs.get('theta_opt', 60.0)
        self.theta_opt = np.radians(self.theta_opt)

        self.erosion = kwargs.get('erosion', 1.0)
        self.diffusion = kwargs.get('diffusion', 0.0001)
        self.moment = kwargs.get('moment', 1.0)

        self.x = np.linspace(0, self.sample_len, self.nodes_num, endpoint=False)
        self.dx = self.x[1]
        self.y = np.zeros(self.nodes_num, dtype=float)

        self.y += self.x*self.sample_slope

        self.conv_sigma = kwargs.get('conv_sigma', 10)
        self.conv_multi = kwargs.get('conv_multi', 2.0)

        x_ = self.x[self.x <= self.conv_sigma*self.conv_multi*2.0]
        if len(x_)%2 != 0:
            x_ = self.x[:len(x_)+1]

        conv_center = x_[-1]*0.5
        self.wrap_len = int(len(x_)/2)
        self.conv_fun = np.exp(-np.power(x_-conv_center, 2)/(2*np.power(self.conv_sigma, 2)))
        self.conv_fun = self.conv_fun/np.sum(self.conv_fun)

        self.gauss_random = np.pad(self.conv_fun, (0, self.nodes_num-len(self.conv_fun)), 'constant' )
        self.beam_gauss = np.exp(-np.power(self.x-self.sample_len/2.0, 2)/(2*np.power(self.sample_len/6.0, 2)))

        self.y_history = []
        self.y_history.append(self.y.copy())

    def run(self, run_steps):
        for i in range(run_steps):
            self.single_step()

    def single_step(self):
        #wrap_angles = np.pad(l_angles, (wrap_len, wrap_len-1), mode='wrap')
        #angles = np.convolve(wrap_angles, conv_fun, mode='valid')

        l_y = np.pad(self.y, (0, 1), mode='wrap')
        l_y[-1] += self.slope_corr

        l_slopes = np.diff(l_y, 1)/self.dx
        l_angles = np.arctan(l_slopes)
        #*np.abs(np.random.normal(1,0.4,self.nodes_num))

        wrap_angles = np.pad(l_angles, (self.wrap_len, self.wrap_len-1), mode='wrap')
        angles = np.convolve(wrap_angles, self.conv_fun, mode='valid')
        #plt.plot(np.degrees(angles))
        #plt.show()


        self.y -= self.erosion * yamamura(angles,self.theta_opt, self.f)*np.abs(np.random.normal(1,0.3,self.nodes_num)) #*self.beam_gauss
        #*np.roll(self.gauss_random, np.random.randint(self.nodes_num))h
        #plt.plot(yamamura(angles,self.theta_opt, self.f))
        #plt.show()


        # - (1 - cos(angl)) + (1 - cos(angl[+1]))
        cos_res = np.cos(4.0*angles)
        #self.y += self.moment*(np.cos(angles) - np.cos(np.roll(angles, 1)))
        #moment = self.moment*(cos_res - np.roll(cos_res, 1))

        removal = (1.0 - cos_res) # positive allways
        forward_list = angles <= 0
        forward_gain = removal.copy()
        backward_gain = removal.copy()

        forward_gain[np.logical_not(forward_list)] = 0.0
        backward_gain[forward_list] = 0.0

        self.y += self.moment*(np.roll(forward_gain, 1) + np.roll(backward_gain, -1) - removal)#*self.beam_gauss


        self.y_history.append(self.y.copy())

        l_y = np.pad(self.y, (1, 1), mode='wrap')
        l_y[-1] += self.slope_corr
        l_y[0] += -self.slope_corr

        try:
            self.y += self.diffusion*np.diff(l_y, 2)
        except:
            self.diffusion*np.diff(l_y, 2)
            sys.exit(1)

        # sample slope correction
        #l_angles = np.arc


    def show(self, rotate=True, aspect=1):
        self.rotate = rotate
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)

        ax.set_aspect(aspect)

        if rotate:
            rot_matrix = np.array([[np.cos(self.theta), - np.sin(self.theta)], [np.sin(self.theta), np.cos(self.theta)]])
            xy = np.array([self.x, self.y_history[-1]])
            xy = np.dot(rot_matrix, xy)
            xy[0] = xy[0] - np.mean(xy[0])
            xy[1] = xy[1] - np.mean(xy[1])

            plot, = ax.plot(xy[0], xy[1])
        else:
            plot, = ax.plot(self.x, self.y_history[-1]-np.mean(self.y_history[-1]))

        axslider = plt.axes([0.15, 0.1, 0.65, 0.05])
        slider = Slider(axslider, 'Tmp', 0, len(self.y_history)-1, valinit=1)


        def update(val):
            selection = self.y_history[int(val)]
            if self.rotate:
                xy = np.array([self.x, selection])
                xy = np.dot(rot_matrix, xy)
                xy[0] = xy[0] - np.mean(xy[0])
                xy[1] = xy[1] - np.mean(xy[1])

                #plot.set_ydata(xy[1])
                ax.clear()
                ax.plot(xy[0], xy[1])
            else:
                traveled=np.mean(selection)
                plot.set_ydata(selection -traveled)
                print("Traveled:{}".format(traveled))

        slider.on_changed(update)
        plt.show()


    def show_rotated(self):
        rot_matrix = np.array([[np.cos(self.theta), - np.sin(self.theta)], [np.sin(self.theta), np.cos(self.theta)]])
        print(np.degrees(self.theta), np.degrees(self.theta))

        xy = np.array([self.x, self.y_history[-1]])

        xy = np.dot(rot_matrix, xy)
        plt.plot(xy[0], xy[1])
        plt.show()



    def add_sin(self, amp, n):
        self.y += amp*np.sin(n*2*np.pi*self.x/self.sample_len)


    def show_yam(self):
        plt.plot(np.linspace(-90,90), yamamura(np.linspace(-np.pi/2.0, np.pi/2.0), self.theta_opt, self.f))
        plt.show()


class model2d:
    def __init__(self, **kwargs):
        self.sample_len = kwargs.get('sample_len', 200)
        self.nodes_num = kwargs.get('nodes_num', 200)

        self.xy_spacing = np.linspace(0, self.sample_len, self.nodes_num, endpoint=False)
        self.dx = self.xy_spacing[1]
        self.X = np.tile(self.xy_spacing, (self.nodes_num, 1))
        self.Y = self.X.T
        self.Z = np.zeros((self.nodes_num, self.nodes_num), dtype=float)

        self.f = kwargs.get('f', 2.4)

        self.theta = kwargs.get('theta', 60.0)
        self.theta = np.radians(self.theta)
        self.sample_slope = np.tan(-self.theta)
        self.sample_slope_xy= -np.sqrt(0.5*(1.0/np.power(np.cos(self.theta),2)-1.0))

        #self.Z += self.X*self.sample_slope
        self.Z += self.X*self.sample_slope_xy + self.Y*self.sample_slope_xy

        self.slope_corr = self.sample_len*self.sample_slope

        self.slope_corr_diff1 = np.zeros((self.nodes_num+1, self.nodes_num+1), dtype=float)
        self.slope_corr_diff1[-1,:] = self.sample_len*self.sample_slope_xy
        self.slope_corr_diff1[:,-1] = self.sample_len*self.sample_slope_xy

        self.slope_corr_diff2 = np.zeros((self.nodes_num+2, self.nodes_num+2), dtype=float)
        self.slope_corr_diff2[0,:] = -self.sample_len*self.sample_slope_xy
        self.slope_corr_diff2[:,0] = -self.sample_len*self.sample_slope_xy
        self.slope_corr_diff2[-1,:] = self.sample_len*self.sample_slope_xy
        self.slope_corr_diff2[:,-1] = self.sample_len*self.sample_slope_xy

        self.theta_opt = kwargs.get('theta_opt', 74.0)
        self.theta_opt = np.radians(self.theta_opt)

        self.erosion = kwargs.get('erosion', 1.0)
        self.diffusion = kwargs.get('diffusion', 0.0001)
        self.moment = kwargs.get('moment', 1.0)
        print(self.moment)

        self.conv_sigma = kwargs.get('conv_sigma', 10)
        self.conv_multi = kwargs.get('conv_multi', 2.0)

        x_ = self.xy_spacing[self.xy_spacing <= self.conv_sigma*self.conv_multi*2.0]
        if len(x_)%2 != 0:
            x_ = self.xy_spacing[:len(x_)+1]

        conv_center = x_[-1]*0.5
        self.wrap_len = int(len(x_)/2)
        self.conv_fun = np.exp(-np.power(x_-conv_center, 2)/(2*np.power(self.conv_sigma, 2)))
        self.conv_fun = self.conv_fun/np.sum(self.conv_fun)
        print(self.conv_fun)
        print(np.sum(self.conv_fun))
        #plt.plot(self.conv_fun, 'r+')
        #plt.show()

        self.angles_x = np.empty(self.Z.shape, dtype=float)
        self.angles_y = np.empty(self.Z.shape, dtype=float)
        self.Z_history = []

    def single_step(self, look_up=False):
        self.Z_history.append(self.Z.copy())
        l_z = np.pad(self.Z, ((0, 1), (0,1)), mode='wrap')
        l_z += self.slope_corr_diff1
        #l_z[:,-1] += self.slope_corr

        l_slopes_x = np.diff(l_z, 1, axis=1)[:-1]/self.dx
        l_slopes_y = np.diff(l_z, 1, axis=0)[:,:-1]/self.dx

        l_angles_x = np.arctan(l_slopes_x)
        #print(np.degrees(l_angles_x))
        l_angles_y = np.arctan(l_slopes_y)
        #print(np.degrees(l_angles_y))

        wrap_angles_x = np.pad(l_angles_x, (self.wrap_len, self.wrap_len-1), mode='wrap')
        #wrap_angles_x = np.zeros(wrap_angles_x.shape, dtype=float)
        # ! Transpose y before convolution (and after) !
        wrap_angles_y = np.pad(l_angles_y, (self.wrap_len, self.wrap_len-1), mode='wrap').T
        #wrap_angles_y =np.zeros(wrap_angles_y.shape, dtype=float) #np.pad(l_angles_y, (self.wrap_len, self.wrap_len-1), mode='wrap').T
        #angles_x = np.convolve(wrap_angles_x, self.conv_fun, mode='valid')
        #angles_y = np.convolve(wrap_angles_y, self.conv_fun, mode='valid')
        for i in range(self.nodes_num):
            self.angles_x[i] = np.convolve(wrap_angles_x[i], self.conv_fun, mode='valid')
            self.angles_y[i] = np.convolve(wrap_angles_y[i], self.conv_fun, mode='valid')

        l_slopes_x = np.tan(self.angles_x)
        l_slopes_y = np.tan(self.angles_y.T)

        thetas = np.arccos(1/np.sqrt(np.power(l_slopes_x, 2) + np.power(l_slopes_y, 2) + 1))

        self.Z -= np.power(self.dx, 2) * self.erosion * yamamura(thetas,self.theta_opt, self.f)*np.abs(np.random.normal(1,0.3,(self.nodes_num, self.nodes_num))) #*self.beam_gauss
        #self.Z -= self.erosion * yamamura(self.angles_y,self.theta_opt, self.f)*np.abs(np.random.normal(1,0.3,(self.nodes_num, self.nodes_num))) #*self.beam_gauss


        # erosion
        omegas = np.arctan2(l_slopes_y, l_slopes_x)
        omegas = np.abs(omegas)
        omegas[omegas >= np.pi*0.5] = np.pi - omegas[omegas >= np.pi*0.5]

        x_back_mask = l_slopes_x > 0.0
        x_for_mask = np.logical_not(x_back_mask)

        y_back_mask = l_slopes_y > 0.0
        y_for_mask = np.logical_not(y_back_mask)

        ero_00 = (1.0-np.cos(4.0*thetas))*self.moment/np.power(self.dx, 3)
        sin_omega = np.sin(omegas)
        cos_omega = np.cos(omegas)

        acc_00 = (1-sin_omega)*(1-cos_omega)*ero_00
        acc_01 = cos_omega*(1-sin_omega)*ero_00
        acc_10 = (1-cos_omega)*sin_omega*ero_00
        acc_11 = sin_omega*cos_omega*ero_00

        # lets roll
        acc_01 = np.roll(x_for_mask*acc_01, (1, 0), axis=(1, 0)) \
            + np.roll(x_back_mask*acc_01, (-1, 0), axis=(1,0))

        acc_10 = np.roll(y_for_mask*acc_10, (0, 1), axis=(1,0)) \
            + np.roll(y_back_mask*acc_10, (0, -1), axis=(1,0))

        """
        (-1, -1) | (-1, 0) | (-1, 1)
        ----------------------------
        (0, -1)  |         | (0, 1)
        ----------------------------
        (1, -1)  | (1, 0)  | (1, 1)
        """

        acc_11 = np.roll(np.logical_and(x_for_mask, y_for_mask)*acc_11, (1, 1), axis=(1,0)) \
            + np.roll(np.logical_and(x_for_mask, y_back_mask)*acc_11, (1, -1), axis=(1,0)) \
            + np.roll(np.logical_and(x_back_mask, y_back_mask)*acc_11, (-1, -1), axis=(1,0)) \
            + np.roll(np.logical_and(x_back_mask, y_for_mask)*acc_11, (-1, 1), axis=(1,0))

        summary = -ero_00+acc_00+acc_01+acc_10+acc_11
        if look_up:
            fig = plt.figure()
            plt.subplot(221).set_title("Thetas")
            plt.imshow(np.degrees(thetas))
            plt.subplot(222).set_title("Moment_sum")
            plt.imshow(summary)

            plt.subplot(223).set_title("Normalized Z")
            plt.imshow(self.Z - self.X*self.sample_slope_xy - self.Y*self.sample_slope_xy)
            plt.subplot(224).set_title("Slope y")
            plt.imshow(l_slopes_y)
            plt.show()

        #print("{} {} {} {} {}".format(np.sum(summary), np.sum(acc_00), np.sum(acc_01), np.sum(acc_10), np.sum(acc_11)))
        self.Z += summary
        #self.y += self.moment*(np.cos(angles) - np.cos(np.roll(angles, 1)))
        #moment = self.moment*(cos_res - np.roll(cos_res, 1))

        if False:
            print(np.degrees(omegas))
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')

            ax.plot_surface(self.X, self.Y, np.degrees(omegas), cmap=cm.coolwarm, linewidth=0, antialiased=False)
            plt.show()

        if True:
            # ! new diffusion requied !
            Z_pad = np.pad(self.Z, 1, 'wrap')

            Z_pad += self.slope_corr_diff2
            #Z_pad[:, 0] += -self.slope_corr#-self.sample_slope*self.dx
            #Z_pad[:, -1] += self.slope_corr

            x_diff = np.diff(Z_pad, 2, 1)[1:-1]
            y_diff = np.diff(Z_pad, 2, 0)[:,1:-1]

            self.Z += self.diffusion*(x_diff + y_diff)/np.power(self.dx, 2)


    def add_sin(self, amp, nx, ny):
        self.Z += amp*np.sin(2*np.pi*(nx*self.X/self.sample_len + ny*self.Y/self.sample_len))

    def add_cos(self, amp, nx, ny):
        self.Z += amp*np.cos(2*np.pi*(nx*self.X/self.sample_len + ny*self.Y/self.sample_len))

    def show(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(self.X, self.Y, self.Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # ! rotate !
        # unrotate
        #ax.plot_surface(self.X, self.Y, self.Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ra = np.arctan(self.sample_slope_xy)
        rot_matrix_y = np.array([[np.cos(ra), 0, np.sin(ra)], [0, 1, 0], [-np.sin(ra), 0, np.cos(ra)]])
        xyz = np.stack((self.X, self.Y, self.Z), axis=-1)
        xyz_rot = np.tensordot(xyz, rot_matrix_y, axes=([2], [0]))
        xyz_unstacked = np.split(xyz_rot, 3, axis=-1)

        X_ = np.squeeze(xyz_unstacked[0])
        Y_ = np.squeeze(xyz_unstacked[1])
        Z_ = np.squeeze(xyz_unstacked[2])
        #plt.plot(self.X[0],self.Z[0])
        #plt.plot(self.X[0],self.X[0]*self.sample_slope+self.Z[0][0])
        wrap_len = 200
        look = np.pad(self.Z[0], (wrap_len), mode='wrap')
        look[:wrap_len] -= self.slope_corr
        look[-wrap_len:] += self.slope_corr

    def show_history(self):
        self.name_str = "theta={}, sput={}, moment={}, diff={}, conv={}".format(np.degrees(self.theta), self.erosion, self.moment, self.diffusion, self.conv_sigma)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        #ax.plot_surface(self.X, self.Y, self.Z_history[0], cmap=cm.coolwarm, linewidth=0, antialiased=False)
        axslider = plt.axes([0.15, 0.1, 0.65, 0.05])
        slider = Slider(axslider, "Tmp", 0, len(self.Z_history)-1, valinit=0)

        def update(val):
            ax.clear()
            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')
            ax.set_title(self.name_str)
            rot_matrix = rotation_matrix(np.array([-1,1,0], dtype=float), self.theta)

            xyz = np.stack((self.X, self.Y, self.Z_history[int(val)]), axis=-1)
            xyz_rot = np.tensordot(xyz, rot_matrix, axes=([2], [0]))
            #xyz_rot = np.tensordot(xyz_rot, rot_matrix_x, axes=([2], [0]))

            xyz_unstacked = np.split(xyz_rot, 3, axis=-1)

            X_ = np.squeeze(xyz_unstacked[0])
            Y_ = np.squeeze(xyz_unstacked[1])
            Z_ = np.squeeze(xyz_unstacked[2])

            #ax.plot_surface(self.X, self.Y, self.Z_history[int(val)], cmap=cm.coolwarm, linewidth=0, antialiased=False)
            ax.plot_surface(X_, Y_,Z_-np.mean(Z_), cmap=cm.afmhot, linewidth=0, antialiased=False)
        slider.on_changed(update)
        plt.show()

    def show_history_1d(self, aspect=1, rotate=True):
        self.x_diag = np.linspace(0, np.sqrt(2)*self.sample_len, self.nodes_num, endpoint=False)

        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)

        ax.set_aspect(aspect)

        if rotate:
            rtheta = self.theta
        else:
            rtheta = 0.0
        rot_matrix = np.array([[np.cos(rtheta), - np.sin(rtheta)], [np.sin(rtheta), np.cos(rtheta)]])
        xy = np.array([self.x_diag, self.Z_history[-1].diagonal()])
        xy = np.dot(rot_matrix, xy)
        xy[0] = xy[0] - np.mean(xy[0])
        xy[1] = xy[1] - np.mean(xy[1])

        plot, = ax.plot(xy[0], xy[1])

        axslider = plt.axes([0.15, 0.1, 0.65, 0.05])
        slider = Slider(axslider, 'Tmp', 0, len(self.Z_history)-1, valinit=1)


        def update(val):
            selection = self.Z_history[int(val)].diagonal()
            xy = np.array([self.x_diag, selection])
            xy = np.dot(rot_matrix, xy)
            xy[0] = xy[0] - np.mean(xy[0])
            xy[1] = xy[1] - np.mean(xy[1])

            #plot.set_ydata(xy[1])
            ax.clear()
            ax.plot(xy[0], xy[1])

        slider.on_changed(update)
        plt.show()



#m2 = model2d(theta=60, moment=0.00, erosion=0.04, diffusion=0.06, sample_len=200, nodes_num=200, conv_sigma=7)
#m2 = model2d(theta=60, moment=0.050, erosion=0.025, diffusion=0.225, sample_len=200, nodes_num=200, conv_sigma=10)
m2 = model2d(theta=60, moment=0.00, erosion=0.055, diffusion=0.225, sample_len=500, nodes_num=500, conv_sigma=7)
#m2.add_cos(10,3,3)
#m2.add_cos(10,3,-3)

import time

run_next = int(input("continue:"))
while run_next != 0:
    for j in range(run_next):
        t = time.time()
        for i in range(10):
            m2.single_step()
        single_loop = time.time()-t
        print("single: {} s \nelepse: {} s  ({} min)".format(single_loop, (run_next-j)*single_loop, (run_next-j)*single_loop/60.0))

    m2.show_history_1d(aspect=5)
    m2.show_history_1d()
    m2.show_history()
    m2.single_step(look_up=True)

    run_next = int(input("continue:"))
