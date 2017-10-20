import sys
import numpy as np
import matplotlib.pyplot as plt
import itertools
from mpl_toolkits.mplot3d import axes3d
from matplotlib.widgets import Slider
from matplotlib import cm

import warnings
warnings.filterwarnings('error')
warnings.filterwarnings('ignore')

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


        self.y -= self.erosion * yamamura(angles,self.theta_opt, self.f)*np.abs(np.random.normal(1,0.4,self.nodes_num)) #*self.beam_gauss
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
        self.x_center = self.X - self.sample_len*0.5
        self.Y = self.X.T
        self.y_center = self.Y - self.sample_len*0.5
        self.Z = np.zeros((self.nodes_num, self.nodes_num), dtype=float)

        self.f = kwargs.get('f', 2.4)

        self.theta = kwargs.get('theta', 60.0)
        self.theta = np.radians(self.theta)
        self.sample_slope = np.tan(-self.theta)
        self.sample_slope_xy= -np.sqrt(0.5*(1.0/np.power(np.cos(self.theta),2)-1.0))

        #self.Z += self.X*self.sample_slope
        self.Z += self.X*self.sample_slope_xy + self.Y*self.sample_slope_xy
        # set ave Z on zero
        #self.Z -= np.average(self.Z)
        self.start_ave_surf = np.average(self.Z)

        self.slope_background = self.X*self.sample_slope_xy + self.Y*self.sample_slope_xy

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
        self.diff_cycles = kwargs.get('diff_cycles', 10)
        self.diff_correction = kwargs.get("diff_correction", True)
        print("diff_correction:{}".format(self.diff_correction))
        self.moment = kwargs.get('moment', 1.0)

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
        #plt.plot(self.conv_fun, 'r+')
        #plt.show()

        self.angles_x = np.empty(self.Z.shape, dtype=float)
        self.angles_y = np.empty(self.Z.shape, dtype=float)
        self.Z_history = []

    def leveled_xyz(self, Z_, n=0):
        # extend X Y Z
        z_mean = np.mean(Z_)
        z_eroded = self.start_ave_surf - z_mean
        print("Eroded_sample: {} ({})".format(np.cos(self.theta)*z_eroded, z_eroded))

        roll_xy = int(np.round((np.sin(self.theta)*z_eroded/np.sqrt(2))/self.dx))
        roll_xy_rest = (np.sin(self.theta)*z_eroded/np.sqrt(2))/self.dx - np.round((np.sin(self.theta)*z_eroded/np.sqrt(2))/self.dx)
        print("Roll: {}\nRoll rest: {}".format(roll_xy, roll_xy_rest))
        Z_normalized = Z_ - (self.slope_background - np.mean(self.slope_background)) - z_mean
        Z_normalized = np.roll(Z_normalized, (roll_xy, roll_xy), axis=(0,1))
        Z_normalized = np.pad(Z_normalized, (0, self.nodes_num*n), mode='wrap')

        xy_spacing = np.linspace(0, self.sample_len*(n+1), self.nodes_num*(n+1), endpoint=False) - self.sample_len*0.5*(n+1) + roll_xy_rest
        x_center = np.tile(xy_spacing, (self.nodes_num*(n+1), 1))
        print("Xcenter:{}".format(np.mean(x_center)))
        y_center = x_center.T

        Z_ = Z_normalized + self.sample_slope_xy*x_center + self.sample_slope_xy*y_center

        # rotate to xy plane
        xyz = np.stack((x_center, y_center, Z_), axis=-1)
        rot_matrix = rotation_matrix(np.array([-1,1,0], dtype=float), self.theta)
        xyz_rot = np.tensordot(xyz, rot_matrix, axes=([2], [0]))
        # rotate around z axis
        rot_matrix = rotation_matrix(np.array([0,0,1], dtype=float), np.pi*0.25)
        xyz_rot = np.tensordot(xyz_rot, rot_matrix, axes=([2], [0]))

        xyz_unstacked = np.split(xyz_rot, 3, axis=-1)

        X_ = np.squeeze(xyz_unstacked[0])
        Y_ = np.squeeze(xyz_unstacked[1])
        Z_ = np.squeeze(xyz_unstacked[2])

        return X_, Y_, Z_

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

        conv_slopes_x = np.tan(self.angles_x)
        conv_slopes_y = np.tan(self.angles_y.T)

        thetas = np.arccos(1.0/np.sqrt(np.power(conv_slopes_x, 2) + np.power(conv_slopes_y, 2) + 1.0))

        self.Z -= np.power(self.dx, 2) * self.erosion * yamamura(thetas,self.theta_opt, self.f)*np.abs(np.random.normal(1,0.5,(self.nodes_num, self.nodes_num))) #*self.beam_gauss
        #self.Z -= self.erosion * yamamura(self.angles_y,self.theta_opt, self.f)*np.abs(np.random.normal(1,0.3,(self.nodes_num, self.nodes_num))) #*self.beam_gauss


        # erosion
        omegas = np.arctan2(conv_slopes_y, conv_slopes_x)
        omegas = np.abs(omegas)
        omegas[omegas >= np.pi*0.5] = np.pi - omegas[omegas >= np.pi*0.5]

        x_back_mask = conv_slopes_x > 0.0
        x_for_mask = np.logical_not(x_back_mask)

        y_back_mask = conv_slopes_y > 0.0
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
            plt.subplot(222).set_title("Normalized Z")
            plt.imshow(self.Z - self.X*self.sample_slope_xy - self.Y*self.sample_slope_xy)
            print(self.sample_slope_xy)
            #plt.imshow(summary)

            plt.subplot(223).set_title("Local Slope x")
            plt.imshow(l_slopes_x)
            plt.subplot(224).set_title("Conv Slope y")
            plt.imshow(conv_slopes_x)
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
            for i in range(self.diff_cycles):
                node_angles_x = np.roll(l_angles_x, (1, 0), axis=(1, 0)) - l_angles_x
                node_energy_x = np.tan((node_angles_x)/2.0)

                if self.diff_correction:
                    forward_transport_x = (np.roll(node_energy_x, (-1, 0), axis=(1, 0)) - node_energy_x)/np.cos(l_angles_x)
                else:
                    forward_transport_x = np.roll(node_energy_x, (-1, 0), axis=(1, 0)) - node_energy_x

                backward_transport_x = -np.roll(forward_transport_x, (1, 0), axis=(1, 0))


                node_angles_y = np.roll(l_angles_y, (0, 1), axis=(1, 0)) - l_angles_y
                node_energy_y = np.tan((node_angles_y)/2.0)

                if self.diff_correction:
                    forward_transport_y = (np.roll(node_energy_y, (0 -1), axis=(1, 0)) - node_energy_y)/np.cos(l_angles_y)
                else:
                    forward_transport_y = np.roll(node_energy_y, (0 -1), axis=(1, 0)) - node_energy_y
                backward_transport_y = -np.roll(forward_transport_y, (0, 1), axis=(1, 0))

                total_transport = forward_transport_x + backward_transport_x + forward_transport_y + backward_transport_y
                self.Z += self.diffusion*total_transport

        else:
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

    def show_history(self, n=0):
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
            #ax.plot_surface(self.X, self.Y, self.Z_history[int(val)], cmap=cm.coolwarm, linewidth=0, antialiased=False)
            X_, Y_, Z_ = self.leveled_xyz(self.Z_history[int(val)].copy(), n)
            ax.plot_surface(X_, Y_,Z_-np.mean(Z_), cmap=cm.afmhot, linewidth=0, antialiased=False)
        slider.on_changed(update)
        plt.show()

    def show_history_1d(self, aspect=1, rotate=True):
        self.x_diag = np.linspace(0, np.sqrt(2)*self.sample_len, self.nodes_num, endpoint=False)-self.sample_len*0.5*np.sqrt(2)

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
            ax.clear()

            z_ = self.Z_history[int(val)].diagonal().copy()
            eroded = -np.average(z_)
            print(eroded)
            z_ += eroded
            z_ -= self.slope_background.diagonal()


            roll_xy = int(np.round((np.sin(self.theta)*eroded)/self.dx))
            z_ = np.roll(z_, roll_xy)
            z_ += self.slope_background.diagonal()

            xz = np.array([self.x_diag, z_])
            xz = np.dot(rot_matrix, xz)
            x_ = xz[0]
            z_ = xz[1]

            ax.plot(x_, z_)
            #y_ = np.roll(y_, -roll_xy)

            #plot.set_ydata(xy[1])
            #ax.plot(self.x_diag, after_roll_back)
            #ax.plot(self.x_diag, z_)

        slider.on_changed(update)
        plt.show()

    def show_yam(self):
        plt.plot(np.linspace(0,90), yamamura(np.linspace(0, np.pi/2.0), self.theta_opt, self.f))
        plt.show()

    def write_xyz(self, file_name):
        ofile = open(file_name, 'w')
        ofile.write("{}\n{}\n".format(self.nodes_num**2, self.nodes_num**2))

        rot_matrix = rotation_matrix(np.array([-1,1,0], dtype=float), self.theta)
        Z_ = self.Z_history[-1].copy()
        eroded = -np.average(Z_)
        Z_ += eroded
        Z_ -= self.slope_background

        roll_xy = int(np.round((np.sin(self.theta)*eroded/np.sqrt(2))/self.dx))
        Z_ = np.roll(Z_, (roll_xy, roll_xy), axis=(1,0))
        Z_ += self.slope_background

        xyz = np.stack((self.x_center, self.y_center, Z_), axis=-1)
        xyz_rot = np.tensordot(xyz, rot_matrix, axes=([2], [0]))
        print(xyz_rot.shape)

        for i in range(self.nodes_num):
            for j in range(self.nodes_num):
                ofile.write("H {} {} {}\n".format(xyz_rot[i][j][0], xyz_rot[i][j][1], xyz_rot[i][j][2]))

        ofile.close()

    def get_img_boundary(self, X_, Y_, Z_):
        # 1 calc pane
        a = [None, None, None, None]
        b = [None, None, None, None]

        # slope = (y2-y1)/(x2-y1); b = y2 - x2*slope
        # corener array
        c_arr = ((0,0), (0,-1), (-1, -1), (-1, 0))
        for i in range(len(c_arr)):
            cid1, cid2 = c_arr[i-1]
            tid1, tid2 = c_arr[i]

            a[i] = (Y_[tid1, tid2]- Y_[cid1, cid2])/(X_[tid1, tid2] - X_[cid1, cid2])
            b[i] = Y_[tid1, tid2] - X_[tid1, tid2]*a[i]
        # check if a[0] ~ a[2] and a[1] ~ a[3]
        #a = [(a[0]+a[2])/2.0, (a[1]+a[3])/2.0]
        a = -1*np.around(np.array(a), 5)
        b = -1*np.around(np.array(b), 5)

        eq_left = np.array([[1-a[0], -1, 0, 1], [-a[1],0, 0, 1], [0, -a[0], 0, 1], [0, -a[1], 1, 0]], dtype=float)
        eq_right = np.array(b, dtype=float)
        x1, x2, y1, y2 = np.linalg.solve(eq_left, eq_right)
        boundary = np.round([[x1, x2], [y1, y2]])
        return boundary

    def show_img(self, num, bin_step=1, r=5, bin_rep=3, cell_rep=1, show=True, boundary=None):
        # bin_rep: 1, 2, 3, ....
        print("DISPL NUM::{}".format(num))
        X_, Y_, Z_ = self.leveled_xyz(self.Z_history[num], cell_rep)
        flat_ = [None, None, None]
        flat_[0] = X_.flatten()
        flat_[1] = Y_.flatten()
        flat_[2] = Z_.flatten()

        if boundary is None:
            boundary = self.get_img_boundary(X_, Y_, Z_)

        r2 = np.power(r,2)
        bin_centers = [np.arange(boundary[0,0], boundary[0,1]+0.001, bin_step),
                                np.arange(boundary[1,0], boundary[1,1]+0.001, bin_step)]

        bin_edges = np.array([bin_centers[0][:-1]+0.5*bin_step, bin_centers[1][:-1]+0.5*bin_step])

        indexes = [None, None]
        indexes[0] = np.digitize(flat_[0], bin_edges[0])
        indexes[1] = np.digitize(flat_[1], bin_edges[1])

        #Z_img = np.ones((bin_centers[0].shape[0], bin_centers[1].shape[0]), dtype=float)*-10000
        Z_holder = np.ones((bin_centers[1].shape[0], bin_centers[0].shape[0], 3, 3), dtype=float)*-1000

        for i in range(len(flat_[0])):
            ix = indexes[0][i]
            iy = indexes[1][i]
            cr_z = flat_[2][i]
            if(Z_holder[iy,ix,0,2] < cr_z):
                Z_holder[iy,ix,2] = Z_holder[iy,ix,1]
                Z_holder[iy,ix,1] = Z_holder[iy,ix,0]
                Z_holder[iy,ix,0] = [flat_[0][i], flat_[1][i], cr_z]

            elif(Z_holder[iy,ix,1,2] < cr_z):
                Z_holder[iy,ix,2] = Z_holder[iy,ix,1]
                Z_holder[iy,ix,1] = [flat_[0][i], flat_[1][i], cr_z]

            elif(Z_holder[iy,ix,2,2] < cr_z):
                Z_holder[iy,ix,2] = [flat_[0][i], flat_[1][i], cr_z]


        # discard edges
        Z_holder = Z_holder[1:-1, 1:-1, :, :]
        probe_bins = (bin_rep*2+1)*(bin_rep*2+1)
        probe_bins_total = probe_bins*3
        Z_full_holder = np.empty( (Z_holder.shape[0], Z_holder.shape[1], probe_bins_total, 3) )

        bin_rep_arr = list(range(-bin_rep, bin_rep+1))
        for n_, roll_ in zip(range(probe_bins), itertools.product(bin_rep_arr, bin_rep_arr)):
            #print(roll_, n_*3, (n_+1)*3)
            Z_full_holder[:,:,n_*3:(n_+1)*3] = np.roll(Z_holder, (roll_[0], roll_[1]), axis=(0,1))
        Z_full_holder = Z_full_holder[1:-1, 1:-1, :, :]

        bin_centers[0] = bin_centers[0][2:-2]
        bin_centers[1] = bin_centers[1][2:-2]

        probe = np.stack(np.meshgrid(bin_centers[0], bin_centers[1]), -1)
        probe = np.append(probe, np.zeros((probe.shape[0], probe.shape[1], 1)), axis=2)
        probe = np.tile(probe, probe_bins_total).reshape(probe.shape[0], probe.shape[1], probe_bins_total, 3)

        probe_diff = Z_full_holder - probe
        probe_height = np.sqrt(r2 - (np.power(probe_diff[:,:,:,0],2) + np.power(probe_diff[:,:,:,1],2)))
        probe_z = Z_full_holder[:,:,:,2] + probe_height

        probe_z[np.isnan(probe_z)] = -1000
        probe_z = np.amax(probe_z, -1)

        probe_z[probe_z < -999] = 0.0

        if show:
            #plt.imshow(probe_z, cmap=cm.afmhot)
            plt.gray()
            plt.imshow(probe_z) #, cmap=cm.afmhot)
            plt.show()
        else:
            return probe_z

        #indexed_si = np.digitize(self.start_all_z_list[0]- self.ave_surf, self.dp_bin_edges)
        #[np.linspace(boundary[0,0], boundary[0,1], 1.0)

    def write_img(self, out_dir, period=10, bin_step=1, r=5, bin_rep=3, cell_rep=1, boundary=None):
        if boundary is None:
            eX_, eY_, eZ_ = self.leveled_xyz(self.Z_history[-1], cell_rep)
            boundary = self.get_img_boundary(eX_, eY_, eZ_)

        print("{}, {}, {}".format(0, len(self.Z_history)-1, period))
        for i in range(0, len(self.Z_history)-1, period):

            data = self.show_img(i, bin_step=bin_step, r=r, bin_rep=bin_rep, cell_rep=cell_rep, boundary=boundary, show=False)
            name = "rip.{}.png".format(str(int(i/period)).zfill(5))
            plt.gray()
            plt.imsave("{}/{}".format(out_dir,name), data)#, cmap=cm.afmhot)


"""
m2 = model2d(theta=30, moment=0.07, erosion=0.07, diffusion=0.01, sample_len=100, nodes_num=100, conv_sigma=0.3, diff_cycles=10) #, f=0.3, theta_opt=10)
ripples and holes
"""

#m2 = model2d(theta=60, moment=0.00, erosion=0.04, diffusion=0.06, sample_len=200, nodes_num=200, conv_sigma=7)
#m2 = model2d(theta=60, moment=0.050, erosion=0.025, diffusion=0.225, sample_len=200, nodes_num=200, conv_sigma=10)
#m2 = model2d(theta=30, moment=0.04, erosion=0.009, diffusion=0.02, sample_len=100, nodes_num=100, conv_sigma=0.3, diff_cycles=4, diff_correction=True) #, f=0.3, theta_opt=10)
#m2 = model2d(theta=60, moment=0.04, erosion=0.018, diffusion=0.02, sample_len=100, nodes_num=100, conv_sigma=0.3, diff_cycles=4, diff_correction=True) #, f=0.3, theta_opt=10)
m2 = model2d(theta=55, moment=0.02, erosion=0.01, diffusion=0.03, sample_len=200, nodes_num=200, conv_sigma=0.3, diff_cycles=5, diff_correction=True) #, f=0.3, theta_opt=10)

import time

run_next = int(input("continue:"))
while run_next != 0:
    for j in range(run_next):
        t = time.time()
        for i in range(10):
            m2.single_step()
        single_loop = time.time()-t
        print("single: {} s \nelepse: {} s  ({} min)".format(single_loop, (run_next-j)*single_loop, (run_next-j)*single_loop/60.0))

    m2.show_history_1d(aspect=30)
    m2.show_history_1d(aspect=5)
    m2.show_history_1d(aspect=1)
    #m2.show_history_1d()
    m2.show_history(n=1)
    #m2.single_step(look_up=True)
    #num = int(input("1: {}\nnum:".format(len(m2.Z_history))))
    #m2.write_xyz("/tmp/t.xyz")
    m2.show_img(len(m2.Z_history)-1) #, bin_step=1, r=5, bin_rep=3, cell_rep=1)
    run_next = int(input("continue:"))
m2.write_img("/tmp/playground",period=100, r=9 )
