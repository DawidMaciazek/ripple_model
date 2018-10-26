import time
import math
import numpy as np
import matplotlib.pyplot as plt
import itertools

from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.mplot3d import axes3d
from matplotlib.widgets import Slider
from matplotlib import cm
from matplotlib.colors import Normalize

import pickle

from matplotlib.tri.triangulation import Triangulation
from matplotlib.tri.triinterpolate import CubicTriInterpolator, LinearTriInterpolator

from scipy.interpolate import interp1d

import sys
import warnings
warnings.filterwarnings('error')
warnings.filterwarnings('ignore')

def save_figure(fileName,fig=None,**kwargs):
    ''' Save a Matplotlib figure as an image without borders or frames.
       Args:
            fileName (str): String that ends in .png etc.

            fig (Matplotlib figure instance): figure you want to save as the image
        Keyword Args:
            orig_size (tuple): width, height of the original image used to maintain
            aspect ratio.
    '''
    fig_size = fig.get_size_inches()
    w,h = fig_size[0], fig_size[1]
    fig.patch.set_alpha(0)
    '''
    if kwargs.has_key('orig_size'): # Aspect ratio scaling if required
        w,h = kwargs['orig_size']
        w2,h2 = fig_size[0],fig_size[1]
        fig.set_size_inches([(w2/w)*w,(w2/w)*h])
        fig.set_dpi((w2/w)*fig.get_dpi())
    '''
    a=fig.gca()
    a.set_frame_on(False)
    a.set_xticks([]); a.set_yticks([])
    plt.axis('off')
    plt.xlim(0,h); plt.ylim(w,0)
    fig.savefig(fileName, transparent=True, bbox_inches='tight', \
                        pad_inches=0)


# on fast edit
#
md_res = np.loadtxt("3000ar.txt")
sput_base = md_res[:,1]
redist_base = md_res[:,2]
rad_theta_base = np.radians(md_res[:,0])
test_x = np.arange(0,0.5*np.pi, 0.01)

sput_fit = interp1d(rad_theta_base, sput_base, kind="cubic")
redist_fit = interp1d(rad_theta_base, redist_base, kind="cubic")

plt.plot(np.degrees(rad_theta_base), sput_base)
plt.plot(np.degrees(test_x), sput_fit(test_x))
plt.show()

plt.plot(np.degrees(rad_theta_base), redist_base)
plt.plot(np.degrees(test_x), redist_fit(test_x))
plt.show()

def yamamura(theta, ytheta, f):
    #return np.power(np.cos(theta), -f)*np.exp(f * (1-np.power(np.cos(theta), -1)) * np.cos(ytheta)  )
    return sput_fit(theta)

def redist_fun(theta):
    return redist_fit(theta)

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

def condense_Z(Z, xy_spacing):
    size = len(Z)
    size_new = size*2-1

    xy_spacing_new = np.linspace(xy_spacing[0], xy_spacing[-1], size_new)
    Z_new = np.empty((size_new, size_new), dtype=float)

    # fill with previous vaues
    Z_new[::2, ::2] = Z

    # along y axis
    Z_new[1:-1:2, ::2] = (Z[1:, :] + Z[:-1, :])*0.5

    # along x axis
    Z_new[::2, 1:-1:2] = (Z[:, 1:] + Z[:, :-1])*0.5

    #         %           [0:-2:2, 1:-1:2]         %
    # [1:-1:2, 0:-2:2]            X         [1:-1:2, 2::2]
    #         %            [2::2, 1:-1:2]          %

    Z_new[1:-1:2, 1:-1:2] = (Z_new[0:-2:2, 1:-1:2] + Z_new[2::2, 1:-1:2] +
                             Z_new[1:-1:2, 0:-2:2] + Z_new[1:-1:2, 2::2])*0.25

    return Z_new, xy_spacing_new

class model2d:
    '''
    TO DO LIST:
        >. Initial gaussian roughness
        >. Better image show class
    '''
    def __init__(self, **kwargs):
        self.sample_len = kwargs.get('sample_len', 100) # [ nm ]
        self.nodes_num = kwargs.get('nodes_num', 100)
        self.img_dx = kwargs.get('img_dx', 1)

        self.xy_spacing = np.linspace(0, self.sample_len, self.nodes_num, endpoint=False)
        self.dx = self.xy_spacing[1] # [ nm ]
        self.X = np.tile(self.xy_spacing, (self.nodes_num, 1))
        self.x_center = self.X - self.sample_len*0.5
        self.Y = self.X.T
        self.y_center = self.Y - self.sample_len*0.5
        self.Z = np.zeros((self.nodes_num, self.nodes_num), dtype=float)

        self.theta = kwargs.get('theta', 60.0)
        self.theta = np.radians(self.theta)
        self.sample_slope = np.tan(-self.theta)
        self.sample_slope_xy= -np.sqrt(0.5*(1.0/np.power(np.cos(self.theta),2)-1.0))

        self.Z += self.X*self.sample_slope_xy + self.Y*self.sample_slope_xy

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

        self.f = kwargs.get('f', 1.57)
        self.yamp = kwargs.get('yamp', 1.85) # [ atoms/projectile ] // Si
        self.ytheta = kwargs.get('ytheta', 78.0)
        self.ytheta = np.radians(self.ytheta)

        # amp*((1-cos(theta))*0.5)
        self.mamp = kwargs.get('mamp', 24)
        self.mtheta = kwargs.get("mtheta", 45)
        self.mpow = math.log(1-self.mtheta/90.0, 0.5)

        self.noise = kwargs.get('noise', 0.1) # noise level
        self.flux = kwargs.get('flux', 1.0) # [ proj/s nm^2 ] # Si
        self.dt = kwargs.get('dt', 1.0) # [ s ]
        self.vola = kwargs.get('vola', 0.0027324) # [ nm^3 ] atom volume // Si
        print(self.vola)

        self.flux_const = self.flux*self.dt*self.vola # [ proj ] poj in dt on cell

        self.z_rotation = np.radians(kwargs.get('z_rotation', 45))

        self.damp = kwargs.get('damp', 0.1)
        self.diff_cycles = kwargs.get('diff_cycles', 1)

        self.Z_history = []

    def single_step(self, look_up=False):
        self.Z_history.append(self.Z.copy())
        l_z = np.pad(self.Z, ((0, 1), (0,1)), mode='wrap')
        l_z += self.slope_corr_diff1

        l_slopes_x = np.diff(l_z, 1, axis=1)[:-1]/self.dx
        l_slopes_y = np.diff(l_z, 1, axis=0)[:,:-1]/self.dx

        l_angles_x = np.arctan(l_slopes_x)
        l_angles_y = np.arctan(l_slopes_y)

        angles_x = (np.roll(l_angles_x, 1, axis=1) + l_angles_x)*0.5
        angles_y = (np.roll(l_angles_y, 1, axis=0) + l_angles_y)*0.5
        slopes_x = np.tan(angles_x)
        slopes_y = np.tan(angles_y)

        normal_magnitude = np.sqrt(np.power(slopes_x, 2) + np.power(slopes_y, 2) + 1.0)
        thetas = np.arccos(1.0/normal_magnitude)
        beam_randomness = np.abs(np.random.normal(1,self.noise,(self.nodes_num, self.nodes_num)))

        # Sputtering erosion
        sputtered = (self.yamp * self.flux_const) * yamamura(thetas,self.ytheta, self.f)*beam_randomness
        #print("SPUT", np.sum(sputtered)/(len(sputtered)*len(sputtered)))
        self.Z -= sputtered

        # Moment erosion accumulation #
        omegas = np.arctan2(slopes_y, slopes_x)
        omegas = np.abs(omegas)
        omegas[omegas >= np.pi*0.5] = np.pi - omegas[omegas >= np.pi*0.5]

        x_back_mask = slopes_x > 0.0
        x_for_mask = np.logical_not(x_back_mask)

        y_back_mask = slopes_y > 0.0
        y_for_mask = np.logical_not(y_back_mask)

        try:
            if self.init_lookup == False:
                pass
        except:
            self.init_lookup = False
            __x__ = np.linspace(0, np.pi*0.5)
            #plt.plot(__x__, __x__)
            #plt.plot(__x__, 0.5*np.pi*np.power(__x__*2/np.pi, self.mpow))
            #plt.show()
            #plt.plot(np.degrees(__x__), (1.0-np.cos(4.0*0.5*np.pi*np.power(__x__*2/np.pi, self.mpow))), label='Displacement')
            plt.plot(np.degrees(__x__), (np.sin(np.pi*np.power(__x__*2/np.pi, self.mpow))), label='Displacement')
            plt.plot(np.degrees(__x__), yamamura(__x__, self.ytheta, self.f), label='Sputtering')
            plt.legend()
            plt.show()
        # ero_00 = (1.0-np.cos(4.0*thetas))*self.moment/(normal_magnitude*np.power(self.dx, 3))
        # ANGLE NORMALIZATION INSIDE DEFINITION BELOW
        # ero_00 = (self.mamp * self.flux_const * (1.0/self.dx) * 0.5) * (1.0-np.cos(4.0*thetas))

        # theta 0 > Pi/2
        #skewed_thetas =  0.5*np.pi*np.power(thetas*2/np.pi, self.mpow)
        #ero_00 = (self.mamp * self.flux_const * (1.0/self.dx) * 0.5) * np.sin(np.pi*np.power(skewed_thetas*2/np.pi, self.mpow))
        ero_00 = self.mamp * self.flux_const * (1.0/self.dx) * 0.5 * redist_fun(thetas)



        #ero_00 = (self.mamp * self.flux_const * (1.0/self.dx) * 0.5) * (1.0-np.cos(4.0*skewed_thetas))


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

            plt.subplot(223).set_title("Local Slope x")
            plt.imshow(l_slopes_x)
            plt.subplot(224).set_title("Conv Slope y")
            plt.imshow(self.Z- self.slope_background)
            plt.show()

        self.Z += summary*beam_randomness

        if False:
            print(np.degrees(omegas))
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')

            ax.plot_surface(self.X, self.Y, np.degrees(omegas), cmap=cm.coolwarm, linewidth=0, antialiased=False)
            plt.show()

        if False:
            # "kinetic montecarlo"
            for i in range(self.diff_cycles):
                Z_pad = np.pad(self.Z, 1, 'wrap')

                Z_pad += self.slope_corr_diff2

                x_diff = np.diff(Z_pad, 2, 1)[1:-1]
                y_diff = np.diff(Z_pad, 2, 0)[:,1:-1]

                node_energy_x = x_diff*-0.922 # x_diff*-0.922 # 1.55/(1.0+np.exp(x_diff*51+1.7))
                node_energy_y = y_diff*-0.922 # 1.55/(1.0+np.exp(y_diff*51+1.7))
                node_energy = node_energy_x + node_energy_y

                energy_diff_x = np.roll(node_energy, -1, axis=1) - node_energy
                energy_diff_y = np.roll(node_energy, -1, axis=0) - node_energy

                forward_x = -np.exp(-self.diffusion*energy_diff_x)+np.exp(self.diffusion*energy_diff_x)
                back_x = -np.roll(forward_x, 1, axis=1)
                forward_y = -np.exp(-self.diffusion*energy_diff_y)+np.exp(self.diffusion*energy_diff_y)
                back_y = -np.roll(forward_y, 1, axis=0)
                plt.imshow(energy_diff_x)
                plt.show()
                plt.imshow(energy_diff_y)
                plt.show()

                total_transport = forward_x + forward_y + back_x + back_y

                self.Z += total_transport
        else:
            for i in range(self.diff_cycles):
                l_z = np.pad(self.Z, ((0, 1), (0,1)), mode='wrap')
                l_z += self.slope_corr_diff1

                l_slopes_x = np.diff(l_z, 1, axis=1)[:-1]/self.dx
                l_slopes_y = np.diff(l_z, 1, axis=0)[:,:-1]/self.dx

                l_angles_x = np.arctan(l_slopes_x)
                l_angles_y = np.arctan(l_slopes_y)

                # /\ 180, \/ -180
                #rx = np.arange(-np.radians(360), np.radians(360), 0.1)
                #plt.plot(rx, np.tan(ext*rx*0.25) )
                #plt.show()

                node_angles_x = np.roll(l_angles_x, (1, 0), axis=(1, 0)) - l_angles_x
                node_angles_y = np.roll(l_angles_y, (0, 1), axis=(1, 0)) - l_angles_y

                node_angles = (node_angles_x + node_angles_y)*0.5

                # forward distance
                l_dist_x = np.cos(l_angles_x) # self.dx
                l_dist_y = np.cos(l_angles_y) # self.dx

                # for now assuming constant surface and max 1 atom layer per cycle
                vol_h = np.power(self.vola, 1./3.)
                forward_transport_x = -vol_h*np.sin(0.25*(node_angles - np.roll(node_angles, -1, axis=1)))/l_dist_x
                forward_transport_y = -vol_h*np.sin(0.25*(node_angles - np.roll(node_angles, -1, axis=0)))/l_dist_y
                """
                ext = 3.0
                #print(np.degrees(np.max(l_angles_x)), np.degrees(np.max(l_angles_y)))
                forward_transport_x = -vol_h*np.tan(ext*0.25*(node_angles - np.roll(node_angles, -1, axis=1)))/l_dist_x
                forward_transport_y = -vol_h*np.tan(ext*0.25*(node_angles - np.roll(node_angles, -1, axis=0)))/l_dist_y
                """


                backward_transport_x = -np.roll(forward_transport_x, 1, axis=1)
                backward_transport_y = -np.roll(forward_transport_y, 1, axis=0)
                total_transport = forward_transport_x + backward_transport_x + forward_transport_y + backward_transport_y

                self.Z += self.damp*total_transport

    ''' Modify/test surface functions '''
    def add_sin(self, amp, nx, ny):
        self.Z += amp*np.sin(2*np.pi*(nx*self.X/self.sample_len + ny*self.Y/self.sample_len))

    def add_gauss(self, amp, sigmax, sigmay):
        center = self.xy_spacing[int(len(self.xy_spacing)/2)]
        x_c = self.X-center
        y_c = self.Y-center

        self.Z += amp*np.exp(-np.power(x_c,2)/(2*np.power(sigmax,2))
                             -np.power(y_c,2)/(2*np.power(sigmay,2)))

    def add_cos(self, amp, nx, ny):
        self.Z += amp*np.cos(2*np.pi*(nx*self.X/self.sample_len + ny*self.Y/self.sample_len))

    def add_pilar(self, amp, x0, y0, ratio=0.1):
        r = self.sample_len*ratio
        a = -amp/np.power(r,2)

        pilar = a*(np.power(self.X-x0,2) + np.power(self.Y-y0, 2))+amp
        pilar[pilar<0.0] = 0.0
        self.Z += pilar
        self.Z_history.append(self.Z)

    ''' Display functions '''
    def leveled_xyz(self, Z_, n=0, correction=True):
        # extend X Y Z
        z_mean = np.mean(Z_)
        z_eroded = self.start_ave_surf - z_mean
        print("Eroded_sample: {} ({})".format(np.cos(self.theta)*z_eroded, z_eroded))

        if correction:
            roll_xy = int(np.round((np.sin(self.theta)*z_eroded/np.sqrt(2))/self.dx))
            roll_xy_rest = (np.sin(self.theta)*z_eroded/np.sqrt(2))/self.dx - np.round((np.sin(self.theta)*z_eroded/np.sqrt(2))/self.dx)
            print("Roll: {}\nRoll rest: {}".format(roll_xy, roll_xy_rest))
        else:
            roll_xy = 0

        Z_normalized = Z_ - (self.slope_background - np.mean(self.slope_background)) - z_mean
        Z_normalized = np.roll(Z_normalized, (roll_xy, roll_xy), axis=(0,1))

        # extend by n
        Z_normalized = np.pad(Z_normalized, (0, self.nodes_num*n), mode='wrap')
        xy_spacing = np.linspace(0, self.sample_len*(n+1), self.nodes_num*(n+1), endpoint=False) - self.sample_len*0.5*(n+1)
        if correction:
            xy_spacing += roll_xy_rest

        x_center = np.tile(xy_spacing, (xy_spacing.shape[0], 1))
        y_center = x_center.T

        Z_ = Z_normalized + self.sample_slope_xy*x_center + self.sample_slope_xy*y_center

        # rotate to xy plane
        xyz = np.stack((x_center, y_center, Z_), axis=-1)
        rot_matrix = rotation_matrix(np.array([-1,1,0], dtype=float), self.theta)
        xyz_rot = np.tensordot(xyz, rot_matrix, axes=([2], [0]))
        # rotate around z axis
        rot_matrix = rotation_matrix(np.array([0,0,1], dtype=float), self.z_rotation)
        xyz_rot = np.tensordot(xyz_rot, rot_matrix, axes=([2], [0]))

        xyz_unstacked = np.split(xyz_rot, 3, axis=-1)

        X_ = np.squeeze(xyz_unstacked[0])
        Y_ = np.squeeze(xyz_unstacked[1])
        Z_ = np.squeeze(xyz_unstacked[2])

        return X_, Y_, Z_

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

    '''
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
    '''

    def show_history(self, n=0):
        #self.name_str = "theta={}, sput={}, moment={}, diff={}, conv={}".format(np.degrees(self.theta), self.erosion, self.moment, self.diffusion, self.conv_sigma)
        self.name_str =" - "
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

        plot, = ax.plot(xy[0], xy[1], 'r+')

        axslider = plt.axes([0.15, 0.1, 0.65, 0.05])
        slider = Slider(axslider, 'Tmp', 0, len(self.Z_history)-1, valinit=1)

        def update(val):
            num = int(val)
            Z_ = self.Z_history[num]
            l_z = np.pad(Z_, ((0, 1), (0,1)), mode='wrap')
            l_z += self.slope_corr_diff1

            l_slopes_x = np.diff(l_z, 1, axis=1)[:-1]/self.dx
            l_slopes_y = np.diff(l_z, 1, axis=0)[:,:-1]/self.dx

            l_angles_x = np.arctan(l_slopes_x)
            l_angles_y = np.arctan(l_slopes_y)

            angles_x = (np.roll(l_angles_x, 1, axis=1) + l_angles_x)*0.5
            angles_y = (np.roll(l_angles_y, 1, axis=0) + l_angles_y)*0.5
            slopes_x = np.tan(angles_x)
            slopes_y = np.tan(angles_y)

            normal_magnitude = np.sqrt(np.power(slopes_x, 2) + np.power(slopes_y, 2) + 1.0)
            thetas = np.arccos(1.0/normal_magnitude)
            omegas = np.arctan2(slopes_y, slopes_x)
            omegas = np.abs(omegas)
            omegas[omegas >= np.pi*0.5] = np.pi - omegas[omegas >= np.pi*0.5]

            x_back_mask = slopes_x > 0.0
            x_for_mask = np.logical_not(x_back_mask)

            y_back_mask = slopes_y > 0.0
            y_for_mask = np.logical_not(y_back_mask)

            # ero_00 = (1.0-np.cos(4.0*thetas))*self.moment/(normal_magnitude*np.power(self.dx, 3))
            # ANGLE NORMALIZATION INSIDE DEFINITION BELOW
            ero_00 = (self.mamp * self.flux_const * (1.0/self.dx) * 0.5) * (1.0-np.cos(4.0*thetas))
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

            moment = -ero_00+acc_00+acc_01+acc_10+acc_11
            moment_diag = np.diag(moment)
            sput = (self.yamp * self.flux_const) * yamamura(thetas,self.ytheta, self.f)
            sput_diag = np.diag(sput)
            print(np.min(sput_diag), np.max(sput_diag))

            X_, Y_, Z_ = self.leveled_xyz(Z_, 0, correction=True)

            z_diag = np.diag(Z_)
            x_diag = np.diag(X_)

            ax.clear()
            ax.plot(x_diag,z_diag)
            ax.scatter(x_diag,z_diag)

            max_x = np.max(np.abs(z_diag))
            max_sm = np.max([np.max(np.abs(sput_diag)), np.max(np.abs(moment_diag))])
            sput_diag = -(max_x/max_sm)*sput_diag
            moment_diag = (max_x/max_sm)*moment_diag

            ax.plot(x_diag, sput_diag)
            ax.scatter(x_diag, sput_diag, c='r')

            ax.plot(x_diag, moment_diag)
            ax.scatter(x_diag, moment_diag)

        slider.on_changed(update)
        plt.show()

    def show_img(self, start_frame, period=None, output=None,
                 cell_rep=1, vmax=None, scalebar=False,
                 boundary=None, offset=5, boundary_offset=5, img_offset=10,
                 cmap="gray", mode='tri',
                 bin_step=1, r=5, bin_rep=7,
                 tri_method='linear', roll_correction=True):

        if start_frame < 0:
            start_frame = len(self.Z_history)-1

        if period is None:
            period = 100000000

        for num in range(start_frame, len(self.Z_history), period):
            img_name = "rip.{}.png".format(str(int(num/period)).zfill(5))

            print("DISPL NUM::{}".format(num))
            X_, Y_, Z_ = self.leveled_xyz(self.Z_history[num], cell_rep, correction=roll_correction)
            flat = [None, None, None]
            flat[0] = X_.flatten()
            flat[1] = Y_.flatten()
            flat[2] = Z_.flatten()
            flat = np.asarray(flat)

            if boundary is None:
                boundary = self.get_img_boundary(X_, Y_, Z_)
                boundary[0][0] += offset
                boundary[1][0] += offset
                boundary[0][1] -= offset
                boundary[1][1] -= offset

            print(boundary)

            if mode == 'scatter3d' or mode == 'scatter':
                #x_flag = np.logical_and(flat_[0] > boundary[0][0], flat_[0] < boundary[0][1])
                #y_flag = np.logical_and(flat_[1] > boundary[1][0], flat_[1] < boundary[1][1])
                #flat = np.array(flat_)
                #flat = flat[:, np.logical_and(x_flag, y_flag)]

                if mode == 'scatter3d':
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')

                    ax.scatter(flat[0], flat[1], flat[2])
                    ax.set_xlabel("X axis")
                    ax.set_ylabel("Y axis")
                    plt.show()

                else:
                    plt.scatter(flat[0], flat[1])
                    plt.show()


            elif mode == 'surf':
                r2 = np.power(r,2)
                bin_centers = [np.arange(boundary[0,0], boundary[0,1]+0.001, bin_step),
                                        np.arange(boundary[1,0], boundary[1,1]+0.001, bin_step)]

                bin_edges = np.array([bin_centers[0][:-1]+0.5*bin_step, bin_centers[1][:-1]+0.5*bin_step])

                indexes = [None, None]
                indexes[0] = np.digitize(flat_[0], bin_edges[0]) # x bins
                indexes[1] = np.digitize(flat_[1], bin_edges[1]) # y bins

                safe_offset = -1000000
                Z_holder = np.ones((bin_centers[1].shape[0], bin_centers[0].shape[0], 3, 3), dtype=float)*safe_offset

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

                probe_z[np.isnan(probe_z)] = safe_offset
                probe_z = np.amax(probe_z, -1)

                probe_z[probe_z <= safe_offset+1] = 0.0
                probe_z -= np.min(probe_z)
                print("Probe {}".format(np.max(probe_z)))

                if vmax is None:
                    vmax = np.max(probe_z)

                if vmax == "get":
                    return np.max(probe_z)

                # normalize
                probe_z = probe_z / vmax
                probe_z = (0.5-np.mean(probe_z)) + probe_z

                if output:
                    plt.imsave("{}/{}".format(output, img_name), probe_z, cmap=cmap)
                else:
                    plt.imshow(probe_z, cmap=cmap, vmin=0.0, vmax=1.0)
                    plt.show()

            elif mode == 'tri' or mode == 'tri_scatter':
                tri = Triangulation(flat[0], flat[1])
                if tri_method == 'cube':
                    interpol = CubicTriInterpolator(tri, flat[2])
                else:
                    interpol = LinearTriInterpolator(tri, flat[2])

                x_new = np.arange(boundary[0][0], boundary[0][1], self.img_dx)
                y_new = np.arange(boundary[1][0], boundary[1][1], self.img_dx)
                X_new, Y_new = np.meshgrid(x_new, y_new)

                Z_new = interpol(X_new, Y_new)

                print("Orginal range: {}   {}".format(np.min(Z_new), np.max(Z_new)))

                Z_new = Z_new - np.min(Z_new)

                if vmax == "get":
                    return np.max(Z_new)
                if vmax is None:
                    vmax = np.max(Z_new)

                Z_new = Z_new / vmax
                Z_new += (1-np.max(Z_new))*0.5


                if scalebar:
                    plt.gca().add_artist(ScaleBar(1.0/self.img_dx, 'nm'))

                if output:
                    if mode == 'tri':
                        plt.imsave("{}/{}".format(output, img_name), Z_new, vmin=0, vmax=1, cmap=cmap)
                    else:
                        inside_box = np.logical_and(
                            np.logical_and(flat[0]>boundary[0][0], flat[0]<boundary[0][1]),
                            np.logical_and(flat[1]>boundary[1][0], flat[1]<boundary[1][1]))
                        scatter_x = flat[0, inside_box]
                        scatter_y = flat[1, inside_box]

                        plt.imshow(Z_new, vmin=0, vmax=1, cmap=cmap)
                        plt.scatter(scatter_x-boundary[0][0], scatter_y-boundary[1][0], marker=',', s=1, lw=0, color='red')
                        plt.savefig("{}/{}".format(output, img_name))
                        plt.cla()


                else:
                    plt.imshow(Z_new, vmin=0, vmax=1, cmap=cmap)
                    if mode == 'tri_scatter':
                        inside_box = np.logical_and(
                            np.logical_and(flat[0]>boundary[0][0], flat[0]<boundary[0][1]),
                            np.logical_and(flat[1]>boundary[1][0], flat[1]<boundary[1][1]))
                        scatter_x = flat[0, inside_box]
                        scatter_y = flat[1, inside_box]


                        plt.scatter(scatter_x-boundary[0][0], scatter_y-boundary[1][0], marker=',', s=1, lw=0, color='red')
                    plt.show()

        #indexed_si = np.digitize(self.start_all_z_list[0]- self.ave_surf, self.dp_bin_edges)
        #[np.linspace(boundary[0,0], boundary[0,1], 1.0)

    def show_yam(self):
        plt.plot(np.linspace(0,90), self.yamp*yamamura(np.linspace(0, np.pi/2.0), self.ytheta, self.f))
        plt.xlim([0,90])
        plt.show()

    def show_color(self, num=-1, displ='sput'):
        Z_ = self.Z_history[num]
        X_ = self.X
        Y_ = self.Y
        l_z = np.pad(Z_, ((0, 1), (0,1)), mode='wrap')
        l_z += self.slope_corr_diff1

        l_slopes_x = np.diff(l_z, 1, axis=1)[:-1]/self.dx
        l_slopes_y = np.diff(l_z, 1, axis=0)[:,:-1]/self.dx

        l_angles_x = np.arctan(l_slopes_x)
        l_angles_y = np.arctan(l_slopes_y)

        angles_x = (np.roll(l_angles_x, 1, axis=1) + l_angles_x)*0.5
        angles_y = (np.roll(l_angles_y, 1, axis=0) + l_angles_y)*0.5
        slopes_x = np.tan(angles_x)
        slopes_y = np.tan(angles_y)

        normal_magnitude = np.sqrt(np.power(slopes_x, 2) + np.power(slopes_y, 2) + 1.0)
        thetas = np.arccos(1.0/normal_magnitude)
        if displ == 'moment':
            omegas = np.arctan2(slopes_y, slopes_x)
            omegas = np.abs(omegas)
            omegas[omegas >= np.pi*0.5] = np.pi - omegas[omegas >= np.pi*0.5]

            x_back_mask = slopes_x > 0.0
            x_for_mask = np.logical_not(x_back_mask)

            y_back_mask = slopes_y > 0.0
            y_for_mask = np.logical_not(y_back_mask)

            # ero_00 = (1.0-np.cos(4.0*thetas))*self.moment/(normal_magnitude*np.power(self.dx, 3))
            # ANGLE NORMALIZATION INSIDE DEFINITION BELOW
            ero_00 = (self.mamp * self.flux_const * (1.0/self.dx) * 0.5) * (1.0-np.cos(4.0*thetas))
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

            results = -ero_00+acc_00+acc_01+acc_10+acc_11

        elif displ == 'sput':
            results = (self.yamp * self.flux_const) * yamamura(thetas,self.ytheta, self.f)

        res_max = np.max(results)
        res_min = np.min(results)

        print(res_max, res_min)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_aspect('equal')

        if displ == 'moment':
            abs_max = max(abs(res_min), abs(res_max))
            normalized = (results/(2*abs_max))+0.5
            # 0 -> 255
            digitized = np.array(255*normalized, dtype=int)
            print(np.min(normalized), np.max(normalized))
            print(np.min(digitized), np.max(digitized))
            my_col = cm.seismic(digitized)

        elif displ == 'sput':
            my_col = cm.afmhot((results-res_min)/(res_max-res_min))

        max_range=np.array([X_.max()-X_.min(),
                         Y_.max()-Y_.min(),
                         Z_.max()-Y_.min()]).max() / 2.0

        mid_x = (X_.max()+X_.min())*0.5
        mid_y = (Y_.max()+Y_.min())*0.5
        mid_z = (Z_.max()+Z_.min())*0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        surf = ax.plot_surface(X_, Y_, Z_, facecolors=my_col)
        #, lrstride=1, cstride=1, inewidth=0)#,antialiased=False
        plt.show()
        f, (a3, a4) = plt.subplots(1,2)
        #a1.imshow(l_slopes_x)
        #a2.imshow(np.degrees(angles_x))
        a3.imshow(np.degrees(thetas))
        a4.imshow(results)
        plt.show()

    ''' Read write functions '''
    @classmethod
    def load(cls, iname):
        with open(iname, 'br') as f:
            return pickle.load(f)

    def dump(self, oname):
        with open(oname, 'bw') as f:
            pickle.dump(self, f)

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

    def write_img(self, out_dir, period=10, cell_rep=1,
                  vmax=None, show=True,
                  boundary=None, boundary_offset=10, img_offset=10,
                  cmap="gray", mode='tri',
                  bin_step=1, r=5, bin_rep=7,
                  tri_method='linear'):

        print("{}, {}, {}".format(0, len(self.Z_history)-1, period))

        for i in range(0, len(self.Z_history)-1, period):
            name = "rip.{}.png".format(str(int(i/period)).zfill(5))

            #fig = self.show_img(i, bin_step=bin_step, r=r, bin_rep=bin_rep, cell_rep=cell_rep, vmax=vmax, boundary=boundary, show=False)
            #save_figure("{}/{}".format(out_dir, name), fig)
            #plt.savefig("{}/{}".format(out_dir, name), transparent=True, bbox_inches='tight', pad_inches=0)

            data = self.show_img(i, bin_step=bin_step, r=r, bin_rep=bin_rep, cell_rep=cell_rep, vmax=vmax, boundary=boundary, show=False)
            plt.imsave("{}/{}".format(out_dir,name), data, cmap=cmap) #, vmin=0.0, vmax=1.0)#, cmap=cm.afmhot)
    ''' Convinience functions '''
    def run(self, steps):
        info_steps = 1000
        print("Starting {} steps with info period {}".format(steps, info_steps))

        prev_time = time.time()
        for i in range(steps):
            self.single_step()
            if i%info_steps == 0 and i != 0:

                cycle_time = time.time() - prev_time
                prev_time = time.time()
                print("Cycle time {}".format(cycle_time))


    def irun(self, stepnum=10):
        #interactive run
        try:
            steps = int(input("Steps to perform:"))
        except:
            steps = 0

        while steps != 0:
            for step in range(steps):
                t = time.time()
                for i in range(stepnum):
                    self.single_step()
                single_loop = time.time() - t
                loops_to_end = steps-step
                print("single:    {} s \nestimated: {} s  ({} min)".format(single_loop, loops_to_end*single_loop, loops_to_end*single_loop/60.0))

            self.show_history_1d(aspect=1, rotate=False)
            self.show_history(n=1)
            self.show_img(len(self.Z_history)-1, cmap='afmhot')
            try:
                steps = int(input("Steps to perform:"))
            except:
                steps = 0

    def iwrite(self):
        per=int(input("You have {} frames now, choose period:".format(len(self.Z_history))))
        self.write_img("/tmp/playground2",period=per, r=9, bin_rep=4)

    def __len__(self):
        return len(self.Z_history)

vol_au = 0.0168
theta = float(input("Enter incidence angle:"))
m = model2d(theta=theta, mtheta=26, vola=vol_au, yamp=1, mamp=1, dt=0.001,
            sample_len=50, nodes_num=40, damp=0.005, diff_cycles=2, noise=0.10)
m.irun()
