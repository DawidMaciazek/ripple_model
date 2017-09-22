import numpy as np
import matplotlib.pyplot as plt


def yamamura(theta, theta_opt, f):
    return np.power(np.cos(theta), -f)*np.exp(f * (1-np.power(np.cos(theta), -1)) * np.cos(theta_opt)  )


class SputRate:
    def __init__(self, nodes_cnt=100, samp_size=100):
        self.samp_size = samp_size
        self.x = np.linspace(0, self.samp_size, nodes_cnt, endpoint=False)
        self.y = np.zeros((nodes_cnt, ), dtype=float)

    def add_sin(self, n, amp):
        self.y += amp*np.sin(n*2.0*np.pi*self.x/self.samp_size)

    def add_obliq(self, theta):
        self.y += np.tan(np.radians(theta))*self.x

    def show(self, sigma, theta_opt=60, f=2.4, display=True, show_yam=False):
        sigma_multi = 2
        theta_opt = np.radians(theta_opt)
        if show_yam:
            plt.plot(np.linspace(0,90,endpoint=False), yamamura(np.linspace(0, np.pi*0.5,endpoint=False), theta_opt, f))
            plt.show()

        # step 1: calc middle points
        l_slopes = np.diff(np.pad(self.y, (0, 1), mode='wrap'), 1)/self.x[1]
        l_angles = np.arctan(l_slopes)

        # setp 2: setup gauss conv_function
        x_ = self.x[self.x<=sigma*sigma_multi*2.0]
        if len(x_)%2 != 0:
            x_ = self.x[:len(x_)+1]

        wrap_len = int(len(x_)/2)

        conv_center = x_[-1]*0.5
        conv_fun = np.exp(-np.power(x_-conv_center, 2)/(2*np.power(sigma, 2)) )
        conv_fun = conv_fun/np.sum(conv_fun)

        wrap_angles = np.pad(l_angles, (wrap_len, wrap_len-1), mode='wrap')
        angles = np.convolve(wrap_angles, conv_fun, mode='valid')

        plt.plot(self.x, l_angles)
        plt.plot(self.x, angles)
        plt.show()

        plt.plot(self.x, yamamura(angles, theta_opt, f))
        plt.plot(self.x, yamamura(l_angles, theta_opt, f))
        plt.show()

    def run_convolution(self, sigma, skip_conv=False):
        x_ = self.x[self.x<sigma*4]
        conv_fun = np.exp( - np.power(x_-sigma*2.0,2)/(2*np.power(sigma,2)) )
        conv_fun /= np.sum(conv_fun*self.x[1])

        #plt.plot(x_,conv_fun)
        dif = np.diff(np.pad(self.y,(0,1),mode='wrap'), 1)/self.x[1]
        angles = np.degrees(np.arctan(dif))
        angles_conv = np.convolve(angles, conv_fun, mode='same')


        #plt.plot(angles)
        if skip_conv:
            plt.plot(angles)
        else:
            plt.plot(angles_conv)

        #plt.show()




sp = SputRate()
sp.add_sin(2,6)
sp.add_sin(17,1)
sp.add_obliq(-15)
plt.plot(sp.x, sp.y)
plt.show()
sp.show(10, theta_opt=70)

#sp.run_convolution(10)

# -------

