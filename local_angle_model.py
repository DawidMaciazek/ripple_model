import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

#import warnings
#warnings.filterwarnings('error')

def yamamura(theta, theta_opt, f):
    return np.power(np.cos(theta), -f)*np.exp(f * (1-np.power(np.cos(theta), -1)) * np.cos(theta_opt)  )


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


        self.y -= yamamura(angles,self.theta_opt, self.f)*np.abs(np.random.normal(1,0.3,self.nodes_num)) #*np.roll(self.gauss_random, np.random.randint(self.nodes_num))
        # *self.beam_gauss
        #plt.plot(yamamura(angles,self.theta_opt, self.f))
        #plt.show()


        # - (1 - cos(angl)) + (1 - cos(angl[+1]))
        cos_res = np.cos(2*angles)
        #self.y += self.moment*(np.cos(angles) - np.cos(np.roll(angles, 1)))
        moment = self.moment*(cos_res - np.roll(cos_res, 1))

        self.y += moment

        self.counter+=1
        if self.counter%10000 == 0:
            print(cos_res)
            print(moment)
            plt.plot(moment)
            plt.show()


        self.y_history.append(self.y.copy())

        l_y = np.pad(self.y, (1, 1), mode='wrap')
        l_y[-1] += self.slope_corr
        l_y[0] -= self.slope_corr-self.sample_slope*self.dx

        self.y += self.diffusion*np.diff(l_y, 2)

        # sample slope correction
        #l_angles = np.arc


    def show(self):
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)

        ax.set_aspect(1)
        plot, = ax.plot(self.x, self.y_history[-1]-np.mean(self.y_history[-1]))

        axslider = plt.axes([0.15, 0.1, 0.65, 0.05])
        slider = Slider(axslider, 'Tmp', 0, len(self.y_history)-1, valinit=1)


        def update(val):
            selection = self.y_history[int(val)]
            traveled=np.mean(selection)
            plot.set_ydata(selection -traveled)
            print("Traveled:{}".format(traveled))

        slider.on_changed(update)
        plt.show()

    def add_sin(self, amp, n):
        self.y += amp*np.sin(n*2*np.pi*self.x/self.sample_len)


    def show_yam(self):
        plt.plot(np.linspace(0,90), yamamura(np.linspace(0, np.pi/2.0), self.theta_opt, self.f))
        plt.show()


m = model(theta=60, theta_opt=76, conv_sigma=1, erosion=0.1, moment=1.1, diffusion=0.1, sample_len=500, nodes_num=2000)
#m.show_yam()
m.add_sin(25,2)
#m.add_sin(2,12)
m.run(8000)
m.show()

