import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


class EnergyDepositionModel:
    def __init__(self, **kwargs):
        """
        ------
        # sample parameters
        sample_len: float
            Sample length (A)

        x_nodes: int
            nodes in x dimension

        # equation parameters
        decay: float
            Sputtering yield decay depth (1/A)

        gshift: float
            Gaussian shift (A)

        gsigma: float
            Gaussian sigma (A)
        """

        self.sample_len = kwargs.get('samp_len', 500) # [A]
        self.dist_scale = 1.0/self.sample_len # [dist*/A]

        self.x_nodes = kwargs.get('x_nodes', 500)
        self.x_n = np.linspace(0, 1.0, self.x_nodes, endpoint=False)
        self.dx_n = self.x_n[1]
        self.x = self.x_n*self.sample_len

        self.y_n = np.zeros(self.x_n.shape, dtype=float)

        self.y_history = []
        self.history_initialized = False

        self.erosion = kwargs.get('erosion', 0.01) # [A/s]
        self.erosion_n = self.erosion*self.dist_scale # [dist*/s]

        self.decay = kwargs.get('decay', 0.05) # [1/A]
        self.decay_n = self.decay*(1/self.dist_scale) # [1/dist*]

        self.gshift = kwargs.get('gshift', 30) # [A]
        self.gshift_n = self.gshift*self.dist_scale # [dist*]

        self.gsigma = kwargs.get('gsigma', 30) # [A]
        self.gsigma_n = self.gsigma*self.dist_scale # [dist*]

        self.theta_deg = kwargs.get('theta', 60) # [deg]
        self.theta = np.radians(self.theta_deg) # [rad]
        print self.theta_deg, self.theta

        self.diffusion = kwargs.get('diffusion', 1) # [A^2/s]
        self.diffusion_n = self.diffusion*np.power(self.dist_scale, 2) / np.power(self.dx_n, 2) #

        # energy deposition gaussian
        self.ion_range = 4*self.gsigma*np.sin(self.theta) # [A] ! alogn x !
        if self.ion_range > self.sample_len:
            raise ValueError(("Ion range (gsigma*3 = {}) "
                              "bigger than sample length (sample_len = {})!")
                             .format(self.ion_range, self.sample_len))
        self.ion_range_n = self.ion_range*self.dist_scale # [dist*]

        self.ion_range_nodes = int(self.ion_range_n/self.dx_n) # [number of nodes]
        if self.ion_range_nodes < 2:
            raise ValueError("Something wrong with number of ion_range_nodes")

        self.energy_gauss = (self.erosion_n * np.exp(
            -np.power((self.x_n[:self.ion_range_nodes]-self.gshift_n)*(1.0/np.sin(self.theta)), 2)
            /(2.0*np.power(self.gsigma_n, 2))))

        # surface small RMS:
        self.rms  = kwargs.get("rms", 5) # A
        self.rms_n = self.rms*self.dist_scale # dist*
        self.rms_cut = kwargs.get("rmscut", 0.05)
        self.rms_slope = -np.log(self.rms_cut/(1-self.rms_cut))/self.rms_n
        ##


    def add_sin(self, amp, n):
        self.y_n += self.dist_scale*amp*np.sin(self.x_n*(2*n*np.pi))

    def add_gauss(self, amp, sigma, shift):
        self.y_n += amp*self.dist_scale*np.exp(-np.power(self.x_n-shift*self.dist_scale,2)/(2*np.power(sigma*self.dist_scale, 2)))


    def show_energy_gauss(self):
        plt.plot(self.x[:self.ion_range_nodes], self.energy_gauss)
        plt.show()

    def show_surface(self, keep_ratio=False):
        plt.plot(self.x, self.y_n/self.dist_scale)
        if keep_ratio:
            plt.ylim((-0.5*self.sample_len, 0.5*self.sample_len))

        plt.show()

    def single_update(self, show=False):
        if not self.history_initialized:
            self.history_initialized = True
            self.y_history.append(self.y_n/self.dist_scale)

        erosion_sum = np.zeros(self.x_n.shape, dtype=float)
        pad_len = self.x_nodes-self.ion_range_nodes
        min_value = 1000.0
        max_value = -1000.0

        x_n = np.linspace(0, 2.0, 2*self.x_nodes, endpoint=False)
        x = x_n*self.sample_len

        normal_noise = np.abs(np.random.normal(0, 1, self.x_nodes))
        for i in range(self.x_nodes):
            # @1 roll surface
            y_roll = np.roll(self.y_n, -i)
            # @2 calculate diff in distace
            y_diff = (y_roll[:self.ion_range_nodes] - y_roll[0]) + self.x_n[:self.ion_range_nodes]/np.tan(self.theta)

            if show:
                plt.plot(x[i:(self.ion_range_nodes+i)] ,y_diff)

            min_value = min(np.min(y_diff), min_value)
            max_value = max(np.max(y_diff), min_value)

            decay_x = np.exp(-self.decay_n * y_diff)/(1.0+np.exp(-self.rms_slope*(y_diff-self.rms_n)))


            #plt.plot(self.x, np.lib.pad(decay_x, (0, pad_len), 'constant'))
            #plt.show()

            #plt.plot(y_diff/self.dist_scale, (1.0/(1.0+np.exp(-self.rms_slope*(y_diff-self.rms_n)))))
            #plt.show()

            # @3 multipl decay and energy depostion and roll back
            erosion_sum += np.roll(np.lib.pad(self.energy_gauss * decay_x, (0, pad_len), 'constant'), i)*normal_noise[i]

        if show:
            print min_value, max_value
            print "SIZE:", max(self.x_n) - min(self.y_n)
            plt.plot(self.x, erosion_sum/max(erosion_sum))
            plt.plot(self.x, self.y_n/max(self.y_n))
            plt.show()

        self.y_n -= erosion_sum

        # compute diffusion
        if self.diffusion:
            diff = np.diff(np.pad(self.y_n, (1, 1), 'wrap'), 2) * self.diffusion
            self.y_n += diff

        self.y_n -= np.mean(self.y_n)
        self.y_history.append(self.y_n/self.dist_scale)

    def show_history(self):
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)

        plt.axis([self.x[0], self.x[-1], -0.2, 0.2])

        plot_current, = plt.plot(self.x, self.y_history[0])

        axslider = plt.axes([0.15, 0.1, 0.65, 0.05])
        slider = Slider(axslider, 'Tmp', 0, len(self.y_history)-1, valinit=0)

        def update(val):
            #print "choosen {}/{}".format(int(val), len(self.y_history))
            selection = self.y_history[int(val)]
            plot_current.set_ydata(selection)

        slider.on_changed(update)
        plt.show()


ed = EnergyDepositionModel(x_nodes=800, samp_len=400, gsigma=28, gshift=30, erosion=0.03, rms=1, theta=75, diffusion=0.002)
#ed.show_energy_gauss()
#ed.add_gauss(0.1, 20, 50)
#ed.add_sin(3.1,4)
#ed.add_sin(0.1,31)
#ed.add_sin(0.1,52)
for i in range(1000):
    ed.single_update()
ed.show_history()
#ed.show_surface(True)
#for i in range(30):
#    ed.single_update()
#    ed.single_update(True)
