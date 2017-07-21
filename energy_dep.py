import numpy as np
import matplotlib.pyplot as plt


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
        self.dx = self.x_n[1]
        self.x = self.x_n*self.sample_len

        self.y_n = np.zeros(self.x_n.shape, dtype=float)


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

        # energy deposition gaussian
        self.ion_range = 4*self.gsigma*np.sin(self.theta) # [A] ! alogn x !
        if self.ion_range > self.sample_len:
            raise ValueError(("Ion range (gsigma*3 = {}) "
                              "bigger than sample length (sample_len = {})!")
                             .format(self.ion_range, self.sample_len))
        self.ion_range_n = self.ion_range*self.dist_scale # [dist*]

        self.ion_range_nodes = int(self.ion_range_n/self.dx) # [number of nodes]
        if self.ion_range_nodes < 2:
            raise ValueError("Something wrong with number of ion_range_nodes")

        self.energy_gauss = (self.erosion_n * np.exp(
            -np.power((self.x_n[:self.ion_range_nodes]-self.gshift_n)*(1.0/np.sin(self.theta)), 2)
            /(2.0*np.power(self.gsigma_n, 2))))

    def add_sin(self, amp, n):
        self.y_n += self.dist_scale*amp*np.sin(self.x_n*(2*n*np.pi))

    def show_energy_gauss(self):
        plt.plot(self.x[:self.ion_range_nodes], self.energy_gauss)
        plt.show()

    def show_surface(self, keep_ratio=False):
        plt.plot(self.x, self.y_n/self.dist_scale)
        if keep_ratio:
            plt.ylim((-0.5*self.sample_len, 0.5*self.sample_len))

        plt.show()

    def single_update(self, show=False):
        erosion_sum = np.zeros(self.x_n.shape, dtype=float)
        pad_len = self.x_nodes-self.ion_range_nodes
        min_value = 1000.0
        max_value = -1000.0

        x_n = np.linspace(0, 2.0, 2*self.x_nodes, endpoint=False)
        x = x_n*self.sample_len

        for i in range(self.x_nodes):
            # @1 roll surface
            y_roll = np.roll(self.y_n, -i)
            # @2 calculate diff in distace
            y_diff = (y_roll[:self.ion_range_nodes] - y_roll[0]) + self.x_n[:self.ion_range_nodes]

            if show:
                plt.plot(x[i:(self.ion_range_nodes+i)] ,y_diff)

            min_value = min(np.min(y_diff), min_value)
            max_value = max(np.max(y_diff), min_value)

            decay_x = np.exp(-self.decay_n * y_diff)

            # @3 multipl decay and energy depostion and roll back
            erosion_sum += np.roll(np.lib.pad(self.energy_gauss * decay_x, (0, pad_len), 'constant'), i)

        if show:
            print min_value, max_value
            plt.plot(self.x, erosion_sum/max(erosion_sum))
            plt.plot(self.x, self.y_n/max(self.y_n))
            plt.show()

        self.y_n -= erosion_sum
        self.y_n -= np.mean(self.y_n)



ed = EnergyDepositionModel(x_nodes=40, samp_len=200, gsigma=30, gshift=30, erosion=1)
#ed.show_energy_gauss()
ed.add_sin(1,2)
#ed.show_surface(True)
#for i in range(30):
#    ed.single_update()
#    ed.single_update(True)



