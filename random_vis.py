import matplotlib.pyplot as plt
import numpy as np

class RandomVisual:
    def __init__(self, size_mul, gsigma=0.1, base_size=100):
        plt.figure(0)
        self.size_mul = size_mul
        #self.small_ax = plt.subplot2grid((2, self.size_mul), (0, 0), colspan=1)
        self.small_ax = plt.subplot2grid((2, self.size_mul), (0, 0), colspan=self.size_mul)
        self.large_ax = plt.subplot2grid((2, self.size_mul), (1, 0), colspan=self.size_mul)

        self.small_size = base_size
        self.large_size = base_size*self.size_mul

        self.small_ax.set_xlim([0, 1.0])
        self.large_ax.set_xlim([0, 1.0*self.size_mul])

        self.small_x = np.linspace(0, 1.0, base_size, False)
        self.small_x += self.small_x[1]*0.5

        self.large_x = np.linspace(0, 1.0*self.size_mul, base_size*self.size_mul, False)
        self.large_x += self.large_x[1]*0.5

        self.small_y = np.zeros(base_size, dtype=float)
        self.large_y = np.zeros(base_size*self.size_mul, dtype=float)

        # gaussian for convolution
        if gsigma*6 > 1.0:
            raise ValueError("Sigma for gaussian if to large!")
        self.gsigma = gsigma

        self.conv_x_length =  self.gsigma*6.0
        self.conv_x = self.small_x[self.small_x < self.conv_x_length]
        self.conv_x_size = len(self.conv_x)
        self.conv_gauss = np.exp(-np.power((3.0*self.gsigma - self.conv_x), 2)/(2.0*np.power(self.gsigma,2)))

        if False:
            #self.conv_ax = plt.subplot2grid((2, self.size_mul), (0, self.size_mul-1), colspan=1)
            self.conv_ax = plt.subplot2grid((2, self.size_mul), (0, 1), colspan=self.size_mul-1)
            self.conv_ax.set_xlim([0, 1.0])
            self.conv_ax.plot(self.conv_x, self.conv_gauss)

    def run_epoch(self, epoch_size=200):
        for i in np.random.randint(self.small_size, size=epoch_size):
            self.small_y[i] += 1.0

        for i in np.random.randint(self.large_size, size=epoch_size*self.size_mul):
            self.large_y[i] += 1.0

    def show(self, conv=False):
        if conv:
            small_y_ = np.append(self.small_y, self.small_y[:self.conv_x_size])
            small_y_ = np.append(self.small_y[-self.conv_x_size:] ,small_y_)

            small_y_conv = np.convolve(small_y_, self.conv_gauss, mode='same')
            small_y_conv = small_y_conv[self.conv_x_size:-self.conv_x_size]

            large_y_ = np.append(self.large_y, self.large_y[:self.conv_x_size])
            large_y_ = np.append(self.large_y[-self.conv_x_size:], large_y_)

            large_y_conv = np.convolve(large_y_, self.conv_gauss, mode='same')
            large_y_conv = large_y_conv[self.conv_x_size:-self.conv_x_size]

            #self.small_ax.plot(self.small_x, small_y_conv)
            #self.large_ax.plot(self.large_x, large_y_conv)
            small_y_conv_ext = small_y_conv
            for i in range(self.size_mul-1):
                small_y_conv_ext = np.append(small_y_conv_ext, small_y_conv)
            self.small_ax.set_xlim((0, self.size_mul))
            self.small_ax.plot(self.large_x, small_y_conv_ext)
            self.large_ax.plot(self.large_x, large_y_conv)

            maxx = max(np.max(small_y_conv), np.max(large_y_conv))
            minn = min(np.min(small_y_conv), np.min(large_y_conv))
        else:
            self.small_ax.plot(self.small_x, self.small_y)
            self.large_ax.plot(self.large_x, self.large_y)

            maxx = max(np.max(self.small_y), np.max(self.large_y))
            minn = min(np.min(self.small_y), np.min(self.large_y))
        self.small_ax.set_ylim((minn, maxx))
        self.large_ax.set_ylim((minn, maxx))

        plt.show()

rv = RandomVisual(50, gsigma=0.12)
rv.run_epoch(1000)
rv.show(True)
