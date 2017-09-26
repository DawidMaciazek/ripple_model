import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

class conv_diff:
    def __init__(self, diff=0.1, nodes=500, sample_len=500, sigma=10):
        self.sample_len = sample_len
        self.sigma = sigma
        self.diff = diff
        self.nodes = nodes

        self.x = np.linspace(0, sample_len, nodes, endpoint=False)
        self.y = np.zeros(self.x.shape, dtype=float)

        gauss_len = sigma*4
        gauss_x = self.x[self.x<gauss_len]
        if len(gauss_x) % 2 == 0:
            gauss_x = self.x[:len(gauss_x)+1]
        self.gauss_half = int(len(gauss_x)/2)
        x0 = gauss_x[self.gauss_half]
        self.gauss = np.exp(-np.power(gauss_x-x0,2)/(2*np.power(self.sigma,2)))
        self.gauss = self.gauss/np.sum(self.gauss)

        self.y_history = []
        #self.conv_gauss = np.

    def add_cos(self, amp, n):
        self.y_history.append(self.y.copy)
        self.y += amp*np.cos(2*n*np.pi*(self.x/self.sample_len))


    def add_noise(self, amp):
        self.y_history.append(self.y.copy)
        self.y += np.random.normal(0, amp, self.y.shape)

    def single_run(self):

        #self.y += self.diff*(self.y-np.convolve(np.pad(self.y, (self.gauss_half, self.gauss_half), mode='wrap'), self.gauss, mode='valid'))
        self.y_history.append(self.y.copy())
        self.y = np.convolve(np.pad(self.y, (self.gauss_half, self.gauss_half), mode='wrap'), self.gauss, mode='valid')
        #plt.plot(self.x, oldy)
        #plt.plot(self.x, self.y)
        #plt.show()

    def show(self):
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)

        #plt.axis([self.x[0], self.x[-1], -0.2, 0.2])

        plot_current, = plt.plot(self.x, self.y_history[0])

        axslider = plt.axes([0.15, 0.1, 0.65, 0.05])
        slider = Slider(axslider, 'Tmp', 0, len(self.y_history)-1, valinit=0)

        def update(val):
            #print "choosen {}/{}".format(int(val), len(self.y_history))
            selection = self.y_history[int(val)]
            plot_current.set_ydata(selection)

        slider.on_changed(update)
        plt.show()



conv = conv_diff()
conv.add_cos(10, 3)
conv.add_noise(2)
for i in range(10):
    conv.single_run()
conv.show()

