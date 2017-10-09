import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

class diff_model:
    def __init__(self, diff=0.1, nodes=50, sample_len=50, sigma=10):
        self.sample_len = sample_len
        self.sigma = sigma
        self.diff = diff
        self.nodes = nodes

        self.x = np.linspace(0, sample_len, nodes, endpoint=False)
        self.dx = self.x[1]
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
        self.y_history.append(self.y.copy())
        #self.conv_gauss = np.

    def add_cos(self, amp, n):
        self.y += amp*np.cos(2*n*np.pi*(self.x/self.sample_len))
        self.y_history.append(self.y.copy())

    def add_noise(self, amp):
        self.y += np.random.normal(0, amp, self.y.shape)
        self.y_history.append(self.y.copy())

    def add_spike(self, amp):
        sect_start = int(self.nodes/3.0)
        sect_mid = int(sect_start/2.0)

        self.y[sect_start:sect_mid+sect_start] += amp*np.linspace(0,1, sect_mid)
        self.y[sect_mid+sect_start:sect_mid*2+sect_start] += amp*np.linspace(1,0, sect_mid)
        self.y_history.append(self.y.copy())


    def erosion_gauss(self, amp, sigma):
        g = amp*np.exp(-np.power(self.x-self.sample_len/2.0, 2)/(2*np.power(sigma, 2)))
        self.y -= np.roll(g, random.randint(0, self.nodes-1))

    def single_conv(self):
        #self.y += self.diff*(self.y-np.convolve(np.pad(self.y, (self.gauss_half, self.gauss_half), mode='wrap'), self.gauss, mode='valid'))
        self.y_history.append(self.y.copy())
        self.y = np.convolve(np.pad(self.y, (self.gauss_half, self.gauss_half), mode='wrap'), self.gauss, mode='valid')
        #plt.plot(self.x, oldy)
        #plt.plot(self.x, self.y)
        #plt.show()

    def angle_energy(self, angles):
        return np.tan((angles - np.pi)/2.0)

    def single_energy(self):
        #1 calc local energy

        dy = np.diff(np.pad(self.y, (0,1), mode='wrap'))
        l_angles = np.arctan(dy/self.dx)

        # 180 + (alpha + beta)
        node_angles = np.pi + (np.roll(l_angles, 1) - l_angles)

        node_energy = self.angle_energy(node_angles)

        forward_transport = np.roll(node_energy, -1) - node_energy
        backward_transport = -np.roll(forward_transport, 1)

        total_transport = forward_transport+backward_transport

        # add angle correction
        self.y += self.diff * total_transport # / np.cos(l_angles)
        self.y -= np.mean(self.y)
        self.y_history.append(self.y.copy())

    def single_normal(self):
        self.y_history.append(self.y.copy())
        self.y += self.diff*np.diff(np.pad(self.y, (1,1), mode='wrap'), 2)

    def single_normal4(self):
        self.y_history.append(self.y.copy())
        self.y += self.diff*np.diff(np.pad(self.y, (2,2), mode='wrap'), 4)


    def show(self):
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)

        #plt.axis([self.x[0], self.x[-1], -0.2, 0.2])

        plot_current, = plt.plot(self.x, self.y_history[-1])

        axslider = plt.axes([0.15, 0.1, 0.65, 0.05])
        slider = Slider(axslider, 'Tmp', 0, len(self.y_history)-1, valinit=0)

        def update(val):
            #print "choosen {}/{}".format(int(val), len(self.y_history))
            selection = self.y_history[int(val)]
            plot_current.set_ydata(selection)

        slider.on_changed(update)
        plt.show()



rand = 0.05
m = diff_model(nodes=500, sample_len=500,diff=0.1)

cont = int(input("Cont:"))
erosion = 0.1
while cont:
    m.erosion_gauss(erosion, 3)
    cont -= 1
    for i in range(20):
        m.single_energy()

    if(cont == 0):
        m.show()
        cont = int(input("Cont:"))
        erosion = 0.0


    #m.single_normal()
    #m.single_normal4()
    #m.single_normal()
