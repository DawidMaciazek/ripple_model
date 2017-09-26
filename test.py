import numpy as np
import matplotlib.pyplot as plt
import sys

def gauss(x, sigma, d):
    return np.exp(- np.power(x-d,2)/(2.0*np.power(sigma,2)))

class model:
    def __init__(self):
        self.x_len = 100
        self.x = np.linspace(0, self.x_len, self.x_len, endpoint=False)
        self.y = np.zeros(self.x_len)
        self.gauss_amp = 0.03

    def show(self):
        plt.plot(self.x, self.y)
        plt.show()

    def add_sin(self, amp=10, n=2):
        self.y -= amp*np.sin(n*2*np.pi*(self.x/self.x_len))

    def calc(self, sigma, d1, d2):
        self.z_len = int(sigma*3 + d2)
        self.z = np.linspace(0, self.z_len, int(self.z_len),endpoint=False)
        self.displ = self.gauss_amp*(gauss(self.z, sigma, d2)*1.0-1.2*gauss(self.z,sigma, d1))
        self.z = -self.z
        plt.plot(self.z, self.displ)
        plt.show()
        self.y_res = np.zeros(self.x_len, dtype=float)

        for node in range(self.x_len):
            depths = self.y[node] + self.z
            x0 = self.x[node]
            for i in range(self.z_len):
                value = self.displ[i]
                depth = depths[i]

                dist_list = np.power(self.x - x0,2) + np.power(self.y-depth, 2)
                index = np.argmin(dist_list)
                self.y_res[index] += value
        plt.plot(self.x, self.y_res)
        plt.show()
        plt.plot(self.x, self.y)
        plt.plot(self.x, self.y+self.y_res)
        plt.show()
        self.y += self.y_res

m = model()
m.y-=m.x*2.0
m.add_sin()
for i in range(20):
    m.calc(10.,25.,56.)
