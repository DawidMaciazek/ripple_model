import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import sys
import numpy as np

np.seterr(all='warn')

import warnings
warnings.filterwarnings('error')

def yamamura(theta, theta_opt, f):
    try:
        return -np.power(np.cos(theta), -f)*np.exp(f * (1-np.power(np.cos(theta), -1)) * np.cos(theta_opt)  )
    except Warning:
        print "-np.power(np.cos(theta), -f)", np.cos(theta), -f
        sys.exit(0)

#x = np.linspace(0,1.5)
#y = yamamura(x, 70/57.3, 1.9)


class Solver:
    def __init__(self, angle, length, points):
        self.yamp = 0.01

        self.angle = np.radians(angle)

        self.length = length
        self.points = points
        self.points_sep = length/float(points-1)

        self.boundary_cond = "0"


        self.x = np.linspace(0, length, num=points)
        self.y = np.full(points, 0, dtype=float)

        self.frames = []

    def align_z(self):
        self.y -= np.average(self.y)

    def add_normal_noise(self, sigma):
        normal = np.random.normal(0, sigma, self.points)
        for i in xrange(self.points):
            self.y[i] += normal[i]

    def sin_distortion(self, amp, period):
        self.y += amp*np.sin(self.x/period)
        self.align_z()

    def show(self, yspace=[-2, 2], zoom=None):

        if zoom:
            zi_start = int(zoom[0]*self.points)
            zi_end = int(zoom[1]*(self.points-1))
        else:
            zi_start = 0
            zi_end = self.points - 1

        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)

        plt.axis([self.x[zi_start], self.x[zi_end] , yspace[0], yspace[1]])
        if zoom:
            l, = plt.plot(self.x[zi_start:zi_end], self.y[zi_start:zi_end], 'r+')
        else:
            l, = plt.plot(self.x[zi_start:zi_end], self.y[zi_start:zi_end])

        axslider = plt.axes([0.15, 0.1, 0.65, 0.05])
        slider = Slider(axslider, 'Tmp', 0, len(self.frames)-1, valinit=0)

        def update(val):
            sel = self.frames[int(val)][zi_start:zi_end]
            sel -= np.average(sel)
            l.set_ydata(sel)
        slider.on_changed(update)
        plt.show()

    def calc_step(self):
        yamp = self.yamp
        y = self.y
        points_sep = self.points_sep
        angle = self.angle

        max_angle = np.pi/2.0 - 0.01
        pairs = np.empty(self.points-1, dtype=float)

        #yamamura(theta, theta_opt, f):
        for i in xrange(self.points-1):
            langle = np.arctan((y[i+1]-y[i])/points_sep)
            fangle = langle + angle
            if fangle > max_angle:
                fangle = max_angle
            pairs[i] = yamp*yamamura(fangle, np.radians(67), 1.95)

        for i in xrange(self.points-1):
            y[i] += pairs[i]
            y[i+1] += pairs[i]

        y[0] += pairs[0]
        y[-1] += pairs[-1]

        # b-condition
        #self.bcondition()

        self.frames.append(np.copy(self.y))

    def run(self, num=100):
        for i in xrange(num):
            self.calc_step()



solver = Solver(67, 500, 4000)
solver.sin_distortion(1, 20)
solver.add_normal_noise(0.02)
solver.run(2000)
solver.show()
solver.show(zoom=[0,0.1])
#solver.show(zoom=[0.98,1])
#solver.show()

