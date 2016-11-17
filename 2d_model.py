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
        self.max_h = 2
        self.min_h = -2

        self.angle = np.radians(angle)

        self.length = length
        self.points = points
        self.points_sep = length/float(points-1)


        self.x = np.linspace(0, length, num=points)
        self.y = np.full(points, 0, dtype=float)

        self.frames = []

    def align_z(self):
        self.y -= np.average(self.y)

    def random_distortion(self, val):
        pass

    def sin_distortion(self, amp, period):
        self.y += amp*np.sin(self.x/period)
        self.align_z()

    def show(self):
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)

        plt.axis([0, self.length, self.min_h, self.max_h])
        l, = plt.plot(self.x, self.y)

        axslider = plt.axes([0.15, 0.1, 0.65, 0.05])
        slider = Slider(axslider, 'Tmp', 0, len(self.frames)-1, valinit=0)

        def update(val):
            l.set_ydata(self.frames[int(val)])
        slider.on_changed(update)
        plt.show()

    def calc_step(self):
        yamp = self.yamp
        y = self.y
        points_sep = self.points_sep
        angle = self.angle

        max_angle = np.pi/2.0 - 0.01

        #yamamura(theta, theta_opt, f):
        for i in xrange(self.points-1):
            langle = np.arctan((y[i+1]-y[i])/points_sep)
            fangle = langle + angle
            if fangle > max_angle:
                fangle = max_angle
            lyield = yamp*yamamura(fangle, np.radians(67), 1.95)


            y[i] += lyield
            y[i+1] += lyield

        byield = yamp*yamamura(angle, np.radians(67), 1.95)
        y[0] += byield
        y[self.points-1] += byield

        self.align_z()

        self.frames.append(np.copy(self.y))

    def run(self, num=100):
        for i in xrange(num):
            self.calc_step()



solver = Solver(67, 100, 1000)
solver.sin_distortion(1, 15)
solver.run(500)
solver.show()
#solver.show()

