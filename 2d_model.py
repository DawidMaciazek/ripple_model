import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math
from matplotlib.widgets import Slider

import sys
import numpy as np

np.seterr(all='warn')

import warnings
warnings.filterwarnings('error')

def yamamura(theta, theta_opt, f):
    try:
        return np.power(np.cos(theta), -f)*np.exp(f * (1-np.power(np.cos(theta), -1)) * np.cos(theta_opt)  )
    except Warning:
        print "np.power(np.cos(theta), -f)", np.cos(theta), -f
        sys.exit(0)

#x = np.linspace(0,1.5)
#y = yamamura(x, 70/57.3, 1.9)


class Solver:
    def __init__(self, angle, length, points):
        self.yamp = 0.01
        self.thetao = np.radians(77)

        self.angle = np.radians(angle)

        self.length = length
        self.points = points
        self.points_sep = length/float(points-1)

        self.boundary_cond = "0"


        self.x = np.linspace(0, length, num=points)
        self.y = np.full(points, 0, dtype=float)

        self.frames = []
        self.py_frames = []
        self.angle_frames = []

    def align_z(self):
        self.y -= np.average(self.y)

    def add_normal_noise(self, sigma):
        normal = np.random.normal(0, sigma, self.points)
        for i in xrange(self.points):
            self.y[i] += normal[i]

    def sin_distortion(self, amp, period):
        self.y += amp*np.sin(self.x/period)
        self.align_z()


    def gauss_distortion(self, amp, mu, sigma):
        self.y += amp*mlab.normpdf(self.x ,mu, sigma)


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
        ll = None
        lll = None
        if zoom:
            l, ll, lll = plt.plot(self.x[zi_start:zi_end], self.y[zi_start:zi_end], 'r+',
                                  self.x[zi_start:zi_end], self.y[zi_start:zi_end], 'bx',
                                  self.x[zi_start:zi_end], self.y[zi_start:zi_end], 'g.')
        else:
            l, = plt.plot(self.x[zi_start:zi_end], self.y[zi_start:zi_end])

        axslider = plt.axes([0.15, 0.1, 0.65, 0.05])
        slider = Slider(axslider, 'Tmp', 0, len(self.frames)-1, valinit=0)

        def update(val):
            sel = self.frames[int(val)][zi_start:zi_end]
            sel -= np.average(sel)
            l.set_ydata(sel)
            if(ll):
                sel_y = self.py_frames[int(val)][zi_start:zi_end]/(max(self.py_frames[int(val)][zi_start:zi_end]))
                sel_y -= np.average(sel_y)
                ll.set_ydata(sel_y)

            if(lll):
                sel_a = self.angle_frames[int(val)][zi_start:zi_end]*(0.5729578*4)-2
                lll.set_ydata(sel_a)


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
                print "over max:", fangle * 57.3
                print "result::", yamamura(fangle, self.thetao, 1.95)
                fangle = max_angle
            pairs[i] = yamp*yamamura(fangle, self.thetao, 1.95)

        for i in xrange(self.points-1):
            y[i] += pairs[i]
            y[i+1] += pairs[i]

        y[0] += pairs[0]
        y[-1] += pairs[-1]

        self.frames.append(np.copy(self.y))

    def calc_step_ave(self):
        yamp = self.yamp
        y = self.y
        points_sep = self.points_sep
        pair_sep = points_sep*2
        angle = self.angle

        max_angle = np.pi/2.0 - 0.01

        partial_yield = np.empty(self.points, dtype=float)
        angles = np.empty(self.points, dtype=float)

        for i in xrange(1, self.points-1):
            langle = np.arctan((y[i+1]-y[i-1])/(2*pair_sep))
            fangle = -langle + angle
            if fangle > max_angle:
                print "WARNING !:", fangle
                fangle = max_angle
                print "result::", yamamura(fangle, self.thetao, 1.95)
            partial_yield[i] = yamp*yamamura(fangle, self.thetao, 1.95)
            angles[i] = fangle

        langle = np.arctan(2*((y[1]-y[0])/pair_sep))
        partial_yield[0] = yamp*yamamura(langle + angle, self.thetao, 1.95)
        angles[0] = -langle + angle

        langle = np.arctan(2*((y[-1]-y[-2])/pair_sep))
        partial_yield[-1] = yamp*yamamura(langle + angle, self.thetao, 1.95)
        angles[-1] = -langle + angle

        for i in xrange(self.points):
            y[i] -= partial_yield[i]

        self.frames.append(np.copy(self.y))
        self.py_frames.append(partial_yield)
        self.angle_frames.append(angles)


    def calc_step_4(self):
        yamp = self.yamp
        y = self.y
        points_sep = self.points_sep
        pair_sep = points_sep*2
        angle = self.angle

        max_angle = np.pi/2.0 - 0.01

        partial_yield = np.empty(self.points, dtype=float)
        angles = np.empty(self.points, dtype=float)
        coef = [1.0/12, -2.0/3, 2.0/3, -1.0/12]

        for i in xrange(2, self.points-2):
            tangle = y[i-2]*coef[0] + y[i-1]*coef[1] + y[i+1]*coef[2] + y[i+2]*coef[3]
            langle = np.arctan(tangle/pair_sep)
            fangle =  angle - langle
            if fangle > max_angle:
                print "WARNING !:", fangle
                fangle = max_angle
                print "result::", yamamura(fangle, self.thetao, 1.95)
            partial_yield[i] = yamp*yamamura(fangle, self.thetao, 1.95)
            angles[i] = fangle

        angles[0] = angles[2]
        angles[1] = angles[2]
        angles[-1] = angles[-3]
        angles[-2] = angles[-3]


        for i in xrange(self.points):
            y[i] -= partial_yield[i]
        y[0]=y[1] = y[2]
        y[-1]=y[-2] = y[-3]

        self.frames.append(np.copy(self.y))
        self.py_frames.append(partial_yield)
        self.angle_frames.append(angles)

    def run(self, num=100, mode=0):
        for i in xrange(num):
            if mode == 0:
                self.calc_step()
            elif mode == 1:
                self.calc_step_4()
            else:
                self.calc_step_ave()



solver = Solver(67, 500, 2000)
#solver.sin_distortion(1, 20)
solver.gauss_distortion(100, 250, 20)
solver.add_normal_noise(0.001)
solver.run(500, 2)
solver.show()
solver.show(zoom=[0.5,0.6])
#solver.show(zoom=[0.08,0.1])
#solver.show(zoom=[0.1,0.3])
#solver.show(zoom=[0.8,1])
#solver.show(zoom=[0.98,1])
#solver.show()

x = np.linspace(0,np.pi*0.499 , num=100)
y = yamamura(x, np.radians(77), 1.94)

#plt.plot(x, y)
#plt.show()

