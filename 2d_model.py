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
    def __init__(self, angle, length, points, yamp=0.01, d=0.01):
        self.dist_traveled = 0.0
        self.d = d
        self.yamp = yamp
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
        ave = np.average(self.y)
        self.y -= ave

    def add_normal_noise(self, sigma):
        normal = np.random.normal(0, sigma, self.points)
        for i in xrange(self.points):
            self.y[i] += normal[i]

    def sin_distortion(self, amp, period):
        self.y += amp*np.sin(self.x/period)
        self.align_z()


    def gauss_distortion(self, amp, mu, sigma):
        self.y += amp*mlab.normpdf(self.x ,mu, sigma)

    def triangle_distortion(self, amp, start, end):
        start = float(start)
        end = float(end)
        amp = float(amp)
        mid = start + (end - start)/2
        sel_up = np.logical_and(self.x>start, self.x<=mid)
        sel_down = np.logical_and(self.x>mid, self.x<end)

        a_up = (amp)/(mid-start)
        b_up = -start*(amp)/(mid-start)

        a_down = (-amp)/(end-mid)
        b_down = amp-mid*(-amp)/(end-mid)
        print "TRIANGLE"
        print "start:{}, mid:{}, end{}, amp{}".format(start, mid, end, amp)
        for i, v in enumerate(sel_up):
            if v:
                self.y[i] += a_up*self.x[i] + b_up

        for i, v in enumerate(sel_down):
            if v:
                self.y[i] += a_down*self.x[i] + b_down

    def show(self, yspace=[-2, 2], zoom=None):
        print "TOTAL DISTANCE TRAVELED:"
        print np.average(self.y)
        print "-----------------------"

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


    def calc_step_4_p(self):
        yamp = self.yamp
        y = self.y
        d = self.d
        points_sep = self.points_sep
        pair_sep = points_sep*2
        angle = self.angle

        max_angle = np.pi/2.0 - 0.01

        partial_yield = np.empty(self.points, dtype=float)
        diffusion = np.empty(self.points, dtype=float)
        angles = np.empty(self.points, dtype=float)
        coef  = [1.0/12, -2.0/3, 2.0/3, -1.0/12]

        dcoef = [-1.0/12, 4.0/3, -5.0/2.0, 4.0/3, -1.0/12]

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

            dangle = y[i-2]*dcoef[0] + y[i-1]*dcoef[1] + y[i]*dcoef[2] + y[i+1]*dcoef[3] + y[i+2]*dcoef[4]
            diffusion[i] = d*(dangle/(pair_sep*pair_sep))

        bcon = {0:[-2,-1,1,2], 1:[-1,0,2,3],
                -1:[-3,-2,0,1], -2:[-4,-3, -1, 0]}
        for i in bcon:
            ilist = bcon[i]
            tangle = y[ilist[0]]*coef[0] + y[ilist[1]]*coef[1] + y[ilist[2]]*coef[2] + y[ilist[3]]*coef[3]
            langle = np.arctan(tangle/pair_sep)
            fangle = angle - langle

            if fangle > max_angle:
                print "WARNING !:", fangle
                fangle = max_angle
                print "BONDARY RESULT::", yamamura(fangle, self.thetao, 1.95)

            partial_yield[i] = yamp*yamamura(fangle, self.thetao, 1.95)
            angles[i] = fangle

            dangle = y[ilist[0]]*dcoef[0] + y[ilist[1]]*dcoef[1] + y[i]*dcoef[2] + y[ilist[2]]*dcoef[3] + y[ilist[3]]*dcoef[4]
            diffusion[i] = d*(dangle/(pair_sep*pair_sep))






        for i in xrange(self.points):
            y[i] -= partial_yield[i]
            y[i] += diffusion[i]

        self.frames.append(np.copy(self.y))
        self.py_frames.append(partial_yield)
        self.angle_frames.append(angles)

    def calc_step_6(self):
        yamp = self.yamp
        y = self.y
        points_sep = self.points_sep
        pair_sep = points_sep*2
        angle = self.angle

        max_angle = np.pi/2.0 - 0.01

        partial_yield = np.empty(self.points, dtype=float)
        angles = np.empty(self.points, dtype=float)
        coef = [-1.0/60, 3.0/20, -3.0/4, 3.0/4,  -3.0/20, 1.0/60]

        for i in xrange(3, self.points-3):
            tangle = y[i-3]*coef[0] + y[i-2]*coef[1] + y[i-1]*coef[2] + y[i+1]*coef[3] + y[i+2]*coef[4] + y[i+3]*coef[5]
            langle = np.arctan(tangle/pair_sep)
            fangle =  angle - langle
            if fangle > max_angle:
                print "WARNING !:", fangle
                fangle = max_angle
                print "result::", yamamura(fangle, self.thetao, 1.95)
            partial_yield[i] = yamp*yamamura(fangle, self.thetao, 1.95)
            angles[i] = fangle

        angles[0] = angles[1] = angles[2]  = angles[3]
        angles[-1] = angles[-2] = angles[-3]  = angles[-4]


        for i in xrange(self.points):
            y[i] -= partial_yield[i]
        y[0] = y[1] = y[2] = y[3]
        y[-1]=y[-2] = y[-3] = y[4]

        self.frames.append(np.copy(self.y))
        self.py_frames.append(partial_yield)
        self.angle_frames.append(angles)

    def run(self, num=100, mode=0):
        for i in xrange(num):
            if mode == 0:
                self.calc_step()
            elif mode == 1:
                self.calc_step_4()
            elif mode == 2:
                self.calc_step_6()
            elif mode == 3:
                self.calc_step_4_p()
            else:
                self.calc_step_ave()



solver = Solver(67, 500, 200, yamp=0.012, d=0.0035)
#solver.sin_distortion(1, 20)
solver.gauss_distortion(25, 365, 25)

solver.triangle_distortion(1, 200, 400)
solver.gauss_distortion(-50, 400, 15)
solver.gauss_distortion(-50, 50, 15)
solver.gauss_distortion(50, 150, 15)
solver.add_normal_noise(0.01)

c = 10000
while c!=0:
    solver.run(c, 3)
    solver.show()
    c = input("continue :")



#solver.show(zoom=[0.3,0.7])

