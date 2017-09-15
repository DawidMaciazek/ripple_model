import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math
from matplotlib.widgets import Slider
import random

import sys
import numpy as np

GLOBAL_COEFF=-10.0
np.seterr(all='warn')

import warnings
warnings.filterwarnings('error')

def yamamura(theta, theta_opt, f):
    try:
        return np.power(np.cos(theta), -f+1)*np.exp(f * (1-np.power(np.cos(theta), -1)) * np.cos(theta_opt)  )
    except Warning:
        print("Shit in the fans!")
        theta[theta > 1.5607] =1.5607
        return np.power(np.cos(theta), -f+1)*np.exp(f * (1-np.power(np.cos(theta), -1)) * np.cos(theta_opt)  )
        sys.exit(0)

#x = np.linspace(0,1.5)
#y = yamamura(x, 70/57.3, 1.9)


class Solver:
    def __init__(self, angle, length, points, yamp=0.01, d=0.01, fval=2.4, thetao=72.7, gauss_sigma=7):
        self.fval = fval
        self.dist_traveled = 0.0
        self.d = d
        self.yamp = yamp
        self.thetao = np.radians(thetao)

        self.angle = np.radians(angle)

        self.length = length
        self.points = points
        self.points_sep = length/float(points)

        self.boundary_cond = "0"

        self.lfrog_init = False

        self.x = np.linspace(0, length, num=points, endpoint=False)
        self.y = np.full(points, 0, dtype=float)
        self.py = np.full(points, 0, dtype=float)

        self.gauss_arr = np.zeros((self.points, self.points), dtype=float)
        for i in range(self.points):
            self.gauss_arr[i] = np.exp(-np.power(np.roll(self.x, i)-self.length/2.0 ,2)/(2.0*gauss_sigma*gauss_sigma)  )

        #plt.imshow(self.gauss_arr)
        #plt.show()


        self.frames = []
        self.py_frames = []
        self.angle_frames = []

    def align_z(self):
        ave = np.average(self.y)
        self.y -= ave

    def add_normal_noise(self, sigma):
        normal = np.random.normal(0, sigma, self.points)
        for i in range(self.points):
            self.y[i] += normal[i]

    def sin_distortion(self, amp, rep=1):
        pi2 = np.pi * 2
        self.y += amp*np.sin((rep*pi2*self.x)/self.length)
        self.align_z()


    def zigzag_distortion(self, amp):
        rep = (self.points)/4.0
        pi2 = np.pi * 2
        self.y += amp*np.sin((rep*pi2*self.x)/(self.length + self.length/float(self.points)))
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
        print("TRIANGLE")
        print("start:{}, mid:{}, end{}, amp{}".format(start, mid, end, amp))
        for i, v in enumerate(sel_up):
            if v:
                self.y[i] += a_up*self.x[i] + b_up

        for i, v in enumerate(sel_down):
            if v:
                self.y[i] += a_down*self.x[i] + b_down

    def show(self, yspace=[-2, 2], zoom=None, center=True):
        print("TOTAL DISTANCE TRAVELED:")
        print(np.average(self.y))
        print("-----------------------")

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
            l, lrc, ll, yl, lll, aplhha, lyield = plt.plot(self.x[zi_start:zi_end], self.y[zi_start:zi_end], 'r',
                                  self.x[zi_start:zi_end], self.y[zi_start:zi_end], 'r+',
                                  self.x[zi_start:zi_end], self.y[zi_start:zi_end], 'b-',
                                  self.x[zi_start:zi_end], self.y[zi_start:zi_end], 'y-',
                                  self.x[zi_start:zi_end], self.y[zi_start:zi_end], 'g.',
                                  self.x[zi_start:zi_end], self.y[zi_start:zi_end], 'g+',
                                  self.x[zi_start:zi_end], self.y[zi_start:zi_end], 'm.')
        else:
            l, = plt.plot(self.x[zi_start:zi_end], self.y[zi_start:zi_end])

        axslider = plt.axes([0.15, 0.1, 0.65, 0.05])
        slider = Slider(axslider, 'Tmp', 0, len(self.frames)-1, valinit=0)

        def update(val):
            sel = self.frames[int(val)][zi_start:zi_end]
            if center:
                sel -= np.average(sel)
            l.set_ydata(sel)

            if(False):
                sel_y = self.py_frames[int(val)][zi_start:zi_end]/(max(self.py_frames[int(val)][zi_start:zi_end]))
                sel_y -= np.average(sel_y)
                ll.set_ydata(sel_y)

            if(lll):
                lrc.set_ydata(sel)
                a = (1)*2*np.abs(yspace[0])/np.pi
                sel_a = self.angle_frames[int(val)][zi_start:zi_end]*a+yspace[0]*1
                lll.set_ydata(sel_a)
                sel_apha = (self.angle-self.angle_frames[int(val)][zi_start:zi_end])*a
                aplhha.set_ydata(sel_apha)

                sel_yield = yamamura(self.angle_frames[int(val)][zi_start:zi_end], self.thetao, self.fval) * (yspace[1]/5.0)
                lyield.set_ydata(sel_yield)


                b = np.radians(72.7)*a + yspace[0]
                sel_b = np.full(len(sel_a), b)
                ll.set_ydata(sel_b)

                c = np.radians(45.0)*a + yspace[0]
                sel_c = np.full(len(sel_a), c)
                yl.set_ydata(sel_c)


        slider.on_changed(update)
        plt.show()

    def calc_step(self):
        yamp = self.yamp
        y = self.y
        points_sep = self.points_sep
        angle = self.angle

        max_angle = np.pi/2.0 - 0.001
        pairs = np.empty(self.points, dtype=float)
        angles = np.empty(self.points, dtype=float)

        for i in range(self.points-1):
            langle = np.arctan((y[i+1]-y[i])/points_sep)
            fangle = -langle + angle

            if fangle > max_angle:
                print("over max:", fangle * 57.3)
                fangle = max_angle - 0.1
            pairs[i] = yamp*yamamura(fangle, self.thetao, self.fval)
            angles[i] = fangle

        langle = np.arctan((y[0]-y[-1])/points_sep)
        fangle = -langle + angle
        pairs[-1] = yamp*yamamura(fangle, self.thetao, self.fval)

        for i in range(self.points-1):
            y[i] -= pairs[i]*0.5
            y[i+1] -= pairs[i]*0.5
        y[0] -= pairs[-1]*0.5
        y[-1] -= pairs[-1]*0.5



        self.frames.append(np.copy(self.y))
        self.angle_frames.append(angles)
        self.py_frames.append(pairs)


    def calc_step_ave(self):
        yamp = self.yamp
        y = self.y
        points_sep = self.points_sep
        angle = self.angle

        max_angle = np.pi/2.0 - 0.001
        pairs = np.empty(self.points, dtype=float)
        angles = np.empty(self.points, dtype=float)

        normal = np.random.normal(1, 0.0004, self.points)
        for i in range(self.points-1):
            langle = np.arctan((y[i+1]-y[i])/points_sep)
            fangle = -langle + angle

            if fangle > max_angle:
                print("over max:", fangle * 57.3)
                fangle = max_angle - 0.1
            pairs[i] = normal[i]*yamp*yamamura(fangle, self.thetao, self.fval)
            angles[i] = fangle

        langle = np.arctan((y[0]-y[-1])/points_sep)
        fangle = -langle + angle
        pairs[-1] = yamp*yamamura(fangle, self.thetao, self.fval)*normal[i]

        #w = [0.15, 0.35, 0.35, 0.15]
        #w = [0.1, 0.3, 0.4, 0.20]
        #w = [0.05, 0.1, 0.25, 0.3, 0.2, 0.1]
        w = [0.05, 0.15, 0.3, 0.3, 0.15, 0.05]
        for i in range(-3, len(pairs)-3):
            p = pairs[i]
            y[i-2] -= w[0]*p
            y[i-1] -= w[1]*p
            y[i] -= w[2]*p
            y[i+1] -= w[3]*p
            y[i+2] -= w[4]*p
            y[i+3] -= w[5]*p





        self.frames.append(np.copy(self.y))
        self.angle_frames.append(angles)
        self.py_frames.append(pairs)

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

        for i in range(2, self.points-2):
            tangle = y[i-2]*coef[0] + y[i-1]*coef[1] + y[i+1]*coef[2] + y[i+2]*coef[3]
            langle = np.arctan(tangle/pair_sep)
            fangle =  angle - langle
            if fangle > max_angle:
                print("WARNING !:", fangle)
                fangle = max_angle
                print("result::", yamamura(fangle, self.thetao, self.fval))
            partial_yield[i] = yamp*yamamura(fangle, self.thetao, self.fval)
            angles[i] = fangle

        angles[0] = angles[2]
        angles[1] = angles[2]
        angles[-1] = angles[-3]
        angles[-2] = angles[-3]


        for i in range(self.points):
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
        pair_sep = points_sep
        angle = self.angle

        max_angle = np.pi/2.0 - 0.01

        partial_yield = np.empty(self.points, dtype=float)
        diffusion = np.empty(self.points, dtype=float)
        angles = np.empty(self.points, dtype=float)
        coef  = [1.0/12, -2.0/3, 2.0/3, -1.0/12]

        dcoef = [-1.0/12, 4.0/3, -5.0/2.0, 4.0/3, -1.0/12]

        for i in range(2, self.points-2):
            tangle = y[i-2]*coef[0] + y[i-1]*coef[1] + y[i+1]*coef[2] + y[i+2]*coef[3]
            langle = np.arctan(tangle/pair_sep)
            fangle =  angle - langle
            if fangle > max_angle:
                print("WARNING !:", fangle)
                fangle = max_angle
                print("result::", yamamura(fangle, self.thetao, self.fval))
            dangle = y[i-2]*dcoef[0] + y[i-1]*dcoef[1] + y[i]*dcoef[2] + y[i+1]*dcoef[3] + y[i+2]*dcoef[4]
            if i == 10:
                print(np.exp(GLOBAL_COEFF*dangle))
            partial_yield[i] = yamp*yamamura(fangle, self.thetao, self.fval)*np.exp(GLOBAL_COEFF*dangle)
            angles[i] = fangle

            diffusion[i] = d*(dangle/(pair_sep*pair_sep))

        bcon = {0:[-2,-1,1,2], 1:[-1,0,2,3],
                -1:[-3,-2,0,1], -2:[-4,-3, -1, 0]}
        for i in bcon:
            ilist = bcon[i]
            tangle = y[ilist[0]]*coef[0] + y[ilist[1]]*coef[1] + y[ilist[2]]*coef[2] + y[ilist[3]]*coef[3]
            langle = np.arctan(tangle/pair_sep)
            fangle = angle - langle

            if fangle > max_angle:
                print("WARNING !:", fangle)
                fangle = max_angle
                print("BONDARY RESULT::", yamamura(fangle, self.thetao, self.fval))

            dangle = y[ilist[0]]*dcoef[0] + y[ilist[1]]*dcoef[1] + y[i]*dcoef[2] + y[ilist[2]]*dcoef[3] + y[ilist[3]]*dcoef[4]
            partial_yield[i] = yamp*yamamura(fangle, self.thetao, self.fval)*np.exp(GLOBAL_COEFF*dangle)

            angles[i] = fangle

            diffusion[i] = d*(dangle/(pair_sep*pair_sep))

        for i in range(self.points):
            y[i] -= partial_yield[i]
            y[i] += diffusion[i]

        self.frames.append(np.copy(self.y))
        self.py_frames.append(partial_yield)
        self.angle_frames.append(angles)

    def calc_step_4_p_gauss(self):
        yamp = self.yamp
        y = self.y
        d = self.d
        points_sep = self.points_sep
        pair_sep = points_sep
        angle = self.angle

        max_angle = np.pi/2.0 - 0.01

        partial_yield = np.empty(self.points, dtype=float)
        diffusion = np.empty(self.points, dtype=float)
        angles = np.empty(self.points, dtype=float)
        coef  = [1.0/12, -2.0/3, 2.0/3, -1.0/12]

        dcoef = [-1.0/12, 4.0/3, -5.0/2.0, 4.0/3, -1.0/12]

        for i in range(2, self.points-2):
            tangle = y[i-2]*coef[0] + y[i-1]*coef[1] + y[i+1]*coef[2] + y[i+2]*coef[3]
            langle = np.arctan(tangle/pair_sep)
            fangle =  angle - langle
            if fangle > max_angle:
                print("WARNING !:", fangle)
                fangle = max_angle
                print("result::", yamamura(fangle, self.thetao, self.fval))
            partial_yield[i] = yamp*yamamura(fangle, self.thetao, self.fval)
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
                print("WARNING !:", fangle)
                fangle = max_angle
                print("BONDARY RESULT::", yamamura(fangle, self.thetao, self.fval))

            partial_yield[i] = yamp*yamamura(fangle, self.thetao, self.fval)
            angles[i] = fangle

            dangle = y[ilist[0]]*dcoef[0] + y[ilist[1]]*dcoef[1] + y[i]*dcoef[2] + y[ilist[2]]*dcoef[3] + y[ilist[3]]*dcoef[4]
            diffusion[i] = d*(dangle/(pair_sep*pair_sep))

        hit_id = random.randint(0, self.points-1)
        gauss_hit = np.zeros(self.points, dtype=float)

        gauss_fun = 20*mlab.normpdf(self.x[int(self.points*0.3):int(self.points*0.7)], 50, 2)

        #def gauss_distortion(self, amp, mu, sigma):
        #solver.gauss_distortion(22.4, 50, 5)
        #amp*mlab.normpdf(self.x ,mu, sigma)

        for i in range(len(gauss_fun)):
            gauss_hit[hit_id-i] += gauss_fun[i]


        for i in range(self.points):
            y[i] -= partial_yield[i] * gauss_hit[i]
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

        for i in range(3, self.points-3):
            tangle = y[i-3]*coef[0] + y[i-2]*coef[1] + y[i-1]*coef[2] + y[i+1]*coef[3] + y[i+2]*coef[4] + y[i+3]*coef[5]
            langle = np.arctan(tangle/pair_sep)
            fangle =  angle - langle
            if fangle > max_angle:
                print("WARNING !:", fangle)
                fangle = max_angle
                print("result::", yamamura(fangle, self.thetao, self.fval))
            partial_yield[i] = yamp*yamamura(fangle, self.thetao, self.fval)
            angles[i] = fangle

        angles[0] = angles[1] = angles[2]  = angles[3]
        angles[-1] = angles[-2] = angles[-3]  = angles[-4]


        for i in range(self.points):
            y[i] -= partial_yield[i]
        y[0] = y[1] = y[2] = y[3]
        y[-1]=y[-2] = y[-3] = y[4]

        self.frames.append(np.copy(self.y))
        self.py_frames.append(partial_yield)
        self.angle_frames.append(angles)


    def calc_step_f3_p(self):
        yamp = self.yamp
        y = self.y
        d = self.d
        points_sep = self.points_sep
        angle = self.angle

        max_angle = np.pi/2.0 - 0.01
        partial_yield = np.empty(self.points, dtype=float)
        diffusion = np.empty(self.points, dtype=float)
        angles = np.empty(self.points, dtype=float)
        for i in range(self.points):
            tangle = (-y[i-1]+y[i])/points_sep
            #tangle = -(11.0/6.0)*y[i-3] + 3*y[i-2] -(3.0/2.0)*y[i-1] + (1.0/3.0)*y[i]
            langle = np.arctan(tangle)
            fangle =  angle - langle
            if fangle > max_angle:
                print("WARNING !:", fangle)
                fangle = max_angle
                print("result::", yamamura(fangle, self.thetao, self.fval))
            partial_yield[i] = yamp*yamamura(fangle, self.thetao, self.fval)
            angles[i] = fangle

            diffusion[i] = 0

        for i in range(self.points):
            y[i] -= partial_yield[i]
            y[i] += diffusion[i]

        self.frames.append(np.copy(self.y))
        self.py_frames.append(partial_yield)
        self.angle_frames.append(angles)


    def calc_leap_frog_2_p(self):
        if not self.lfrog_init:
            print("init lfrog")
            self.lfrog_init = True
            self.py = np.copy(self.y)

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
        coef  = [1.0/12, -2.0/3, 0.0, 2.0/3, -1.0/12] # coef prime
        #coef  = [0.0, 0.0, -1.0, 1.0, 0.0] # coef prime
        #coef  = [0.0, -1.0, 1.0, 0.0, 0.0] # coef prime

        dcoef = [-1.0/12, 4.0/3, -5.0/2.0, 4.0/3, -1.0/12]
        wflag = False
        for i in range(2, self.points-2):
            tangle = y[i-2]*coef[0] + y[i-1]*coef[1] + y[i]*coef[2] + y[i+1]*coef[3] + y[i+2]*coef[4]
            langle = np.arctan(tangle/points_sep)
            fangle =  angle - langle
            if fangle > max_angle:
                fangle = max_angle - 0.5
                print("F!")
                wflag = True

            partial_yield[i] = yamp*yamamura(fangle, self.thetao, self.fval)
            angles[i] = fangle

            dangle = y[i-2]*dcoef[0] + y[i-1]*dcoef[1] + y[i]*dcoef[2] + y[i+1]*dcoef[3] + y[i+2]*dcoef[4]
            diffusion[i] = d*(dangle/(pair_sep*pair_sep))

        bcon = {0:[-2,-1,0,1,2], 1:[-1,0,1,2,3],
                -1:[-3,-2,-1,0,1], -2:[-4,-3,-2, -1, 0]}
        for i in bcon:
            ilist = bcon[i]
            tangle = y[ilist[0]]*coef[0] + y[ilist[1]]*coef[1] + y[ilist[2]]*coef[2] + y[ilist[3]]*coef[3] + y[ilist[4]]*coef[4]
            langle = np.arctan(tangle/points_sep)
            fangle = angle - langle

            if fangle > max_angle:
                fangle = max_angle - 0.5
                wflag = True

            partial_yield[i] = yamp*yamamura(fangle, self.thetao, self.fval)
            angles[i] = fangle

            dangle = y[ilist[0]]*dcoef[0] + y[ilist[1]]*dcoef[1] + y[i]*dcoef[2] + y[ilist[2]]*dcoef[3] + y[ilist[3]]*dcoef[4]
            diffusion[i] = d*(dangle/(pair_sep*pair_sep))

        # -----------------------
        if wflag:
            "WARNING!: over-somethin... "
        ny = np.copy(self.py)
        ny -= partial_yield
        ny += diffusion

        self.py = y
        self.y = ny

        self.frames.append(np.copy(self.y))
        self.py_frames.append(partial_yield)
        self.angle_frames.append(angles)


    def calc_leap_frog_MIX_p(self):
        if not self.lfrog_init:
            print("init lfrog")
            self.lfrog_init = True
            self.py = np.copy(self.y)

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
        central  = [1.0/12, -2.0/3,  0, 2.0/3, -1.0/12] # coef prime
        c_weight = 1.0
        forward  = [0, 0, -3.0/2.0, 2.0, -0.5] # coef prime
        f_weight = -c_weight + 1.0

        coef = [(cx*c_weight+cy*f_weight) for cx, cy in zip(central, forward)]

        dcoef = [-1.0/12, 4.0/3, -5.0/2.0, 4.0/3, -1.0/12]
        wflag = False
        for i in range(2, self.points-2):
            tangle = y[i-2]*coef[0] + y[i-1]*coef[1] + y[i]*coef[2] + y[i+1]*coef[3] + y[i+2]*coef[4]
            langle = np.arctan(tangle/points_sep)
            fangle =  angle - langle
            if fangle > max_angle:
                fangle = max_angle - 0.5
                print("F!")
                wflag = True

            partial_yield[i] = yamp*yamamura(fangle, self.thetao, self.fval)
            angles[i] = fangle

            dangle = y[i-2]*dcoef[0] + y[i-1]*dcoef[1] + y[i]*dcoef[2] + y[i+1]*dcoef[3] + y[i+2]*dcoef[4]
            diffusion[i] = d*(dangle/(pair_sep*pair_sep))

        bcon = {0:[-2,-1,0,1,2], 1:[-1,0,1,2,3],
                -1:[-3,-2,-1,0,1], -2:[-4,-3,-2, -1, 0]}
        for i in bcon:
            ilist = bcon[i]
            tangle = y[ilist[0]]*coef[0] + y[ilist[1]]*coef[1] + y[ilist[2]]*coef[2] + y[ilist[3]]*coef[3] + y[ilist[4]]*coef[4]
            langle = np.arctan(tangle/points_sep)
            fangle = angle - langle

            if fangle > max_angle:
                fangle = max_angle - 0.5
                wflag = True

            partial_yield[i] = yamp*yamamura(fangle, self.thetao, self.fval)
            angles[i] = fangle

            dangle = y[ilist[0]]*dcoef[0] + y[ilist[1]]*dcoef[1] + y[i]*dcoef[2] + y[ilist[2]]*dcoef[3] + y[ilist[3]]*dcoef[4]
            diffusion[i] = d*(dangle/(pair_sep*pair_sep))

        # -----------------------
        if wflag:
            "WARNING!: over-somethin... "
        ny = np.copy(self.py)
        ny -= partial_yield
        ny += diffusion

        self.py = y
        self.y = ny

        self.frames.append(np.copy(self.y))
        self.py_frames.append(partial_yield)
        self.angle_frames.append(angles)


    def calc_volume_cell(self):

        diff_for = np.diff(np.pad(self.y, (0, 1), 'wrap'), 1)
        diff_back = np.diff(np.pad(self.y, (1, 0), 'wrap'), 1)

        angle_for = -np.arctan(diff_for/self.points_sep) + self.angle
        angle_back = -np.arctan(diff_back/self.points_sep) + self.angle

        sput_for = yamamura(angle_for, self.thetao, self.fval)  * np.cos(angle_for)
        sput_back = yamamura(angle_back, self.thetao, self.fval) * np.cos(angle_back)
        #diffusion = self.d*np.diff(np.pad(self.y, (1,1), 'wrap'), 2)


        self.y -= (sput_for+sput_back)*self.yamp #*np.random.normal(1, 0.10, self.points)
        #self.y -= self.yamp*(sput_for+sput_back)*self.gauss_arr[np.random.randint(self.points)]
        #self.y += diffusion



        self.frames.append(np.copy(self.y))

    def run(self, num=100, mode=0, print_num=500):
        for i in range(num):
            if i%print_num == 0:
                print("step:{}".format(i))
            if mode == 0:
                self.calc_step()
            elif mode == 1:
                self.calc_step_4()
            elif mode == 2:
                self.calc_step_6()
            elif mode == 3:
                self.calc_step_4_p()
            elif mode == 4:
                self.calc_leap_frog_2_p()
            elif mode == 5:
                self.calc_step_f3_p()
            elif mode == 6:
                self.calc_leap_frog_MIX_p()
            elif mode == 7:
                self.calc_step_ave()
            elif mode == 8:
                self.calc_step_4_p_gauss()
            elif mode == 9:
                self.calc_volume_cell()
            else:
                self.calc_step_ave()



#x, np.radians(4.7), 3.01)
solver = Solver(60, 400, 100, yamp=0.00025, d=0.0000, fval=2.0)
#solver = Solver(45, 100, 500, yamp=0.001, d=0.00009, fval=2.4)
$solver.sin_distortion(15, 1)
#solver.sin_distortion(0.023, 13)
#solver.sin_distortion(0.05, 12)
#solver.sin_distortion(0.20, 3)
#solver.sin_distortion(0.12, 7)
#solver.gauss_distortion(22.4, 50, 5)
#solver.gauss_distortion(-3, 60, 10)
#solver.add_normal_noise(0.0003)
#solver.zigzag_distortion(0.01)

#import random
#random.seed(101)
#for i in random.sample(range(15,87),23):
#    solver.gauss_distortion(0.2, i, 6)

#solver.add_normal_noise(0.0010)

c = 2
while c!=0:
    solver.run(c, 9)
    solver.show(yspace=[-2.2, 2.2])
    #solver.show(yspace=[-2.0, 2.0], zoom=[0.0,1.0])
    #solver.show(yspace=[-0.1, 0.1], zoom=[0.53,0.562])
    #solver.show(yspace=[-0.10, 0.25], zoom=[0.555,0.58])
    c = int( input("continue :"))




#solver.show(zoom=[0.3,0.7])
if False:
    x = np.linspace(0.1, np.pi/2.0-0.02, 100)
    #x = np.linspace(0.1, 6, 100)
    #y = np.array([ np.arctan(i) for i in x ])
    y = np.array(yamamura(x, np.radians(4.7), 3.01))
    #y = np.array([ yamamura(np.arctan(i), np.radians(77), self.fval) for i in x ])
    plt.plot(x*57.3, y)
    plt.show()



