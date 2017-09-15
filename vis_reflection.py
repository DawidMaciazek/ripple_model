import numpy as np
import matplotlib.pyplot as plt


class VisReflection:
    def __init__(self, theta, function, nodes=50):
        self.x = np.linspace(0,1,nodes,endpoint=False)
        self.x_line = np.copy(self.x[:10])
        self.y = function(self.x)
        self.theta = np.radians(theta)
        print(self.theta)
        #np.zeros(self.x.shape, dtype=float)

    def show(self, angle=False):

        if angle==False:
            subplot_num = 1

        fig = plt.figure()
        ax = fig.add_subplot(1, subplot_num, 1)
        ax.set_aspect(1)
        ax.plot(self.x, self.y)

        ax.plot(self.x_line, self.x_line*(-np.tan(np.pi*0.5-self.theta)))

        alpha = np.arctan(np.diff(np.pad(self.y, (0,1), 'wrap'))/self.x[1])

        for i in range(self.x.shape[0]):
            b = np.arctan(-self.theta+2*alpha[i]+np.pi*0.5)
            ax.plot(self.x_line, self.x_line*b + self.y[i]-self.x_line[0]*b)
            self.x_line+=self.x[1]

        if angle == True:
            ax = fig.add_subplot(1, subplot_num,2)
            ax.plot(self.x, np.degrees(alpha))
        plt.show()



sin = lambda x: np.sin(x*np.pi*2)*0.125
vis = VisReflection(80, sin)
vis.show()

