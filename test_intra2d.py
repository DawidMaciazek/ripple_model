from scipy import interpolate
import matplotlib.pyplot as plt
from  mpl_toolkits.mplot3d import axes3d
import numpy as np

def rotation_matrix(r_vec, ra):
    r_vec = np.array(r_vec)
    r_vec = r_vec/np.linalg.norm(r_vec)
    rmatrix = [[np.cos(ra) + r_vec[0]*r_vec[0]*(1-np.cos(ra)),
                r_vec[0]*r_vec[1]*(1-np.cos(ra)) - r_vec[2]*np.sin(ra),
                r_vec[0]*r_vec[2]*(1-np.cos(ra)) + r_vec[1]*np.sin(ra)],
               [r_vec[1]*r_vec[0]*(1-np.cos(ra)) + r_vec[2]*np.sin(ra),
                np.cos(ra) + r_vec[1]*r_vec[1]*(1-np.cos(ra)),
                r_vec[1]*r_vec[2]*(1-np.cos(ra)) - r_vec[0]*np.sin(ra)],
               [r_vec[2]*r_vec[0]*(1-np.cos(ra)) - r_vec[1]*np.sin(ra),
                r_vec[2]*r_vec[1]*(1-np.cos(ra)) + r_vec[0]*np.sin(ra),
                np.cos(ra) + r_vec[2]*r_vec[2]*(1-np.cos(ra))]]

    return np.array(rmatrix)


x = np.arange(-5,5.1,0.2)
print(x.shape)
y = x.copy()

x_new = np.arange(-2,2.01,0.1)
y_new = np.arange(-2,2.01,0.1)

xx, yy = np.meshgrid(x, y)
z = np.sin(xx**2)
#plt.imshow(z)
#plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(xx, yy, z)
#plt.show()


rm = rotation_matrix(np.array([0,0,1], dtype=float), np.radians(25))

xyz = np.stack((xx, yy, z), axis=-1)
xyz_rot = np.tensordot(xyz, rm, axes=([2], [0]))

xyz_unstacked = np.split(xyz_rot, 3, axis=-1)

x_ = np.squeeze(xyz_unstacked[0])
y_ = np.squeeze(xyz_unstacked[1])
z_ = np.squeeze(xyz_unstacked[2])

inte = interpolate.interp2d(x_, y_, z_, kind='cubic')

xx_new, yy_new = np.meshgrid(x_new, y_new)
z_new = inte(x_new, y_new)


ax.plot_surface(xx_new, yy_new, z_new)
ax.plot_surface(xx, yy, z)
plt.show()

plt.imshow(z_new)
plt.show()


