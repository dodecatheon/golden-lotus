#!/usr/bin/env python
"""\
Plot golden spiral r = phi ** (theta / 36)
"""
import matplotlib
# matplotlib.use('PS')
import matplotlib.pyplot as plt
import numpy as np
from math import *

# Constants
pi_over_5 = pi / 5.
twopi = 2. * pi
phi = 2. * cos(pi_over_5)

@np.vectorize
def rfunc(angle):
    return phi ** (angle / pi_over_5) \
        if angle <= pi \
        else phi ** ( (twopi - angle) / pi_over_5 )

def rotate(x, y, c, s):
    return x * c + y * s, y * c - x * s

lth = 3

vcos = np.vectorize(cos)
vsin = np.vectorize(sin)

theta = np.array([a/180. * pi for a in range(361)])

rs = rfunc(theta)
xc = vcos(theta)
yc = vsin(theta)
xs = rs * xc
ys = rs * yc

rmax = rs[180]

# Circles at increasing multiples of phi

xx = xc * 1.0
yy = yc * 1.0

for i in range(5):
#     plt.plot(xx,
#              yy,
#              color='grey',
#              linestyle='dotted',
#              linewidth=0.5)
#
    xx *= phi
    yy *= phi

bcolor='gray'
lcolor='darkblue'

plt.plot(xx,
         yy,
         color=bcolor,
         linewidth=1)

# # 20-gon radii:
# for angle in range(0,360,18):
#     plt.plot([0., xx[angle]],
#              [0., yy[angle]],
#              linestyle='dotted',
#              color='grey',
#              linewidth=0.5)

# Rotation angle
c72 = xc[72]
s72 = yc[72]

c36 = xc[36]
s36 = yc[36]

# copy circle
xs0, ys0 = rotate(xs, ys, 0.0, 1.0)

# Petals
for count in range(5):

    # Grey outline
    # plt.plot(xs0, ys0, color='grey', linewidth=0.5)

    # First segment, darker and thicker
    plt.plot(xs0[0:32],
             ys0[0:32],
             color=lcolor,
             linewidth=lth)

    # Second segment, darker and thicker
    plt.plot(xs0[42:88],
             ys0[42:88],
             color=lcolor,
             linewidth=lth)

    # Third segment, darker and thicker
    plt.plot(xs0[93:144],
             ys0[93:144],
             color=lcolor,
             linewidth=lth)

    # Fourth segment, darker and thicker
    plt.plot(xs0[145:251],
             ys0[145:251],
             color=lcolor,
             linewidth=lth)

    # Fifth  segment, darker and thicker
    plt.plot(xs0[254:286],
             ys0[254:286],
             color=lcolor,
             linewidth=lth)

    # Last segment, darker and thicker
    plt.plot(xs0[291:360],
             ys0[291:360],
             color=lcolor,
             linewidth=lth)

    # Rotate coordinates by 72 degrees
    xs0, ys0 = rotate(xs0, ys0, c72, s72)


# Rotate coordinates by 36 degrees
xs0, ys0 = rotate(xs0, ys0, c36, s36)

# interspersed petals
for count in range(5):

    # Grey outline
    plt.plot(xs0[108:253],
             ys0[108:253],
             color=bcolor,
             linewidth=2)

    plt.plot(xs0[73:88],
             ys0[73:88],
             color=lcolor,
             linewidth=lth)

    plt.plot(xs0[92:109],
             ys0[92:109],
             color=lcolor,
             linewidth=lth)

    plt.plot(xs0[252:289],
             ys0[252:289],
             color=lcolor,
             linewidth=lth)

    # Rotate coordinates by 72 degrees
    xs0, ys0 = rotate(xs0, ys0, c72, s72)


# plt.axis([-12., 12., -12., 12.])
plt.axes().set_aspect('equal')
plt.savefig("lotus.pdf", bbox_inches=0, pad_inches=0, transparent=True, dpi=20)
plt.show()
