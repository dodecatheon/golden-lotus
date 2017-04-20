#!/usr/bin/env python
from __future__ import print_function, division
try:
    range = xrange
    input = raw_input
    from itertools import izip as zip
except NameError:
    pass

import matplotlib
matplotlib.use('SVG')
import matplotlib.pyplot as plt
import numpy as np
from math import *

__doc__ = """\
Plot approximate logarithmic spirals in an {n,3} star, using circular arc
approximations.
"""

vcos = np.vectorize(cos)
vsin = np.vectorize(sin)

def xyfromr(r, theta):
    return vcos(theta) * r, vsin(theta) * r

def theta_r_from_xy(x, y):
    """x = array of x coordinates, y = array of y coordinates)
returns theta, r arrays"""
    r = np.array([sqrt(xx * xx + yy * yy) for xx, yy in zip(x, y)])
    th = np.array([atan2(yy, xx) for xx, yy in zip(x, y)])
    return th, r

@np.vectorize
def rfunc(rbase, theta_denom, angle):
    return rbase ** (min(abs(angle), twopi - angle) / theta_denom)

def rotate(x, y, c, s):
    return x * c + y * s, y * c - x * s

# Rotate theta, r coordinates by angle rot around (r,theta) point p,
# then scale the new radius
def mytwist(theta, r, rot, scale, p):
    x, y = xyfromr(r, theta)
    px, py = xyfromr(*p)
    x -= px
    y -= py
    c = cos(rot)
    s = sin(rot)
    xx, yy = rotate(x,y,cos(rot),sin(rot))
    xx += px
    yy += py
    rr, th = theta_r_from_xy(xx,yy)
    rr *= scale
    return (rr, th)

def rdist(r1, th1, r2, th2):
    return xydist(xyfromr(r1,th1), xyfromr(r2,th2))


def plot_lotus(n,
               do_dotted=False,
               show_plot=False,
               lth=3.5,         # foreground thickness
               bth=1,           # background thickness
               bcolor='teal',   # background line color
               lcolor='gold'):  # foreground line color
    "plot a lotus"

    # n = number of sides in polygon
    polyangle = 2 * pi / n
    polyside = 2 * sin( pi / n )
    anglerange = polyangle + 1

    # radius to vertex of polygon enclosed in {n/3}
    rscaleinv = 1 - polyside**2
    rscale = 1 / rscaleinv
    nn = n / 2
    pi_over_nn = pi / nn
    twopi = 2. * pi

    theta = np.array([a/180. * pi for a in range(anglerange)])

    rs = rfunc(rscale, pi_over_nn, theta)
    rc = np.array([1.0 for a in range(anglerange)])

    rmax = rs[180]

    # Clear figure before starting:
    plt.clf()

    # Circles at increasing multiples of phi

    rr = rc * 1.0

    # Polar plot:
    sp = plt.subplot(1, 1, 1, projection='polar')

    # Turn labels off:
    sp.grid(False)
    sp.set_xticklabels([])
    sp.set_yticklabels([])

    for i in range(nn):
        # if do_dotted:
        #     plt.plot(theta,
        #              rr,
        #              color='grey',
        #              linestyle='dotted',
        #              linewidth=0.5)
        rr *= rscale


    plt.plot(theta, rr, color=bcolor, linewidth=1)

# Rotation angles
th18 = theta[18]
th36 = theta[36]
th72 = theta[72]
th90 = theta[90]

if do_dotted:

    # 20-gon radii:
    # for angle in range(0,360,18):
    #     plt.plot([theta[angle], theta[angle]],
    #              [0., rmax],
    #              linestyle='dotted',
    #              color='grey',
    #              linewidth=0.5)

    # Decagrams
    for r in [phi**j for j in range(5,-1,-1)]:
        plt.plot([float((3*i) % 10) * th36 + th18 for i in range(11)],
                 [r for i in range(11)],
                 color='gray',
                 linewidth=0.5)

    # # Pentagrams:
    # plt.plot([float((2*i) % 5) * th72 + th18 for i in range(6)],
    #          [rmax for i in range(6)],
    #          color='gray',
    #          linewidth=1)
    #
    # plt.plot([float((2*i) % 5) * th72 - th90 for i in range(6)],
    #          [phi**3 for i in range(6)],
    #          color='gray',
    #          linewidth=1)

# copy circle, rotated by -90 degrees
th = theta - th90

main_slices = [slice(0,28),
               slice(44,87),
               slice(94,143),
               slice(146,250),
               slice(255,284),
               slice(293,361)]

# Primary petals
for count in range(5):
    for s in main_slices:
        plt.plot(th[s], rs[s], color=lcolor, linewidth=lth)

    # Rotate coordinates by 72 degrees
    th += th72


# Rotate coordinates by 36 degrees
th += th36

# interspersed petals have a lighter outer section and an inner section
# that is part of the knot with the 5 primary petals

outer_slices = [slice(111,125),
                slice(128,143),
                slice(146,161),
                slice(164,197),
                slice(200,233),
                slice(236,250)]

inner_slices = [slice(72,87),
                slice(94,109),
                slice(253,288)]

for count in range(5):
    for s in outer_slices:
        plt.plot(th[s], rs[s], color=bcolor, linewidth=bth)

    for s in inner_slices:
        plt.plot(th[s], rs[s], color=lcolor, linewidth=lth)

    # Rotate coordinates by 72 degrees
    th += th72

# Adjust boundaries on figure
plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)

for ftype in ['svg']:
    plt.savefig("lotus_polar." + ftype,
                bbox_inches='tight',
                pad_inches=0.1,
                transparent=False,
                dpi=300)

if show_plot:
    plt.show()
