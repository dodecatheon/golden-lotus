#!/usr/bin/env python
"""\
Plot interlaced golden spirals,  r = phi ** (theta / 36)
with rotations and reflections.

In this version, also plot a set of golden triangles in red, and their
associated circular arcs in dotted green.
"""
import matplotlib
# matplotlib.use('PS')
import matplotlib.pyplot as plt
import numpy as np
from math import *
from pprint import pprint

# Turn to True if you want to see the dotted lines
do_dotted = True

# Turn to True if you want to interact with the plot:
show_plot = True

# Line thickness
lth = 0.5
bth = 0.5
gth = 0.5

# Line colors
bcolor='gray'
lcolor='darkblue'

# Constants
pi_over_5 = pi / 5.
twopi = 2. * pi
phi = 2. * cos(pi_over_5)

@np.vectorize
def rfunc(angle):
    return phi ** (min(abs(angle), twopi - angle) / pi_over_5)

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

def polar_affine(theta, r, rot, scale, shift):
    """Returns theta and r, rotated by rot, scaled by scale, with xy shift"""
    rr = r * scale
    th = theta + rot
    xx, yy = xyfromr(rr, th)
    xx += shift[0]
    yy += shift[1]
    return theta_r_from_xy(xx, yy)

def xydist(xy1, xy2):
    x1, y1 = xy1
    x2, y2 = xy2
    return sqrt((x1-x2)**2 + (y1-y2)**2)

def rdist(r1, th1, r2, th2):
    return xydist(xyfromr(r1,th1), xyfromr(r2,th2))

@np.vectorize
def rfunc(angle):
    return phi ** (angle / pi_over_5) \
        if angle <= pi \
        else phi ** ( (twopi - angle) / pi_over_5 )

def rotate(x, y, c, s):
    return x * c + y * s, y * c - x * s

theta = np.array([a/180. * pi for a in range(361)])

rs = rfunc(theta)
rc = np.array([1.0 for a in range(361)])

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

for i in range(5):
    # if do_dotted:
    #     plt.plot(theta,
    #              rr,
    #              color='grey',
    #              linestyle='dotted',
    #              linewidth=0.5)
    rr *= phi


plt.plot(theta, rr, color=bcolor, linewidth=1)

# Rotation angles
th18 = theta[18]
th36 = theta[36]
th72 = theta[72]
th90 = theta[90]
th108 = theta[108]

if do_dotted:

    # 20-gon radii:
    # for angle in range(0,360,18):
    #     plt.plot([theta[angle], theta[angle]],
    #              [0., rmax],
    #              linestyle='dotted',
    #              color='grey',
    #              linewidth=0.5)

    # Decagrams
    # for r in [phi**j for j in range(5,-1,-1)]:
    #     plt.plot([float((3*i) % 10) * th36 + th18 for i in range(11)],
    #              [r for i in range(11)],
    #              color='gray',
    #              linewidth=0.5)

    rtri = np.array([1.0, phi, 1.0, 1.0])
    thtri = np.array([-th90, th36 - th90, th18, -th90])

    plt.plot(thtri, rtri, color='red', linewidth=1)

    th_base = np.array([a/180. * pi for a in range(-72,73)])
    # th_base = theta[:37]
    r_base = np.array([1.0 for i in range(len(th_base))])

    tharc, rarc = polar_affine(th_base,
                               r_base,
                               - (th90 + th72),
                               phi,
                               (cos(th18), sin(th18)))
    plt.plot(tharc, rarc, color='black', linewidth=gth)

    for k in range(4):
        rtri *= phi
        thtri += th36

        rarc *= phi
        tharc += th36

        plt.plot(thtri, rtri, color='red', linewidth=1)
        plt.plot(tharc, rarc, color='black', linewidth=gth)


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

for ftype in ['svg', 'pdf']:
    plt.savefig("lotus_tri." + ftype,
                bbox_inches='tight',
                pad_inches=0,
                transparent=True,
                dpi=18)

if show_plot:
    plt.show()
