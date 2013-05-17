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
lth = 1
bth = 1
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

# th_base = np.array([a/180. * pi for a in range(-72,73)])
# r_base = np.array([1.0 for i in range(len(th_base))])

X = 36. / 180. * pi
Y = 72. / 180. * pi

D = Y - X

D2 = D/2.
M1 = -Y + D2
M2 = X + D2

F = 6.

delta = D / F * phi

rb36 = np.array([1.0 for i in range(37)])
rb18 = np.array([1.0 for i in range(19)])

th1_k = [

for i in range(-1,0):
    for (rot,
         scale,
         shift,
         th_incr) in [(-(th90 + th72), phi, (cos(th18), sin(th18)), th36)]:

        for k in range(5):

            delta /= phi
            d1 = D - delta
            d2 = D - 2. * delta
            d3 = D2 - delta
            d4 = D2 - 2. * delta

            th1_opts = [np.array([ - Y + float(i) * D / 36. for i in range(37)]),
                        np.array([ - Y + float(i) * d1 / 36. for i in range(37)]),
                        np.array([ delta - Y + float(i) * d2 / 36. for i in range(37)]),
                        np.array([ - Y + float(i) * D2 / 18. for i in range(19)]),
                        np.array([ - Y + float(i) * d3 / 18. for i in range(19)]),
                        np.array([ delta - Y + float(i) * d4 / 18. for i in range(36)]),
                        np.array([ M1 + float(i) * D2 / 18. for i in range(19)]),
                        np.array([ M1 + float(i) * d3 / 18. for i in range(19)]),
                        np.array([ M1 + delta + float(i) * d4 / 18. for i in range(19)])]

            th1 = th1_opts[th1_k[k]]




            th2 = np.array([ delta + X + float(i) * d / 36. for i in range(37)])
            #tharc, rarc = polar_affine(th_base, r_base, rot, scale, shift)

            tha1, ra1 = polar_affine(th1, r_base, rot, scale, shift)
            tha2, ra2 = polar_affine(th2, r_base, rot, scale, shift)

            tha1 += i * th72 + k * th_incr
            tha2 += i * th72 + k * th_incr
            ra1 *= phi**k
            ra2 *= phi**k
            # plt.plot(tharc, rarc, color='black', linewidth=gth)
            # plt.plot(tharc[1:36], rarc[1:36], color='red', linewidth=2)
            # plt.plot(tharc[109:144], rarc[109:144], color='red', linewidth=2)
            plt.plot(tha1, ra1, color='green', linewidth=3)
            plt.plot(tha2, ra2, color='red', linewidth=3)

            # for k in range(4):
            #     # rarc *= phi
            #     # tharc += th_incr
            #     ra1 *= phi
            #     ra2 *= phi
            #     tha1 += th_incr
            #     tha2 += th_incr
            #     # plt.plot(tharc, rarc, color='black', linewidth=gth)
            #     # plt.plot(tharc[1:36], rarc[1:36], color='red', linewidth=2)
            #     # plt.plot(tharc[109:144], rarc[109:144], color='red', linewidth=2)
            #     plt.plot(tha1, ra1, color='red', linewidth=2)
            #     plt.plot(tha2, ra2, color='red', linewidth=2)

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

for ftype in ['pdf']:
    plt.savefig("lotus_circle." + ftype,
                bbox_inches='tight',
                pad_inches=0,
                transparent=True,
                dpi=18)

if show_plot:
    plt.show()
