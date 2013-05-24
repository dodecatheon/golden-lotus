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

# Here's an interesting question:
#
# We are approximating each 36 degree segment of the spiral by a 36 degree arc
# with radius equal to the spiral's distance from the center at the far
# end of the arc (from the center).
#
# However, we don't want to draw simply arcs, but arcs with gaps to simulate
# the over-under pattern of a knot.  That means that we want to subtract some
# delta from the ends of the 36 degree segments, but in some cases we want to
# split the segment and put the gap "in the middle".
#
# But where is the middle?  It isn't simply 18 degrees along the arc.  The
# middle is where spirals of opposite direction would cross each other.
# At that point, if the outer radius of the arc is phi and the inner is 1, the
# point in the middle has radius sqrt(phi).
#
# Using the formula for logarithmic spiral arc lengths (starting at the center
# with $\theta = -\infty$), we can cancel terms to get that the arc length
# from closer radius to the "middle" is
#
#   (sqrt(phi) - 1) / (phi - 1)
#   = 0.44013703852159741
#
# of the way along the arc.  This value is very close to 44% of 36 degrees
# or pi / 5 if using radians.  This is about 15.85 degrees.  So we will break
# a circular arc into two segments, one with ~15.85 degrees and one with
# ~20.15 degrees.
#
arc_fraction = (sqrt(phi) - 1.) / (phi - 1.)

mid_lo = pi_over_5 * arc_fraction
mid_hi = pi_over_5 * (1. - arc_fraction)

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

X = pi_over_5
Y = 2 * X

D = pi_over_5

n1 = 16
n2 = 20

DD1 = mid_hi
DD2 = mid_lo

M1 = -Y + mid_hi
M2 = X + mid_lo

F = 3.

def arc_segment(index, x, delta, n, d0):
    rn = range(n+1)
    fn = float(n)
    d1 = d0 - delta
    d2 = d0 - 2. * delta
    xx = [x, x, x + delta, x + delta][index]
    dd = [d0, d1, d2, d1][index]
    return np.array([ xx + float(ii) * dd / fn for ii in rn])

def plot_polar_affine(th, rot, scale, shift, i, k, color, lw):
    global th72, th_incr, phi

    rb = np.array([1.0 for ii in range(len(th))])

    tha, ra = polar_affine(th, rb, rot, scale, shift)

    tha += i * th72 + k * th_incr
    ra *= phi**k

    plt.plot(tha, ra, color=color, linewidth=lw)

    return None

lw =
for (rot,
     scale,
     shift,
     th_incr,
     color_lo,
     color_hi) in [(-(th90 + th72),
                     phi,
                     (cos(th18), sin(th18)),
                     th36,
                     'green',
                     'red'),
                   (-(th90 + th36),
                     phi,
                     (cos(th18), sin(th18)),
                     th36,
                     'blue',
                     'purple')]:


# Loop over petals
for i in range(5):

    delta = D / F * phi

    # Loop over 36-degree segments in each petal
    for k in range(5):

        # scale border-angle gap by the radius of curvature
        delta /= phi

        if   k == 0:
            plot_polar_affine(arc_segment(0, -Y, delta, 36, D),
                              rot, scale, shift, i, k, 'green')

            plot_polar_affine(arc_segment(1, X, delta, 36, D),
                              rot, scale, shift, i, k, 'red')
        elif k == 1:
            plot_polar_affine(arc_segment(3, -Y, delta, 36, D),
                              rot, scale, shift, i, k, 'green')

            plot_polar_affine(arc_segment(3, X, delta, 36, D),
                              rot, scale, shift, i, k, 'red')

        elif k == 2:
            plot_polar_affine(arc_segment(2, -Y, delta, 36, D),
                              rot, scale, shift, i, k, 'green')

            plot_polar_affine(arc_segment(3, M2, delta, 36, mid_hi),
                              rot, scale, shift, i, k, 'red')

            plot_polar_affine(arc_segment(1, X, delta, 36, mid_lo),
                              rot, scale, shift, i, k, 'red')

        elif k == 3:
            plot_polar_affine(arc_segment(1, -Y, delta, 36, D),
                              rot, scale, shift, i, k, 'green')

            plot_polar_affine(arc_segment(1, X, delta, 36, D),
                              rot, scale, shift, i, k, 'red')

        else:
            plot_polar_affine(arc_segment(0, -Y, delta, 36, D),
                              rot, scale, shift, i, k, 'green')

            plot_polar_affine(arc_segment(3, X, delta, 36, D),
                              rot, scale, shift, i, k, 'red')

# Loop over petals
for i in range(5):

    delta = D / F * phi

    # Loop over 36-degree segments in each petal
    for k in range(5):

        # scale border-angle gap by the radius of curvature
        delta /= phi

        if   k == 0:
            plot_polar_affine(arc_segment(0, -Y, delta, 36, D),
                              rot, scale, shift, i, k, 'green')

            plot_polar_affine(arc_segment(1, X, delta, 36, D),
                              rot, scale, shift, i, k, 'red')
        elif k == 1:
            plot_polar_affine(arc_segment(3, -Y, delta, 36, D),
                              rot, scale, shift, i, k, 'green')

            plot_polar_affine(arc_segment(3, X, delta, 36, D),
                              rot, scale, shift, i, k, 'red')

        elif k == 2:
            plot_polar_affine(arc_segment(2, -Y, delta, 36, D),
                              rot, scale, shift, i, k, 'green')

            plot_polar_affine(arc_segment(3, M2, delta, 36, mid_hi),
                              rot, scale, shift, i, k, 'red')

            plot_polar_affine(arc_segment(1, X, delta, 36, mid_lo),
                              rot, scale, shift, i, k, 'red')

        elif k == 3:
            plot_polar_affine(arc_segment(1, -Y, delta, 36, D),
                              rot, scale, shift, i, k, 'green')

            plot_polar_affine(arc_segment(1, X, delta, 36, D),
                              rot, scale, shift, i, k, 'red')

        else:
            plot_polar_affine(arc_segment(0, -Y, delta, 36, D),
                              rot, scale, shift, i, k, 'green')

            plot_polar_affine(arc_segment(3, X, delta, 36, D),
                              rot, scale, shift, i, k, 'red')


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
        pass
        # plt.plot(th[s], rs[s], color=lcolor, linewidth=lth)

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
