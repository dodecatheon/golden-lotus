#!/usr/bin/env python
"""\
Plot interlaced golden spirals,  r = phi ** (theta / 36)
with rotations and reflections.
"""
import matplotlib
# matplotlib.use('PS')
import matplotlib.pyplot as plt
import numpy as np
from math import *

# Turn to True if you want to see the dotted lines
do_dotted = True

# Turn to True if you want to interact with the plot:
show_plot = True

# Line thickness
lth = 3.5
bth = 1

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

if do_dotted:

    # 20-gon radii:
    # for angle in range(0,360,18):
    #     plt.plot([theta[angle], theta[angle]],
    #              [0., rmax],
    #              linestyle='dotted',
    #              color='grey',
    #              linewidth=0.5)

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

for ftype in ['svg', 'pdf']:
    plt.savefig("lotus_polar." + ftype,
                bbox_inches='tight',
                pad_inches=0,
                transparent=True,
                dpi=18)

if show_plot:
    plt.show()
