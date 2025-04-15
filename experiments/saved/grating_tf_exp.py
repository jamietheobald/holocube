# a tuning curve displaying a series of temporal frequencies

import holocube.hc5 as hc
import numpy as np

num_frames = 400

# inverted sf, how many degrees for a cycle (assuming a square will
# cover 90 degrees)
isf = 20 * np.pi / 180  # inverted sf
sf = 1 / isf
# multiple tfs 
tfs = np.array([0.1, .2, .5, 1, 2, 5, 10, 20, 50, 100], dtype=float)

# empty list of gratings
gratings = []
# which directions they will flow--here left and right (0, and pi radians)
grating_dirs = [0 * np.pi / 2, 2 * np.pi / 2]
# xyz outline of the square in 3d relative to the camera. This makes a
# square of 90x90 degrees, in front of the viewer (z = -1)
xs = [-1, 1, 1, -1]
ys = [-1, -1, 1, 1]
zs = [-1, -1, -1, -1]
garr = np.array([xs, ys, zs])

# if you want to move the grating to somewhere other than the front of
# the viewer, do it here by substituting rotating degrees:
az = 0  # azimuth degrees
el = 0  # elevation degrees
rate = 120
sd = 0.25

# Here we create and append each spatial frequency grating to the
# list. o is the orientation, c is contrast, rate is your monitor
# frame rate, but defaults to 120 hz. sd is the rate of gaussian
# fading, which puts the grating in a blur circle, the whole thing
# called a gabor stimulus. .25 works well, but try 5 to make an
# unfiltere square, and .05 to show a more severe fade out.
for tf in tfs:
    for grating_dir in grating_dirs:
        g = hc.stim.Movable_grating(hc.window, garr, sf=sf, tf=tf,
                                    o=grating_dir, rate=rate, sd=sd)
        g.set_ry(az)
        g.set_rx(el)
        gratings.append(g)

bg_color = [0.5, 0.5, 0.5, 1.0]  # middle gray

# commands to start the experiment
estarts = [[hc.window.set_far, 50],
           [hc.window.reset_pos_rot],
           [hc.window.set_bg, bg_color]]

# and when all tests are done, reset the visible distance, and
# background to black
eends = [[hc.window.set_far, 2],
         [hc.window.set_bg, [0.0, 0.0, 0.0, 1.0]]]

# add the experiment to the scheduler
hc.scheduler.add_exp(None, estarts, eends)

# now add each test, which is each grating we've made
test_ind = 0
for grating in gratings:
    exp_synch, exp_num = hc.tools.test_bin_flash(test_ind, num_frames)

    # simple test progression just involves on the current grating
    starts = [[grating.on, True]]

    # advancing the frames for each frame in the test
    middles = [[grating.next_frame]]

    # and turning it off
    ends = [[grating.on, False]]

    # then add each test to the experiment
    hc.scheduler.add_test(num_frames, starts, middles, ends)

    test_ind += 1

    # you could add any number of other stimuli here, such as turning
    # on points, or a flower at starts, moving them in middles, then
    # turning them off at ends.

# add the rest, just 20 frames of nothing
num_frames = 20
rbar = hc.stim.cbarr_class(hc.window, dist=1)

starts = [[hc.arduino.set_lmr_scale, -.1],
          ]

middles = [
]

ends = [
]

hc.scheduler.add_rest(num_frames, starts, middles, ends)
