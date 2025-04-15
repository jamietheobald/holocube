# several stimuli to test simple tracking

import holocube.hc5 as hc
import numpy as np
from scipy.signal import find_peaks

# length of each test
fps = 120
num_secs = 2
# we need num_frames to add each test
num_frames = num_secs * fps

# stimuli --- bar, white, and black points
stl = hc.stim.STL(hc.window, 'Flower.stl', scale=0.02)
stl.set_pz(-2)
# bar = hc.stim.cbarr_class(hc.window, dist=1)
pts = hc.stim.Points(hc.window, int(10 ** 4), dims=[(-5, 5), (-5, 5), (-5, 5)], color=1, pt_size=4)

# package the on off states of the stimuli I want to try
stims = [(True, False), (False, True), (True, True)]

# motion --- .5, 1 and 2 Hz turning
# arcsin of a sin is an quick hack to get triangular motion
trip5 = 0.5 * np.arcsin(np.sin(np.linspace(0, 2 * np.pi * num_secs * .5, num_frames))) * 180 / np.pi
tri1 = 0.5 * np.arcsin(np.sin(np.linspace(0, 2 * np.pi * num_secs * 1, num_frames))) * 180 / np.pi
tri2 = 0.5 * np.arcsin(np.sin(np.linspace(0, 2 * np.pi * num_secs * 2, num_frames))) * 180 / np.pi
# package the motion frequencies I want to try
turns = [trip5, tri1, tri2]

# add the exp
# starting the exp commands, executed once when exp starts
estarts = [[hc.window.set_far, 4],
           [hc.window.reset_pos_rot],
           [hc.window.set_bg, [0.5, 0.5, 0.5, 1.0]]]

# commands executed once after we finish
eends = [[hc.window.set_far, 2],
         [hc.window.set_bg, [0.0, 0.0, 0.0, 1.0]]]

# start a new exp that we can add tests to
hc.scheduler.add_exp(None, estarts, eends)

# add each test
test = 0  # number for num_flash

# run through all the condition combinations with nested loops
for stl_on, pts_on in stims:
    for turn in turns:
        # exp indicator flashes
        synch_flash, num_flash = hc.tools.test_bin_flash(test, num_frames)

        # start of each test, executed once at the start of each test
        starts = [[stl.switch, stl_on],
                  [pts.switch, pts_on],
                  ]

        # middle of each test, executed each frame of the test
        middles = [[stl.set_ry, turn],
                   [pts.set_ry, turn],
                   [hc.window.set_ref, 0, synch_flash],
                   [hc.window.set_ref, 1, num_flash],
                   ]

        # start of each test, executed once at the end of each test
        ends = [[stl.switch, False],
                [pts.switch, False],
                ]

        # add the test to the current exp
        hc.scheduler.add_test(num_frames, starts, middles, ends)
        # increment the counter for num_flash
        test += 1
