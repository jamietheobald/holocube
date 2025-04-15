# 12 wide field experiments

import holocube.hc5 as hc
import numpy as np

# exp parameters
num_frames = 120
num_points = 10000
pt_size = 3
num_cycles = 4
rot_amp = 20
pos_amp = 0.5

# make the pts stim
pts = hc.stim.Points(hc.window, int(num_points), dims=[(-5, 5), (-5, 5), (-5, 5)], color=1, pt_size=pt_size)
pt_sizes = np.arange(1, 11)

# and the flower disk stim
flower = hc.stim.Disks(hc.window, .8, (1, 1, 1))
flower.set_pz(-1)
flower_ons = [False, True]

# 4pi -> 2 cycles of motion
angs = np.linspace(0, num_cycles * np.pi, num_frames)
# a hack for getting triangular motion, arcsine of a sine wave
tri_motion = np.arcsin(np.sin(angs))

# add the exp to the scheduler
estarts = [[hc.window.set_far, 3],
           [hc.window.reset_pos_rot]]

eends = [[hc.window.set_far, 1]]

hc.scheduler.add_exp(starts=estarts, ends=eends)

# now each test
test = 0
for flower_on in flower_ons:
    for pt_size in pt_sizes:
        synch_flash, num_flash = hc.tools.test_bin_flash(test, num_frames)

        starts = [[pts.switch, True],
                  [pts.set_pt_size, pt_size],
                  [flower.switch, flower_on],
                  ]

        middles = [[pts.set_px, pos_amp * tri_motion],
                   [hc.window.set_ref, 0, synch_flash],
                   [hc.window.set_ref, 1, num_flash],
                   ]

        ends = [[pts.switch, False],
                [pts.set_pt_size, 1],
                [pts.set_pos, np.array([0, 0, 0.])],
                [pts.set_rot, np.array([0, 0, 0.])],
                [flower.switch, False],
                ]

        hc.scheduler.add_test(num_frames, starts, middles, ends)

        test += 1

# add the rest, time or stimuli between tests
num_frames = 50

starts = []
middles = []
ends = []

hc.scheduler.add_rest(num_frames, starts, middles, ends)
