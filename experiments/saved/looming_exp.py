# slip only in frontal annulus. change ff vel in that annulus
import holocube.hc5 as hc
from numpy import *
from numpy.random import randn

# how long for the exp?
numframes = 1000

alg = randn(3, numframes) * .01
ralg = randn(3, 10) * .01

# single round of experiments (4 total trials)
stim_vs = [0.0125]
numpoints = [10000]

# multiple rounds of experiments (3 x 4 x numpoints)
# stim_vs = [0.0125, 0.015, 0.0175]
# numpoints = linspace(100,10**6,10)

# motion = [0, 90, 180, 270]
motion = [270]

# disk = hc.stim.disk_class(hc.window, .2,(0,0,0))
disk = hc.stim.Disks(hc.window, .2, (0, 0, 0))
disk.set_pz(-5)

hc.scheduler.add_exp()

## experiments
for i_numpoint, numpoint in enumerate(numpoints):
    for i_stim_v, stim_v in enumerate(stim_vs):
        for mot_ind, mot in enumerate(motion):
            stim_v_ind_seq = hc.tools.test_num_flash(i_stim_v + 1, numframes)
            cam_rot_ind_seq = hc.tools.test_num_flash(mot_ind + 2, numframes)
            numpoint_ind_seq = hc.tools.test_num_flash(i_numpoint + 3, numframes)

            points = hc.stim.Points(hc.window, int(numpoint), dims=[(-10, 10), (-10, 10), (-10, 10)], color=0,
                                    pt_size=.1)

            # disk = disks[mot_ind]
            starts = [[disk.switch, True],
                      [disk.set_ry, mot],
                      [hc.window.set_bg, [1, 1, 1, 1]],
                      [hc.window.set_far, 10],
                      [points.switch, True],
                      ]

            middles = [[disk.inc_pz, stim_v],
                       [points.inc_px, alg[0]],
                       [points.inc_py, alg[1]],
                       [points.inc_pz, alg[2]],
                       [hc.window.save_png],
                       # [hc.window.set_ref, 0, stim_v_ind_seq],
                       # [hc.window.set_ref, 1, cam_rot_ind_seq],
                       # [hc.window.set_ref, 2, numpoint_ind_seq],
                       ]

            ends = [[disk.switch, True],
                    [disk.set_ry, 0],
                    [disk.set_pz, -5],
                    [disk.set_px, 0],
                    [disk.set_py, 0],
                    [hc.window.set_far, 1],
                    [hc.window.set_bg, [0, 0, 0, 1]],
                    [points.switch, False]
                    ]
            hc.scheduler.add_test(numframes, starts, middles, ends)

# add the rest
num_frames = 1

starts = []
middles = [[points.inc_px, ralg[0]],
           [points.inc_py, ralg[1]],
           [points.inc_pz, ralg[2]],
           ]
ends = []

hc.scheduler.add_rest(num_frames, starts, middles, ends)
