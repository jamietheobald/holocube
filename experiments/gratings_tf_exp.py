# spatial

import holocore.hc as hc
import numpy as np

num_frames = 200
num_rest_frames = 60
frame_rate = 60

# inverted sfs, how many degrees for a cycle (assuming a square will
# cover 90 degrees)
tfs = np.array([0.05, 0.1, .2, .5, 1, 2, 5, 10, 20], dtype=float)
# sf, cycles/degree
isf = 20*np.pi/180 # inverted sf
sf = 1/isf
contrast = 1.
orientations = [0, 180]

sd = 0.25
pos = [0,0,-1]
iori = [0,0,1]
gratings = []

for tf in tfs:
    for orientation in orientations:
        grating = hc.stim.Grating(hc.window, rate=frame_rate,
                                  sf=sf, tf=tf, c=contrast, o=orientation,
                                  sd=sd, init_pos=pos, init_ori=iori, edge_size=2,
                                )
        gratings.append(grating)


# add the experiment
estarts = [[hc.window.set_far, 2],
           [hc.window.reset_pos_rot],
           [hc.window.set_bg, [0.5, 0.5, 0.5, 1.0]],
           ]

eends = [[hc.window.set_far, 5],
         [hc.window.set_bg, [0.0, 0.0, 0.0, 1.0]],
         ]
hc.scheduler.add_exp(starts=estarts, ends=eends)


# and add each test
for test_ind, grating in enumerate(gratings):
    starts = [
            [grating.switch, True],
              ]

    middles = [
        ]

    ends = [
        [grating.switch, False],
        ]

    # add the test
    hc.scheduler.add_test(num_frames, starts, middles, ends)

# add a blank rest screen
hc.scheduler.add_rest(num_rest_frames, [], [], [])