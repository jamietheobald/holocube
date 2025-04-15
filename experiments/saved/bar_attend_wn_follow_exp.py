# attention forward during bar motion
# 0 r->l bar, rl wn, signalling reg 1 (left) in ref pt 3 and 4
# 1 r->l bar, rl wn, signalling reg 2 (right) in ref pt 3 and 4
# 2 r->l bar, rl wn, signalling reg 1&2 (both) in ref pt 3 and 4

# 3 r->l bar, ud wn, signalling reg 1 (left) in ref pt 3 and 4
# 4 r->l bar, ud wn, signalling reg 2 (right) in ref pt 3 and 4
# 5 r->l bar, ud wn, signalling reg 1&2 (both) in ref pt 3 and 4

# 6 l->r bar, rl wn, signalling reg 1 (left) in ref pt 3 and 4
# 7 l->r bar, rl wn, signalling reg 2 (right) in ref pt 3 and 4
# 8 l->r bar, rl wn, signalling reg 1&2 (both) in ref pt 3 and 4

# 9 l->r bar, ud wn, signalling reg 1 (left) in ref pt 3 and 4
# 10 l->r bar, ud wn, signalling reg 2 (right) in ref pt 3 and 4
# 11 l->r bar, ud wn, signalling reg 1&2 (both) in ref pt 3 and 4

import holocube.hc5 as hc
import numpy as np

# length of each test
# num_frames = 255
num_frames = 511

num_pts = 2000
pt_duration = 10
pt_speed = .00500

# stimuli --- bar and cohere sphere pts 
bar = hc.stim.cbarr_class(hc.window, dist=1)

pts = hc.stim.Dot_cohere_sph(hc.window, num_pts, pt_size=3, duration=pt_duration, speed=pt_speed, new_az=True)

# bar motion
rl = np.linspace(-90, 90, num_frames)
lr = -rl
bar_poss = [rl, lr]

# pt motion, left--right, up--down
pts.add_region(elevation=0, azimuth=0, flow_azimuth=90, flow_elevation=0, coherence=0)
pts.add_region(elevation=0, azimuth=0, flow_azimuth=90, flow_elevation=0, coherence=0)
pts.add_region(elevation=0, azimuth=0, flow_azimuth=0, flow_elevation=90, coherence=0)
pts.add_region(elevation=0, azimuth=0, flow_azimuth=0, flow_elevation=90, coherence=0)
pt_dirs = [[0, 1], [2, 3]]

# pt region motions
lreg_dir = hc.tools.mseq(2, 9, whichSeq=1)
rreg_dir = hc.tools.mseq(2, 9, whichSeq=2)

# pt dir flash tracking
# track the forward with 2 channels, the rear with 2
# then try to track both together
lplus = np.array([255 if item > 0 else 0 for item in lreg_dir])
lminus = np.array([255 if item < 0 else 0 for item in lreg_dir])
rplus = np.array([255 if item > 0 else 0 for item in rreg_dir])
rminus = np.array([255 if item < 0 else 0 for item in rreg_dir])
lpm = np.array([255 if item > 0 else 64 for item in lreg_dir])
rpm = np.array([255 if item > 0 else 64 for item in rreg_dir])

flash_trackers = [[lplus, lminus], [rplus, rminus], [lpm, rpm]]

# add the exp
estarts = [[hc.window.set_far, 4],
           [hc.window.reset_pos_rot],
           [hc.window.set_bg, [0.5, 0.5, 0.5, 1.0]]]

eends = [[hc.window.set_far, 2],
         [hc.window.set_bg, [0.0, 0.0, 0.0, 1.0]]]

hc.scheduler.add_exp(None, estarts, eends)

# add the tests
test = 0
for bar_pos in bar_poss:
    for pt_dir in pt_dirs:
        for flash_tracker in flash_trackers:
            synch_flash, num_flash = hc.tools.test_bin_flash(test, num_frames)
            ft1, ft2 = flash_tracker

            starts = [[pts.switch, True],
                      [bar.switch, True],
                      [pts.update_coherence, pt_dir[0], 1],
                      [pts.update_coherence, pt_dir[1], 1],
                      ]

            middles = [[bar.set_ry, bar_pos],
                       [pts.update_region, pt_dir[0], bar_pos + 15],
                       [pts.update_flow, pt_dir[0], bar_pos + 105],
                       [pts.update_speed, pt_dir[0], lreg_dir * pt_speed],
                       [pts.update_region, pt_dir[1], bar_pos - 15],
                       [pts.update_flow, pt_dir[1], bar_pos - 105],
                       [pts.update_speed, pt_dir[1], rreg_dir * pt_speed],
                       [pts.move],

                       [hc.window.set_ref, 0, synch_flash],
                       [hc.window.set_ref, 1, num_flash],
                       [hc.window.set_ref, 2, ft1],
                       [hc.window.set_ref, 3, ft2],
                       ]

            ends = [[pts.switch, False],
                    [bar.switch, False],
                    [hc.window.unset_refs],
                    [pts.update_coherence, pt_dir[0], 0],
                    [pts.update_coherence, pt_dir[1], 0],
                    ]

            hc.scheduler.add_test(num_frames, starts, middles, ends)
            test += 1

# add the rest
num_frames = 90
rbar = hc.stim.cbarr_class(hc.window, dist=1)

starts = [[hc.arduino.set_lmr_scale, -.1],
          [rbar.set_ry, -90],
          [rbar.switch, True],
          ]

middles = [[rbar.inc_ry, hc.arduino.lmr],
           ]

ends = [[rbar.switch, False],
        ]

hc.scheduler.add_rest(num_frames, starts, middles, ends)
