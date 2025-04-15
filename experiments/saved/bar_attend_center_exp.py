# attention forward during bar motion

import holocube.hc5 as hc
import numpy as np

# length of each test
fps = 120
num_secs = 2
num_frames = num_secs * fps

stim_frames = 16

num_pts = 2000
pt_duration = 10
pt_speed = .00500

# stimuli --- bar and cohere sphere pts 
bar = hc.stim.cbarr_class(hc.window, dist=1)

pts = hc.stim.Dot_cohere_sph(hc.window, num_pts, pt_size=3, duration=pt_duration, speed=pt_speed)

# bar motion
lr = np.linspace(-90, 90, num_frames)
rl = -lr
bar_motions = [lr, rl]

# pt motion
pts.add_region(elevation=0, azimuth=0, flow_azimuth=-90, flow_elevation=0, coherence=0)
pts.add_region(elevation=0, azimuth=0, flow_azimuth=90, flow_elevation=0, coherence=0)
pts.add_region(elevation=0, azimuth=0, flow_azimuth=0, flow_elevation=90, coherence=0)
pts.add_region(elevation=0, azimuth=0, flow_azimuth=0, flow_elevation=-90, coherence=0)
pt_regions = [0, 1, 2, 3]

# pt test slots, before the bar arrives, when it is in front, and after
pre_start = int(np.floor(1 * num_frames / 4 - stim_frames / 2))
pre_stop = int(np.ceil(1 * num_frames / 4 + stim_frames / 2))
pre = np.zeros(num_frames)
pre[pre_start:pre_stop] = 1

on_start = int(np.floor(2 * num_frames / 4 - stim_frames / 2))
on_stop = int(np.ceil(2 * num_frames / 4 + stim_frames / 2))
on = np.zeros(num_frames)
on[on_start:on_stop] = 1

post_start = int(np.floor(3 * num_frames / 4 - stim_frames / 2))
post_stop = int(np.ceil(3 * num_frames / 4 + stim_frames / 2))
post = np.zeros(num_frames)
post[post_start:post_stop] = 1

test_spots = [pre, on, post]

# add the exp
estarts = [[hc.window.set_far, 4],
           [hc.window.reset_pos_rot],
           [hc.window.set_bg, [0.5, 0.5, 0.5, 1.0]]]

eends = [[hc.window.set_far, 2],
         [hc.window.set_bg, [0.0, 0.0, 0.0, 1.0]]]

hc.scheduler.add_exp(None, estarts, eends)

# add the tests
test = 0
for bar_motion in bar_motions:  # [lr, rl]
    for test_spot in test_spots:  # [pre, on, post]
        for pt_region in pt_regions:  # [0,1,2,3] -- l, r, up, down
            synch_flash, num_flash = hc.tools.test_bin_flash(test, num_frames)

            starts = [[pts.switch, True],
                      [bar.switch, True],
                      ]

            middles = [[pts.update_coherence, pt_region, test_spot],
                       [pts.move],
                       [bar.set_ry, bar_motion],
                       [hc.window.set_ref, 0, synch_flash],
                       [hc.window.set_ref, 1, num_flash],
                       [hc.window.set_ref, 2, test_spot * 255],
                       [hc.window.save_png],
                       ]

            ends = [[pts.switch, False],
                    [bar.switch, False],
                    [hc.window.unset_refs],
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
