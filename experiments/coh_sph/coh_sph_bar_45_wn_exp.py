# complex scene attention exp
# import holocube.hc5 as hc
import holocube.hc5 as hc
import numpy as np

# num_frames = np.inf
num_frames = 1023
num_pts = 2000
pt_duration = 10
pt_speed = .00500

pts = hc.stim.Dot_cohere_sph(hc.window, num_pts, pt_size=3, duration=pt_duration, speed=pt_speed)
pts.add_region(elevation=0, azimuth=-90, flow_azimuth=0, coherence=0)

bar = hc.stim.cbarr_class(hc.window, dist=1)

estarts = [[hc.window.set_far, 500],
           [hc.window.reset_pos_rot],
           [pts.on, True],
           [hc.window.set_bg, [0.9, 0.9, 0.9, 1.0]]]

eends = [[hc.window.set_far, 2],
         [pts.on, False],
         [hc.window.set_bg, [0.0, 0.0, 0.0, 1.0]]]

hc.scheduler.add_exp(None, estarts, eends)

azimuths = [-90, -45, 0, 45, 90]

bar_poss = [-45, 45]

wn = hc.tools.mseq(2, 8)
wn_flash_l = np.array([(0, 0 if v == -1 else 100, 0) for v in wn])
wn_flash_r = np.array([(0, 0 if v == 1 else 100, 0) for v in wn])

i = 0
for az in azimuths:
    for bar_pos in bar_poss:
        exp_synch, exp_num = hc.tools.test_bin_flash(i, num_frames)
        i += 1

        starts = [[pts.update_coherence, 0, 1],
                  [pts.update_region, 0, az, 0],
                  [bar.set_ry, bar_pos],
                  [bar.on, True]]

        middles = [[pts.update_flow, 0, az + wn * 90, 0],
                   [pts.move],
                   [hc.window.set_ref, 0, exp_synch],
                   [hc.window.set_ref, 1, exp_num],
                   [hc.window.set_ref, 2, wn_flash_l],
                   [hc.window.set_ref, 3, wn_flash_r]
                   ]

        ends = [[bar.on, False],
                [hc.window.set_ref, 0, (0, 0, 0)],
                [hc.window.set_ref, 1, (0, 0, 0)],
                [hc.window.set_ref, 2, (0, 0, 0)],
                [hc.window.set_ref, 3, (0, 0, 0)],
                [pts.update_coherence, 0, 0]]

        hc.scheduler.add_test(num_frames, starts, middles, ends)

# add the rest
num_frames = 90
rbar = hc.stim.cbarr_class(hc.window, dist=1)

starts = [[hc.arduino.set_lmr_scale, -.1],
          [rbar.set_ry, -90],
          [rbar.switch, True],
          # [pts.replace_region, null_reg]
          ]

middles = [[rbar.inc_ry, hc.arduino.lmr],
           [pts.move],
           # [pts.move  ],
           # [pts.move  ],
           # [hc.window.save_png],
           ]

ends = [[rbar.switch, False]]

hc.scheduler.add_rest(num_frames, starts, middles, ends)
