# complex scene attention coherent flower

import holocube.hc5 as hc
import numpy as np

# num_frames = np.inf
num_frames = 600
num_pts = 5000
pt_duration = 20
pt_speed = 0.0050
pt_size = 4

# back = hc.stim.Deadleaf(hc.window)

back1 = hc.stim.Deadleaf(hc.window, color=[0.7, 0.8, 0.75])
back2 = hc.stim.Deadleaf(hc.window, color=[0.7, 0.8, 0.85])

estarts = [[hc.window.set_far, 500],
           [hc.window.reset_pos_rot],
           [hc.window.set_bg, [0.75, 0.75, 0.75, 1.0]]]

eends = [[hc.window.set_far, 2],
         [hc.window.set_bg, [0.0, 0.0, 0.0, 1.0]]]

hc.scheduler.add_exp(None, estarts, eends)

azimuths = [-90, -45, 0, 45, 90]
dirs = [-90, 90]

ramp = np.zeros(num_frames)
ramp[60:300] = np.linspace(0, 1, 240)
ramp[300:] = 1.
ramp[-1] = 0

az = 45 * np.sin(np.linspace(0, 2 * 2 * np.pi, num_frames))

mv = np.sign(np.cos(np.linspace(0, 2 * 2 * np.pi, num_frames)))

for back in [back1, back2]:
    starts = [
        [back.on, True],
        # [pts.update_region, 0, az[0], 0],
        # [pts.update_coherence, 0, 0.0]
    ]

    middles = [
        # [pts.move_flower, 0, np.diff(az)],
        [back.move_back],
        [back.move_flower, mv],
        # [hc.window.save_png],
    ]

    ends = [  # [pts.move],
        [back.on, False],
    ]

    hc.scheduler.add_test(num_frames, starts, middles, ends)

# add the rest
num_frames = 0
rbar = hc.stim.cbarr_class(hc.window, dist=1)

starts = [[hc.arduino.set_lmr_scale, -.1],
          [rbar.set_ry, -90],
          [rbar.switch, True],
          # [pts.replace_region, null_reg]
          ]

middles = [[rbar.inc_ry, hc.arduino.lmr],
           # [pts.move  ],
           # [pts.move  ],
           # [pts.move  ],
           # [hc.window.save_png],
           ]

ends = [[rbar.switch, False]]

hc.scheduler.add_rest(num_frames, starts, middles, ends)
