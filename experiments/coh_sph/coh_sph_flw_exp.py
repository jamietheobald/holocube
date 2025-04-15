# complex scene attention coherent flower

import holocube.hc5 as hc
import numpy as np

# num_frames = np.inf
num_frames = 600
num_pts = 5000
pt_duration = 20
pt_speed = 0.0050
pt_size = 4

pts = hc.stim.Incoherent_flower(hc.window, num_pts, pt_size=pt_size, duration=pt_duration, speed=pt_speed)

pts.add_region(elevation=30, azimuth=0, flow_azimuth=90, coherence=1, speed=0.005, radius=8)

estarts = [[hc.window.set_far, 500],
           [hc.window.reset_pos_rot],
           [pts.on, True],
           [hc.window.set_bg, [0.9, 0.9, 0.9, 1.0]]]

eends = [[hc.window.set_far, 2],
         [pts.on, False],
         [hc.window.set_bg, [0.0, 0.0, 0.0, 1.0]]]

hc.scheduler.add_exp(None, estarts, eends)

azimuths = [-90, -45, 0, 45, 90]
dirs = [-90, 90]

ramp = np.zeros(num_frames)
ramp[60:300] = np.linspace(0, 1, 240)
ramp[300:] = 1.
ramp[-1] = 0

az = 45 * np.sin(np.linspace(0, 2 * 2 * np.pi, num_frames))

starts = [[pts.update_region, 0, az[0], 0],
          [pts.update_coherence, 0, 0.0]
          ]

middles = [
    [pts.move_flower, 0, np.diff(az)],
    [pts.move],
    # [hc.window.save_png],
]

ends = [  # [pts.move],
]

hc.scheduler.add_test(num_frames, starts, middles, ends)

starts = [[pts.update_region, 0, az[0], 0],
          [pts.update_coherence, 0, 0.1]
          ]
hc.scheduler.add_test(num_frames, starts, middles, ends)

starts = [[pts.update_region, 0, az[0], 0],
          [pts.update_coherence, 0, 0.5]
          ]
hc.scheduler.add_test(num_frames, starts, middles, ends)

starts = [[pts.update_region, 0, az[0], 0],
          [pts.update_coherence, 0, 1]
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
           [pts.move],
           [pts.move],
           [pts.move],
           # [hc.window.save_png],
           ]

ends = [[rbar.switch, False]]

hc.scheduler.add_rest(num_frames, starts, middles, ends)
