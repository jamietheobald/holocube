# load a starmap and display

import holocube.hc as hc
import numpy as np


num_frames = np.inf
stars = np.load('sky.npy')
num_pts = len(stars)

colors = np.array([[r,g,b,a] for r,g,b,a in stars[:,4:]])

verts = np.array([[x,y,z] for x,y,z in stars[:,1:4]])

# pts = hc.stim.Points(hc.window, num_pts, color=cols, verts=vs)
pts = hc.stim.Points(hc.window, num_pts, verts=verts.T, colors=colors.T)

estarts = [[hc.window.set_far, 2],
           [hc.window.set_bg, [0.0, 0.0, 0.0, 1.0]],
           [pts.switch, True],
           ]

eends = [[hc.window.set_far, 5],
         [hc.window.set_bg, [0.0, 0.0, 0.0, 1.0]],
         [pts.switch, False],
         ]

# add the experiment
hc.scheduler.add_exp(starts=estarts, ends=eends)

starts = [
    [hc.control.save_keys],

    [hc.control.add_key_action, 'up', [hc.window.inc_pitch, .05], 'hold'],
    [hc.control.add_key_action, 'down', [hc.window.inc_pitch, -.05], 'hold'],
    [hc.control.add_key_action, 'left', [hc.window.inc_yaw, .05], 'hold'],
    [hc.control.add_key_action, 'right', [hc.window.inc_yaw, -.05], 'hold'],
    [hc.control.add_key_action, 'o', [hc.window.reset_pos_rot]],
          ]

middles = [

]

ends = [
    [hc.control.restore_keys,]
    ]

# add the test
hc.scheduler.add_test(num_frames, starts, middles, ends)
