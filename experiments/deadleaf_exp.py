# try a few runs of the deadleaf stimulus

import holocube.hc as hc
import numpy as np

num_frames = 200
flower_amplitude = 45
flower_cycles = 2

leaves = hc.stim.Deadleaf(hc.window, 250, color=[.4,.6])
flower = hc.stim.Spherical_Segment(hc.window, 0,20,.9)
flower.set_rx(-90)
flower_move = np.arcsin(np.sin(np.linspace(0,flower_cycles*2*np.pi,num_frames)))
flower_move *= flower_amplitude/(np.pi/2)
flower_colors = [.3, .4, .5, .6, .7]

exp_starts = [[hc.window.set_bg, [0.5, 0.5, 0.5, 1.0]]]
exp_ends = [[hc.window.set_bg, [0.0, 0.0, 0.0, 1.0]]]
hc.scheduler.add_exp(starts=exp_starts, ends=exp_ends)

for flower_color in flower_colors:
    starts = [[leaves.on, True],
              [flower.update_colors, flower_color],
              [flower.on, True]]

    middles = [[leaves.move],
               [flower.set_ry, flower_move]
               ]

    ends = [[leaves.on, False],
            [flower.on, False]]

    hc.scheduler.add_test(num_frames, starts, middles, ends)
