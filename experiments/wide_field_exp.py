# an example experiment that moves wide field dot arrays

import holocube.hc as hc
import numpy as np

# frames in a test
num_frames = 360
# motion cycles in a test
num_cycles = 6

# magnitude of translational and rotational motions
t_magnitude = 1
r_magnitude = 30

# declare the stimulus, a field of points
pts = hc.stim.Points(hc.window, 1000, extent=5)
# and how to move it in translation
trans_motion = t_magnitude*np.sin(np.linspace(0,num_cycles*2*np.pi, num_frames))
rot_motion = r_magnitude*np.sin(np.linspace(0,num_cycles*2*np.pi, num_frames))

exp_starts = [
    [hc.window.set_far, 3],
    [hc.window.set_bg, [0.1, 0.1, 0.1, 1.0]],
    [pts.switch, True],
    ]
# reset to black background when experiment is done, turn off points
exp_ends = [[hc.window.set_far, 1],
         [hc.window.set_bg, [0.0, 0.0, 0.0, 1.0]],
         [pts.switch, False]
         ]
# add the exp
hc.scheduler.add_exp(starts=exp_starts, ends=exp_ends)

# now add each test (without a for loop)
starts = []
ends = []

middles = [[pts.set_px, trans_motion]]
hc.scheduler.add_test(num_frames, starts, middles, ends)

middles = [[pts.set_py, trans_motion]]
hc.scheduler.add_test(num_frames, starts, middles, ends)

middles = [[pts.set_pz, trans_motion]]
hc.scheduler.add_test(num_frames, starts, middles, ends)

middles = [[pts.set_rx, rot_motion]]
hc.scheduler.add_test(num_frames, starts, middles, ends)

middles = [[pts.set_ry, rot_motion]]
hc.scheduler.add_test(num_frames, starts, middles, ends)

middles = [[pts.set_rz, rot_motion]]
hc.scheduler.add_test(num_frames, starts, middles, ends)