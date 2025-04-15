import holocube.hc5 as hc
import numpy as np
from pyglet.window import key

num_frames = np.inf

pts = hc.stim.Points(hc.window, int(10 ** 4), dims=[(-5, 5), (-5, 5), (-5, 5)], color=1, pt_size=1)

vs = np.array([[-1, 1, 0], [-1, -1, 1], [-1, -1, -1]])
vsx = [[-1, 1, 0, 1, 1, 1], [-1, -1, 1, -.3, -.3, .5], [-1, -1, -1, -.5, .5, 0]]
tri = hc.stim.Triangles(hc.window, vs)
tri2 = hc.stim.Triangles(hc.window, vs / 2, color=.5)

stl = hc.stim.STL(hc.window, '45_deg.stl', scale=0.02)

hc.scheduler.add_exp()

starts = [[hc.window.set_far, 3],
          [hc.window.add_keyhold_action, key.UP, hc.window.inc_pitch, .05],
          [hc.window.add_keyhold_action, key.DOWN, hc.window.inc_pitch, -.05],
          [hc.window.add_keyhold_action, key.LEFT, hc.window.inc_yaw, .05],
          [hc.window.add_keyhold_action, key.RIGHT, hc.window.inc_yaw, -.05],
          [hc.window.add_keyhold_action, (key.LEFT, key.MOD_SHIFT), hc.window.inc_roll, .05],
          [hc.window.add_keyhold_action, (key.RIGHT, key.MOD_SHIFT), hc.window.inc_roll, -.05],
          [hc.window.add_keyhold_action, (key.UP, key.MOD_CTRL), hc.window.inc_lift, .05],
          [hc.window.add_keyhold_action, (key.DOWN, key.MOD_CTRL), hc.window.inc_lift, -.05],
          [hc.window.add_keyhold_action, (key.LEFT, key.MOD_CTRL), hc.window.inc_slip, .05],
          [hc.window.add_keyhold_action, (key.RIGHT, key.MOD_CTRL), hc.window.inc_slip, -.05],
          [hc.window.add_keyhold_action, (key.UP, key.MOD_CTRL, key.MOD_SHIFT), hc.window.inc_thrust, -.05],
          [hc.window.add_keyhold_action, (key.DOWN, key.MOD_CTRL, key.MOD_SHIFT), hc.window.inc_thrust, .05],
          [hc.window.add_keypress_action, key.END, hc.window.reset_pos_rot],
          [hc.arduino.set_lmr_scale, -0.1],
          # [square.set_ry,                  0],
          # [square.switch,                  True],
          # [tri.switch, True],
          # [tri2.switch, True],
          [stl.set_pz, -1],
          [stl.switch, True],
          [pts.switch, True]]

# middles = [[square.next_frame           ]]
middles = []

ends = [  # [square.switch,             False],
    # [tri.switch, False],
    # [tri2.switch, False],
    [stl.switch, False],
    [pts.switch, False],
    [hc.window.remove_key_action, key.UP],
    [hc.window.remove_key_action, key.DOWN],
    [hc.window.remove_key_action, key.LEFT],
    [hc.window.remove_key_action, key.RIGHT],
    [hc.window.remove_key_action, key.END],
    [hc.window.remove_key_action, (key.LEFT, key.MOD_SHIFT)],
    [hc.window.remove_key_action, (key.RIGHT, key.MOD_SHIFT)],
    [hc.window.remove_key_action, (key.UP, key.MOD_CTRL)],
    [hc.window.remove_key_action, (key.DOWN, key.MOD_CTRL)],
    [hc.window.remove_key_action, (key.LEFT, key.MOD_CTRL)],
    [hc.window.remove_key_action, (key.RIGHT, key.MOD_CTRL)],
    [hc.window.remove_key_action, (key.UP, key.MOD_CTRL, key.MOD_SHIFT)],
    [hc.window.remove_key_action, (key.DOWN, key.MOD_CTRL, key.MOD_SHIFT)],
    [hc.window.set_far, 2],
    [hc.window.reset_pos_rot]]

# add the test
hc.scheduler.add_test(num_frames, starts, middles, ends)
