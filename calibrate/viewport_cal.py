#! /usr/bin/env python
# Move and resize viewports, then save a configuration file

import pyglet
from pyglet.window import key
from numpy import *
import holocore.hc as hc
import os

num_frames = inf

s = hc.stim.Sphere_Lines(hc.window, colors='ring1')

front_image = hc.stim.Image_File(hc.window, 1, 'front.png', init_pos=[0, 0, -.75], init_ori=[0, 0, 1])
left_image = hc.stim.Image_File(hc.window, 1, 'left.png', init_pos=[-.75, 0, 0], init_ori=[1, 0, 0])
right_image = hc.stim.Image_File(hc.window, 1, 'right.png', init_pos=[.75, 0, 0], init_ori=[-1, 0, 0])
back_image = hc.stim.Image_File(hc.window, 1, 'back.png', init_pos=[0, 0, .75], init_ori=[0, 0, -1])
bottom_image = hc.stim.Image_File(hc.window, 1, 'bottom.png', init_pos=[0, -.75, 0], init_ori=[0, 1, 0])
top_image = hc.stim.Image_File(hc.window, 1, 'top.png', init_pos=[0, .75, 0], init_ori=[0, -1, 0])

# shorthand key mods
sh = key.MOD_SHIFT
ct = key.MOD_CTRL
sc = key.MOD_CTRL + key.MOD_SHIFT

# add the experiment
hc.scheduler.add_exp()

starts = [[hc.window.set_far, 5],
          [front_image.switch, True],
          [right_image.switch, True],
          [left_image.switch, True],
          [bottom_image.switch, True],
          [top_image.switch, True],
          [back_image.switch, True],
          [s.switch, True],
          [hc.window.viewport_inc_ind, 0],

          [hc.control.add_key_action, 'tab', hc.window.viewport_inc_ind],
          [hc.control.add_key_action, 'up', [hc.window.viewport_set_val, 'bottom', 1, 'increment'], 'hold'],
          [hc.control.add_key_action, 'down', [hc.window.viewport_set_val, 'bottom', -1, 'increment'], 'hold'],
          [hc.control.add_key_action, 'left', [hc.window.viewport_set_val, 'left', -1, 'increment'], 'hold'],
          [hc.control.add_key_action, 'right', [hc.window.viewport_set_val, 'left', 1, 'increment'], 'hold'],
          [hc.control.add_key_action, 'shift up', [hc.window.viewport_set_val, 'height', 1, 'increment'], 'hold'],
          [hc.control.add_key_action, 'shift down', [hc.window.viewport_set_val, 'height', -1, 'increment'], 'hold'],
          [hc.control.add_key_action, 'shift left', [hc.window.viewport_set_val, 'width', -1, 'increment'], 'hold'],
          [hc.control.add_key_action, 'shift right', [hc.window.viewport_set_val, 'width', 1, 'increment'], 'hold'],
          [hc.control.add_key_action, 'ctrl up', [hc.window.viewport_set_val, 'tilt', 1, 'increment'], 'hold'],
          [hc.control.add_key_action, 'ctrl down', [hc.window.viewport_set_val, 'tilt', -1, 'increment'], 'hold'],
          [hc.control.add_key_action, 'ctrl left', [hc.window.viewport_set_val, 'pan', -1, 'increment'], 'hold'],
          [hc.control.add_key_action, 'ctrl right', [hc.window.viewport_set_val, 'pan', 1, 'increment'], 'hold'],
          [hc.control.add_key_action, 'shift ctrl left', [hc.window.viewport_set_val, 'dutch', -1, 'increment'], 'hold'],
          [hc.control.add_key_action, 'shift ctrl right', [hc.window.viewport_set_val, 'dutch', 1, 'increment'], 'hold'],
          [hc.control.add_key_action, 'x', [hc.window.viewport_set_val, 'scalex', 1, 'increment']],
          [hc.control.add_key_action, 'y', [hc.window.viewport_set_val, 'scaley', 1, 'increment']],
          [hc.control.add_key_action, '1', [hc.window.viewport_set_val, 'fleft', .01, 'increment']],
          [hc.control.add_key_action, '2', [hc.window.viewport_set_val, 'fleft', -.01, 'increment']],
          [hc.control.add_key_action, '3', [hc.window.viewport_set_val, 'fright', .01, 'increment']],
          [hc.control.add_key_action, '4', [hc.window.viewport_set_val, 'fright', -.01, 'increment']],
          [hc.control.add_key_action, '5', [hc.window.viewport_set_val, 'ftop', .01, 'increment']],
          [hc.control.add_key_action, '6', [hc.window.viewport_set_val, 'ftop', -.01, 'increment']],
          [hc.control.add_key_action, '7', [hc.window.viewport_set_val, 'fbottom', .01, 'increment']],
          [hc.control.add_key_action, '8', [hc.window.viewport_set_val, 'fbottom', -.01, 'increment']],
          [hc.control.add_key_action, '9', [hc.window.viewport_set_val, 'near', .01, 'increment']],
          [hc.control.add_key_action, '0', [hc.window.viewport_set_val, 'near', -.01, 'increment']],
          [hc.control.add_key_action, '9', [hc.window.viewport_set_val, 'near', .01, 'increment']],
          [hc.control.add_key_action, '0', [hc.window.viewport_set_val, 'near', -.01, 'increment']],
          [hc.control.add_key_action, 'enter', [hc.window.save_config, 'test_viewport.config']]
          ]

middles = []

ends = [[hc.window.set_far, 2],
        [front_image.switch, False],
        [right_image.switch, False],
        [left_image.switch, False],
        [bottom_image.switch, False],
        [top_image.switch, False],
        [back_image.switch, False],
        [s.switch, False],
        [hc.window.remove_key_action, key.TAB],
        [hc.window.remove_key_action, key.UP],
        [hc.window.remove_key_action, key.DOWN],
        [hc.window.remove_key_action, key.LEFT],
        [hc.window.remove_key_action, key.RIGHT],
        [hc.window.remove_key_action, (key.UP, sh)],
        [hc.window.remove_key_action, (key.DOWN, sh)],
        [hc.window.remove_key_action, (key.LEFT, sh)],
        [hc.window.remove_key_action, (key.RIGHT, sh)],
        [hc.window.remove_key_action, (key.UP, ct)],
        [hc.window.remove_key_action, (key.DOWN, ct)],
        [hc.window.remove_key_action, (key.LEFT, ct)],
        [hc.window.remove_key_action, (key.RIGHT, ct)],
        [hc.window.remove_key_action, (key.LEFT, sc)],
        [hc.window.remove_key_action, (key.RIGHT, sc)],
        [hc.window.remove_key_action, key.X],
        [hc.window.remove_key_action, key.Y],
        [hc.window.remove_key_action, key.ENTER],
        [hc.window.viewport_set_val, 'bg', [0.0, 0.0, 0.0, 1.0], 'set', 'all']
        ]

hc.scheduler.add_test(num_frames, starts, middles, ends)
