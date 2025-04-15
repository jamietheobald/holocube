#! /usr/bin/env python
# run.py
import pyglet
import holocube.hc as hc

hc.control.start()
hc.window.start(config_file='test_viewport.config')
hc.arduino.start('dummy')
hc.scheduler.start(hc.window, hc.control,
                   randomize=False, default_rest_time=.1)
hc.scheduler.load_dir('experiments', suffix=('exp.py', 'rest.py'))
print('ready')

# for debugging
w = hc.window
s = hc.scheduler
t = hc.scheduler.exps[0].tests[0].starts

pyglet.app.run()
