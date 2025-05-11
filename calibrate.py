#! /usr/bin/env python3

import pyglet
import holocore.hc as hc


# start the components
hc.control.start()
# hc.window.start(config_file='cal_viewport.config')
hc.window.start(config_file='test_viewport.config')
hc.arduino.start('dummy')
hc.scheduler.start(hc.window, hc.control,
                   randomize=False, default_rest_time=.1)
hc.scheduler.load_dir('calibrate', suffix='cal.py')
print('ready')

# for debugging
s = hc.scheduler
e = s.exps
w = hc.window
v = w.viewports[0]
c = hc.control
b = c.button_list
t = e[0].tests[0].starts
td = t[19][0].__self__
# d = e[0].tests[0].mids[-1][0].__self__

# run pyglet
pyglet.app.run()
