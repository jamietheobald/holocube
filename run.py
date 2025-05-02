#! /usr/bin/env python
import pyglet
import holocore.hc as hc

hc.control.start()
hc.window.start(config_file='test_viewport.config')
hc.arduino.start('dummy')
hc.scheduler.start(hc.window, hc.control,
                   randomize=False, default_rest_time=.1)
hc.scheduler.load_dir('experiments', suffix=('exp.py', 'rest.py'))
print('ready')

pyglet.app.run()
