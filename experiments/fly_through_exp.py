# rest with a rotating bar

import holocube.hc as hc
import numpy as np


num_frames = np.inf

obj8 = hc.stim.Spherical_Segment(
    hc.window, polang_top=90,
    polang_bot=180, radius=5.5, color=0.2)

pts = hc.stim.Dot_Cohere_Sph(
    hc.window, num=2000, speed=0.002, duration=50
)
pts.add_region((0, 90), 18, (-90, 0), active=True)

obj1 = hc.stim.Regular_Polygon(
    hc.window, 6, rad=2,
    init_rot=0, init_pos=[-3, 0, 0], init_ori=[1, 0, 0], colors='ring0')

obj2 = hc.stim.Regular_Star_Polygon(
    hc.window, 5, rad1=2, turning_num=2, init_pos=[3, 0, 0], init_ori=[-1, 0, 0], colors='ring0')

obj3 = hc.stim.Regular_Polygon(
    hc.window, 4, rad=1.5,
    init_rot=0, init_pos=[0, -2, 0], init_ori=[0, 1, 0], colors='ring1')

mot = np.sin(np.linspace(0, 2 * np.pi, 120, endpoint=False)) / 2
mot2 = np.sin(np.linspace(0, 6 * np.pi, 120, endpoint=False)) / 2

obj4 = hc.stim.Sphere_Lines(hc.window, colors='ring')

obj5 = hc.stim.Grating(
    hc.window, sf=5 * 2 / np.pi, tf=1.0, sd=0.25, c=1, o=1 * np.pi / 4, xres=64, yres=64,
    edge_size=2, init_pos=[0, 0, -1], init_ori=[0, 0, 1])

obj6 = hc.stim.Bar(hc.window,
                   stripes=3,
                   color=[0.8, 0.0, 0.8])

# obj7 = hc.stim.STL(hc.window, 'Flower.stl', scale=0.02, color=0.6)
# obj7.set_pz(2)
#
# front_image = hc.stim.Image_File(hc.window, 1, 'front.png', init_pos=[0, 0, -0.65], init_ori=[0, 0, 1])
# left_image = hc.stim.Image_File(hc.window, 1, 'left.png', init_pos=[-0.65, 0, 0], init_ori=[1, 0, 0])
# right_image = hc.stim.Image_File(hc.window, 1, 'right.png', init_pos=[0.65, 0, 0], init_ori=[-1, 0, 0])
# back_image = hc.stim.Image_File(hc.window, 1, 'back.png', init_pos=[0, 0, 0.65], init_ori=[0, 0, -1])
# bottom_image = hc.stim.Image_File(hc.window, 1, 'bottom.png', init_pos=[0, -0.65, 0], init_ori=[0, 1, 0])
# top_image = hc.stim.Image_File(hc.window, 1, 'top.png', init_pos=[0, 0.65, 0], init_ori=[0, -1, 0])

# im = pyglet.image.load('front.png')
# ss = pyglet.sprite.Sprite(im, 0,0,-4, batch=hc.window.batch)

# add the experiment
hc.scheduler.add_exp()

starts = [[hc.window.set_far, 10],
          [hc.window.set_bg, [0.1, 0.1, 0.1, 1.0]],
          [hc.control.save_keys],

          [hc.control.add_key_action, 't', [hc.window.inc_pitch,  .05], 'hold'],
          [hc.control.add_key_action, 'w', [hc.window.inc_pitch, -.05], 'hold'],
          [hc.control.add_key_action, 'h', [hc.window.inc_yaw,    .05], 'hold'],
          [hc.control.add_key_action, 'n', [hc.window.inc_yaw,   -.05], 'hold'],
          [hc.control.add_key_action, 'g', [hc.window.inc_roll,   .05], 'hold'],
          [hc.control.add_key_action, 'r', [hc.window.inc_roll,  -.05], 'hold'],

          [hc.control.add_key_action, 'ctrl g', [hc.window.inc_lift, .05], 'hold'],
          [hc.control.add_key_action, 'ctrl r', [hc.window.inc_lift, -.05], 'hold'],
          [hc.control.add_key_action, 'ctrl h', [hc.window.inc_slip, -.05], 'hold'],
          [hc.control.add_key_action, 'ctrl n', [hc.window.inc_slip, .05], 'hold'],
          [hc.control.add_key_action, 'ctrl t', [hc.window.inc_thrust, -.05], 'hold'],
          [hc.control.add_key_action, 'ctrl w', [hc.window.inc_thrust, .05], 'hold'],
          [hc.control.add_key_action, 'o', [hc.window.reset_pos_rot]],

          [hc.arduino.set_lmr_scale, -0.1],

          [obj8.switch, True],
          [pts.switch, True],
          [obj1.switch, True],
          [obj2.switch, True],
          [obj3.switch, True],
          [obj4.switch, True],
          [obj5.switch, True],
          [obj6.switch, True],
          # [obj7.switch, True],
          # [front_image.switch, True],
          # [right_image.switch, True],
          # [left_image.switch, True],
          # [bottom_image.switch, True],
          # [top_image.switch, True],
          # [back_image.switch, True],
          ]

middles = [
    [obj1.inc_rx, .01],
    [obj2.set_rx, 1.5 * (mot + .33 * mot2)],
    [obj3.set_py, mot],
    [obj6.inc_ry, .01],
    # [obj1.inc_rx, 0.01],
    [pts.move]
]

ends = [
    [pts.switch, False],
    [obj1.switch, False],
    [obj2.switch, False],
    [obj3.switch, False],
    [obj4.switch, False],
    [obj5.switch, False],
    [obj6.switch, False],
    # [obj7.switch, False],
    [obj8.switch, False],
    # [front_image.switch, False],
    # [right_image.switch, False],
    # [left_image.switch, False],
    # [bottom_image.switch, False],
    # [top_image.switch, False],
    # [back_image.switch, False],
    [hc.control.restore_keys],
    [hc.window.set_far, 2],
    [hc.window.reset_pos_rot]]

# add the test
hc.scheduler.add_test(num_frames, starts, middles, ends)
