# complex scene attention exp
# import holocube.hc5 as hc
import holocube.hc5 as hc
import numpy as np

# num_frames = np.inf
num_frames = 200

square = hc.stim.Movable_grating(hc.window, np.array([[-1, 1, 1, -1], [-1, -1, 1, 1], [-2, -2, -2, -2]]),
                                 sf=5, tf=3, o=np.pi / 4, sd=.35)
# pts = hc.stim.Points(hc.window, int(10**4), dims=[(-5, 5),(-5, 5),(-5, 5)], color=1, pt_size=4)

hor = hc.stim.Horizon(hc.window, depth=-.5, dist=100, color=.3, )

isfs = np.array([10, 12, 15, 20, 25, 32, 40, 50, 65, 80.]) * np.pi / 180  # inverted sfs
sfs = 1. / isfs
tf = 20.

gratings = []
garr = np.array([[-1, 1, 1, -1], [-1, -1, 1, 1], [-2, -2, -2, -2]])
for sf in sfs:
    for orient in [0 * np.pi / 2, 2 * np.pi / 2]:
        g = hc.stim.Movable_grating(hc.window, garr, sf=sf, tf=3, o=orient, sd=.25)
        g.inc_rx(20)
        gratings.append(g)

estarts = [[hc.window.set_far, 500],
           [hc.window.reset_pos_rot],
           [hor.on, True],
           [hc.window.set_bg, [0.9, 0.9, 0.9, 1.0]]]

eends = [[hc.window.set_far, 2],
         [hor.on, False],
         [hc.window.set_bg, [0.0, 0.0, 0.0, 1.0]]]

hc.scheduler.add_exp(None, estarts, eends)

i = 0
for grating in gratings:
    exp_synch, exp_num = hc.tools.test_bin_flash(i, num_frames)
    i += 1

    starts = [[grating.on, True]]

    middles = [[grating.next_frame]]

    ends = [[grating.on, False]]

    hc.scheduler.add_test(num_frames, starts, middles, ends)

# add the rest
num_frames = 90
rbar = hc.stim.cbarr_class(hc.window, dist=1)

starts = [[hc.arduino.set_lmr_scale, -.1],
          [rbar.set_ry, -90],
          [rbar.switch, True],
          # [pts.replace_region, null_reg]
          ]

middles = [[rbar.inc_ry, hc.arduino.lmr],
           # [pts.move  ],
           # [pts.move  ],
           # [hc.window.save_png],
           ]

ends = [[rbar.switch, False]]

hc.scheduler.add_rest(num_frames, starts, middles, ends)
