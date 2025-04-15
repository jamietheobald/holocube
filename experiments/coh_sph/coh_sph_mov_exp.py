# complex scene attention exp
# import holocube.hc5 as hc
import holocube.hc5 as hc
import numpy as np

# num_frames = np.inf
num_frames = 512
num_pts = 3000
pt_duration = 10
pt_speed = .0050

pts = hc.stim.Dot_cohere_sph(hc.window, num_pts, pt_size=3, duration=pt_duration, speed=pt_speed)
pts.add_region(elevation=0, azimuth=-90, flow_azimuth=0, coherence=0)
pts.add_region(elevation=0, azimuth=0, flow_azimuth=90, coherence=1)
pts.add_region(elevation=0, azimuth=90, flow_azimuth=180, flow_elevation=90, coherence=0)

h = hc.stim.Horizon(hc.window, -1, 20, color=.5)

tree = hc.stim.Tree(hc.window, 4)

disk = hc.stim.disk_class(hc.window, radius=0.2, color=(2, 2, 2))

sl = hc.stim.sphere_lines_class(hc.window, 18, 128)

estarts = [[hc.window.set_far, 500],
           [hc.window.reset_pos_rot],
           [pts.on, True],
           [hc.window.set_bg, [0.8, 0.8, 0.8, 1.0]]]

eends = [[hc.window.set_far, 2],
         [pts.on, False],
         [hc.window.set_bg, [0.0, 0.0, 0.0, 1.0]]]

hc.scheduler.add_exp(None, estarts, eends)

wn = hc.tools.mseq(2, 9)

azimuths = [-90, -45, 0, 45, 90]
dirs = [-90, 90]

# ramp0 = np.zeros(num_frames)
# ramp0[60:260] = np.linspace(0,1,200)
# ramp0[260:360] = -1

# ramp1 = np.zeros(num_frames)
# ramp1[460:660] = np.linspace(0,1,200)
# ramp1[660:760] = 1

# ramp2 = np.zeros(num_frames)
# ramp2[860:1060] = np.linspace(0,1,200)
# ramp2[1060:1160] = 1


# ramp_seq = np.zeros(num_frames, dtype='O')
# for i in range(num_frames):
#     ramp_seq[i] = (0,0,0)
# for i in np.arange(11):
#     ind = np.where(ramp>=i/10)[0][0]
#     ramp_seq[ind] = (0,255,0)

i = 0

az = 0
d = dirs[0]
disk_pos = 110

wnn = np.array(wn * 90)
print(wnn)

starts = [
    # [pts.update_region, 0, az, 0],
    # [pts.update_flow, 0, az+d, 0],
    [h.on, True],
    [disk.set_rx, -60],
    [disk.set_ry, disk_pos],
    [disk.on, True],
    [tree.set_rx, -5],
    [tree.set_ry, -30],
    [tree.on, True],
    [sl.on, 0]
]

middles = [[pts.update_coherence, 0, 0],
           [pts.update_flow, 1, wnn, 0],
           [pts.update_coherence, 2, 0],
           [pts.move],
           [hc.window.save_png]
           ]

ends = [[disk.on, False],
        [h.on, False],
        [tree.on, False],
        [sl.on, False]
        ]

hc.scheduler.add_test(num_frames, starts, middles, ends)

# add the rest
num_frames = 0
rbar = hc.stim.cbarr_class(hc.window, dist=1)

starts = [[hc.arduino.set_lmr_scale, -.1],
          [rbar.set_ry, -90],
          [rbar.switch, False],
          # [pts.replace_region, null_reg]
          ]

middles = [[rbar.inc_ry, hc.arduino.lmr],
           [pts.move],
           # [pts.move  ],
           # [pts.move  ],
           # [hc.window.save_png],
           ]

ends = [[rbar.switch, False]]

hc.scheduler.add_rest(num_frames, starts, middles, ends)
