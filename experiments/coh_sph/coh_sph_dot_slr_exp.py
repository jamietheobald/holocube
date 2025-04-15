# complex scene attention exp
# import holocube.hc5 as hc
import holocube.hc5 as hc
import numpy as np

# num_frames = np.inf
num_frames = 360
num_pts = 2000
pt_duration = 10
pt_speed = .0050

pts = hc.stim.Dot_cohere_sph(hc.window, num_pts, pt_size=3, duration=pt_duration, speed=pt_speed)
pts.add_region(elevation=0, azimuth=-90, flow_azimuth=0, coherence=0)

disk = hc.stim.disk_class(hc.window, radius=.3, color=(0, 0, 0))

estarts = [[hc.window.set_far, 500],
           [hc.window.reset_pos_rot],
           [pts.on, True],
           [hc.window.set_bg, [0.9, 0.9, 0.9, 1.0]]]

eends = [[hc.window.set_far, 2],
         [pts.on, False],
         [hc.window.set_bg, [0.0, 0.0, 0.0, 1.0]]]

hc.scheduler.add_exp(None, estarts, eends)

azimuths = [-90, -45, 0, 45, 90]
dirs = [-90, 90]

disk_poss = [-90, 90]

ramp = np.zeros(num_frames)
ramp[60:300] = np.linspace(0, 1, 240)
ramp[300:] = 1.
ramp[-1] = 0

ramp_seq = np.zeros(num_frames, dtype='O')
for i in range(num_frames):
    ramp_seq[i] = (0, 0, 0)
for i in np.arange(11):
    ind = np.where(ramp >= i / 10)[0][0]
    ramp_seq[ind] = (0, 255, 0)

i = 0

for az in azimuths:
    for disk_pos in disk_poss:
        for d in dirs:
            exp_synch, exp_num = hc.tools.test_bin_flash(i, num_frames)
            i += 1

            starts = [[pts.update_region, 0, az, 0],
                      [pts.update_flow, 0, az + d, 0],
                      [disk.set_ry, disk_pos],
                      [disk.on, True]]

            middles = [[pts.update_coherence, 0, ramp],
                       [pts.move],
                       [hc.window.set_ref, 0, exp_synch],
                       [hc.window.set_ref, 1, exp_num],
                       [hc.window.set_ref, 2, ramp_seq]]

            ends = [[disk.on, False]]

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
           [pts.move],
           [pts.move],
           [pts.move],
           # [hc.window.save_png],
           ]

ends = [[rbar.switch, False]]

hc.scheduler.add_rest(num_frames, starts, middles, ends)
