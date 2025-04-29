# A looming disk coming from different directions

# always need hc, and np is usually useful
import holocube.hc as hc
import numpy as np

# how many frames for each test
num_frames = 300
# where does the disk begin
init_dist = -10

# a random cloud of points in a sphere
pts = hc.stim.Points(hc.window, num=1000, extent=np.abs(init_dist))

# a polygon with enough sides to look like a disk
disk = hc.stim.Regular_Polygon(hc.window, num_sides=32, init_ori=[0,0,-1], init_pos=[0,0,0])
# each time it will go from the initial distance to 0, the observer
dist = np.linspace(init_dist, 0, num_frames)
# directions for it to come from, in degrees
angs = np.linspace(0, 360, 8, endpoint=False)


# add the experiment
# change the viewing distance, background color, and turn on ambient points
exp_starts = [[hc.window.set_far, init_dist],
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

# and add each test
for ang in angs:
    # put the disk at its initial distance, turn the camera and points
    # to the proper angle, and make the disk visible.
    starts = [
        [disk.set_pz, init_dist],
        [hc.window.set_ry, ang],
        [pts.set_ry, ang],
        [disk.switch, True],
    ]

    # list the dists for the disk, frame by frame. The scheduler interprets
    # an array not as a single argument, but series of values for each
    # subsequent frame
    middles = [[disk.set_pz, dist]]

    # turn the disk back off
    ends =  [[disk.switch, False],
             ]

    # add the test
    hc.scheduler.add_test(num_frames, starts, middles, ends)
