# present a series of drifting gratings, over a range of spatial frequencies
# and a fixed temporal frequency

import holocore.hc as hc
import numpy as np

num_frames = 250
num_rest_frames = 60
frame_rate = 60

# inverted sfs, how many degrees for a cycle (assuming a square will
# cover 90 degrees). These are good frequencies for a fruit fly.
isfs = np.array([10, 12, 15, 20, 25, 32, 40, 50, 65, 80.])*np.pi/180
# sf, cycles/degree
sfs = 1./isfs
# one tf in Hz. Stripes should pass a point at this rate
tf = 1.
contrast = 1.
# test left and right
orientations = [0, 180]

sd = 0.25
pos = [0,0,-1]
iori = [0,0,1]

# this list will hold all the
gratings = []
# We use nested loops to generate gratings at each spatial frequency,
# and each of those at each orientation
for sf in sfs:
    for orientation in orientations:
        grating = hc.stim.Grating(hc.window, rate=frame_rate,
                                  sf=sf, tf=tf, c=contrast, o=orientation,
                                  sd=sd, init_pos=pos, init_ori=iori, edge_size=2,
                                )
        gratings.append(grating)

# timing dot structure. Position this to a screen or region that the subject
# can't see, but you can monitor with a camera or photodiode. You can signal
# the start and end time of tests, the test number, in case they are presented
# in a random order, or the onset of some visual stimulus.
td = hc.stim.Timing_Dots(hc.window, 2, side_len=.3, dot_side=.05, pos=(-1,-0.5,-0.5), ori=(1,0,0))

# add the experiment. This adds the commands to execute at the start
# and end of the whole experiment, such as switching the timing dots on,
# then off again
estarts = [[hc.window.set_far, 2],
           [hc.window.reset_pos_rot],
           [hc.window.set_bg, [0.5, 0.5, 0.5, 1.0]],
           [td.switch, True],
           ]

eends = [[hc.window.set_far, 5],
         [hc.window.set_bg, [0.0, 0.0, 0.0, 1.0]],
         [td.switch, False],
         ]
hc.scheduler.add_exp(starts=estarts, ends=eends)


# and add each test. Each one was added to the gratings list, so
# we can cycle through that and add to the scheduler at the
# end of each loop.
for test_ind, grating in enumerate(gratings):
    # here we generate the flashes for the timing dots structure
    # this one signals the beginning and end of the test
    test_ends_flashes = hc.tools.test_ends_flash(num_frames)
    # this one signals the number, starting with 1, in the order
    # we added them. Be careful that you have enough frames to
    # display all the flashes
    test_num_flashes = hc.tools.test_num_flash(num_frames, test_ind+1)

    # turn on the correct grating
    starts = [
            [grating.switch, True],
              ]

    # flash the dots
    middles = [
        [td.flash, 1, test_ends_flashes],
        [td.flash, 2, test_num_flashes],
        ]

    # turn off the current grating
    ends = [
        [grating.switch, False],
        ]

    # add the test
    hc.scheduler.add_test(num_frames, starts, middles, ends)

# add a blank rest screen. starts, middles and ends are all empty
hc.scheduler.add_rest(num_rest_frames, [], [], [])