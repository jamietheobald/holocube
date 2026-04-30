# multiple firefly groups flashing

# always need hc, and np is usually useful
import holocore.hc as hc
import numpy as np


# monitor frame rate
fps = 60
# how many frames for each test
num_frames = 600
# how long is each frame
frame_ms = 1000 / fps
# and how long total
t_ms = (np.arange(num_frames) + 0.5) * frame_ms
# time between tests
num_frames_rest = 120

# number of fireflies in each group (for 2 groups)
num_g1, num_g2 = 2, 8
# their pixel sizes
size = 5

# where will each group appear (degrees offset in yaw) in different tests? (as many pairings as you like)
angles = [[-30, 30], [30, -30]]

# different flash patterns for each group, alternate time off and time on, in ms.
# this: [400, 100] is 400ms off followed by 100 on, which produce 2 flashes per second
# this: [800, 200, 800, 200, 5000] is 2 flashes 800 ms off, 200 on, 800 off, 200 on, then a 5 sec break
# then, put pairs next to each other for testing against one another, for each group
flashes = [
    [[400, 100], [400, 100]],
    [[750, 100], [400, 100]],
    [[400, 100], [750, 100]],
    [[750, 100], [750, 100]],
]

# make the frame-by-frame flash arrays
brightnesses = []

for pattern_pair in flashes:
    pair_brightnesses = []

    for pattern in pattern_pair:
        edges = np.cumsum(pattern)
        phase = t_ms % edges[-1]
        segment = np.searchsorted(edges, phase, side="right")
        pair_brightnesses.append(segment % 2)

    brightnesses.append(pair_brightnesses)

brightnesses = np.array(brightnesses, dtype=np.uint8)

# make the groups,
v = np.random.randn(3, num_g1) * 0.1 # random spread around 0,0,0
v[2] = -1 # but set z to -1, in front of viewer
grp1 = hc.stim.Points(hc.window, num=num_g1, colors=1, pt_size=size, verts=v)

v = np.random.randn(3, num_g2) * 0.1 # random spread around 0,0,0
v[2] = -1 # but set z to -1, in front of viewer
grp2 = hc.stim.Points(hc.window, num=num_g2, colors=1, pt_size=size, verts=v)


# add the experiment
# change the viewing distance, background color, and turn on ambient points
exp_starts = [[hc.window.set_far, 2],
           [hc.window.set_bg, [0.1, 0.1, 0.1, 1.0]],
           ]

# reset to black background when experiment is done, turn off points
exp_ends = [[hc.window.set_far, 1],
         [hc.window.set_bg, [0.0, 0.0, 0.0, 1.0]],
         ]
# add the exp
hc.scheduler.add_exp(starts=exp_starts, ends=exp_ends)

# and add each test
for brightness in brightnesses:
    for angle in angles:
        starts = [
            # rotate the groups
            [grp1.set_ry, angle[0]],
            [grp2.set_ry, angle[1]],
            # make them visible
            [grp1.switch, True],
            [grp2.switch, True],
            ]

        middles = [
            # set brightness for each frame
            [grp1.update_colors, brightness[0]],
            [grp2.update_colors, brightness[1]],
        ]

        ends =  [
            # turn them back off
                [grp1.switch, False],
                [grp2.switch, False],
            ]

        # add the test
        hc.scheduler.add_test(num_frames, starts, middles, ends)

hc.scheduler.add_rest(120, [], [], [])