# rest with a rotating bar

import holocube.hc as hc
import numpy as np


num_frames = 200

pts = hc.stim.Dot_Cohere_Sph(
    hc.window, num=2000, speed=0., duration=np.inf
)

pts.set_distributed_regions( 15, (0,0), 30, speed=.01, rel_flow_center=(90, 0))
# pts.view_centers()


# add the experiment
estarts = [[hc.window.set_far, 2],
           [hc.window.set_bg, [0.1, 0.1, 0.1, 1.0]],
           [pts.switch, True],
           ]

eends = [[hc.window.set_far, 5],
         [hc.window.set_bg, [0.0, 0.0, 0.0, 1.0]],
         [pts.switch, False]
         ]

hc.scheduler.add_exp(starts=estarts, ends=eends)

# and add each test
for test_ind in range(len(pts.regions)):
    starts = [
            [pts.activate_region, test_ind],
            [pts.print_active_region_azel]
              ]

    middles = [
        [pts.move]
        ]

    ends = [
        [pts.deactivate_region, test_ind]
        ]

    # add the test
    hc.scheduler.add_test(num_frames, starts, middles, ends)
