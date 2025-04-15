# display a field of static points with line trails behind in the z
# direction to imply their motion

import holocube.windows as win
import holocube.scheduler as s
import holocube.stimuli as stim
from numpy import *

numframes = 800

# a set of points and lines
pts = stim.pts_class2(win.window, 16000, dims=[(-3, 3), (-2, 2), (-5, 5)], color=0., pt_size=3)
lines = stim.lines_class(win.window, 16000, dims=[(-3, 3), (-2, 2), (-5, 5)], color=.5, ln_width=1)

# set the start of line coords to the point coords
lines.coords[:, ::2] = pts.coords
lines.coords[:, 1::2] = lines.coords[:, ::2]  # end of line coords to starts
lines.coords[2, 1::2] -= .1  # move them backwards in Z (2)
lines.coords[0, lines.coords[0] < 0] -= .0001
lines.coords[0, lines.coords[0] > 0] += .0001
lines.coords[1, lines.coords[1] < 0] -= .0001
lines.coords[1, lines.coords[1] > 0] += .0001
lines.coords[2, lines.coords[2] < 0] -= .0001
lines.coords[2, lines.coords[2] > 0] += .0001

# draw the lines first
s.scheduler.add_test(lines.on, 1, 0,
                     pts.on, 1, 0,
                     win.window.ref_light, -1, 0,

                     pts.set_pz, linspace(0, 0, numframes), 1,

                     win.window.ref_light, -1, -1,
                     lines.on, 0, -1,
                     pts.on, 0, -1)

s.scheduler.save_exp('line_trails_demo')
