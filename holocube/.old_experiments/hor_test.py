# horizon test
expi = 1

# how long for the exp?
numframes = 800

# a set of points
hpts1 = pts_class2(window, 4000, dims=[[-3, 3], [-.2, .2], [-5, 5]], color=.5, pt_size=3)

# and a horizon
hor = horizon_class(window, elevation=0, flipped=1)

# the motions
tri = arcsin(sin(linspace(0, 8 * pi, numframes)))
tri_dir = array(sign(ediff1d(tri, None, 0)), dtype='int')
ampl = .1

lights = mod(cumsum(tri_dir), 3)

# first move the points above and below the horizon
for i in linspace(-.2, .2, 5):
    s.add_test(hpts1.on, 1, 0,
               hor.set_elevation, 0, 0,
               hor.on, 1, 0,
               window.ref_light, -1, 0,
               hpts1.set_py, i, 0,

               hpts1.set_pz, linspace(0, 2, numframes), 1,
               hpts1.set_px, tri * ampl, 1,
               window.ref_light, lights, 1,

               window.ref_color_4, array(spikelist(expi, numframes) * 255, dtype='int'), 1,

               hpts1.set_pos, array([0, 0, 0.]), -1,
               window.ref_light, -1, -1,
               hor.on, 0, -1,
               hpts1.on, 0, -1)
    expi += 1

# next move the horizon up and down while the pts are static
for i in linspace(-20, 20, 5):
    s.add_test(hpts1.on, 1, 0,
               hor.set_elevation, i, 0,
               hor.on, 1, 0,
               window.ref_light, -1, 0,

               hpts1.set_pz, linspace(0, 2, numframes), 1,
               hpts1.set_px, tri * ampl, 1,
               window.ref_light, lights, 1,

               window.ref_color_4, array(spikelist(expi, numframes) * 255, dtype='int'), 1,

               hpts1.set_pos, array([0, 0, 0.]), -1,
               window.ref_light, -1, -1,
               hor.on, 0, -1,
               hpts1.on, 0, -1)
    expi += 1

# and a tracking bar
tri = arcsin(sin(linspace(0, 4 * pi, numframes))) * 40
tri_dir = array(sign(ediff1d(tri, None, 0)), dtype='int')
lights = mod(cumsum(tri_dir), 3)
expnumlights = array(spikelist(expi, numframes) * 255, dtype='int')

s.add_test(bar.on, 1, 0,
           window.ref_light, -1, 0,
           bar.set_ry, tri, 1,
           window.ref_light, lights, 1,
           window.ref_color_4, expnumlights, 1,
           window.ref_light, -1, -1,
           bar.on, 0, -1)

s.save_exp('horizon test')
