# point density test
expi = 1

# how long for the exp?
numframes = 800

# a set of points
pts = pts_class2(window, 4000, dims=[(-3, 3), (-3, 3), (-3, 3)], color=.5, pt_size=2)

# the motions
tri = arcsin(sin(linspace(0, 8 * pi, numframes)))
tri_dir = array(sign(ediff1d(tri, None, 0)), dtype='int')
ampl = .1

lights = mod(cumsum(tri_dir), 3)

ptsset = [1000, 2154, 4642, 10000, 21544, 46416, 100000]

# show triangular tracking of points with different densities
for i in ptsset:
    s.add_test(window.ref_light, -1, 0,
               pts.set_num, i, 0,
               pts.on, 1, 0,

               pts.set_pz, linspace(0, 2, numframes), 1,
               pts.set_px, tri * ampl, 1,
               window.ref_light, lights, 1,

               window.ref_color_4, array(spikelist(expi, numframes) * 255, dtype='int'), 1,

               pts.set_pos, array([0, 0, 0.]), -1,
               window.ref_light, -1, -1,
               pts.on, 0, -1)
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
