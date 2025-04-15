# horizon test
expi = 1

# how long for the exp?
numframes = 2000

# a set of quads
forest = forest_class(window)

# the motions
msin = sin(linspace(0, 8 * pi, numframes))
tri = arcsin(sin(linspace(0, 8 * pi, numframes)))
tri_dir = array(sign(ediff1d(tri, None, 0)), dtype='int')
ampl = .1

lights = mod(cumsum(tri_dir), 3)

# first move the points above and below the horizon
for i in linspace(1, 2, 3):
    s.add_test(forest.on, 1, 0,
               window.ref_light, -1, 0,
               window.set_far, 30, 0,

               window.set_thrust, .01, 1,
               forest.set_px, 2 * sin(linspace(0, 32 * i * pi, numframes)), 1,
               window.ref_light, lights, 1,

               window.ref_color_4, array(spikelist(expi, numframes) * 255, dtype='int'), 1,

               forest.set_pos, array([0, 0, 0.]), -1,
               window.ref_light, -1, -1,
               window.set_far, 1, -1,
               hor.on, 0, -1,
               forest.on, 0, -1)
    expi += 1

s.save_exp()
