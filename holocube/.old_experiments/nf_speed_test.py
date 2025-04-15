# nf exp test for speed dependence with the triangle wave tracking
expi = 1

# how long for the exp?
numframes = 800

# two sets of points to move at different speeds
apts = pts_class(window, 4000, dims=[3, 2, 5], color=1, pt_size=3)
bpts = pts_class(window, 4000, dims=[3, 2, 5], color=1, pt_size=3)

tri = arcsin(sin(linspace(0, 8 * pi, numframes)))
tri_dir = array(sign(ediff1d(tri, None, 0)), dtype='int')
ampl = .1

lights = mod(cumsum(tri_dir), 3)

# first move them at the same speed, with or without a triangle wave
for i in [0, 1, 2, 3]:
    s.add_test(apts.on, 1, 0,
               bpts.on, 1, 0,
               window.ref_light, -1, 0,

               apts.set_pz, linspace(0, i, numframes), 1,
               apts.set_px, tri * ampl, 1,
               bpts.set_pz, linspace(0, i, numframes), 1,
               bpts.set_px, tri * ampl, 1,
               window.ref_light, lights, 1,
               window.ref_color_4, array(spikelist(expi, numframes) * 255, dtype='int'), 1,

               apts.set_pos, array([0, 0, 0.]), -1,
               bpts.set_pos, array([0, 0, 0.]), -1,
               window.ref_light, -1, -1,
               apts.on, 0, -1,
               bpts.on, 0, -1)
    expi += 1

# and a bar for tracking

tri = arcsin(sin(linspace(0, 4 * pi, numframes))) * 40
tri_dir = array(sign(ediff1d(tri, None, 0)), dtype='int')
lights = mod(cumsum(tri_dir), 3)
expnumlights = array(spikelist(expi, numframes) * 255, dtype='int')

s.add_test(bar.on, 1, 0,
           window.ref_light, -1, 0,
           bar.set_ry, tri, 1,
           # ard.pip,            array(spikelist(expi, 800)+6, dtype='int'), 1,
           window.ref_light, lights, 1,
           window.ref_color_4, expnumlights, 1,
           window.ref_light, -1, -1,
           bar.on, 0, -1)

# and save it in the scheduler
s.save_exp()
