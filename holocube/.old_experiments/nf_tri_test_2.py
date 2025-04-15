# nf exp #3
expi = 1

# how long for the exp?
numframes = 400

# two sets of points to move at different speeds
tpts = pts_class(window, 4000, dims=[3, 2, 5], color=(1, 1, 1), pt_size=3)
spts = pts_class(window, 4000, dims=[3, 2, 5], color=(1, 1, 1), pt_size=3)

tri = arcsin(sin(linspace(0, 4 * pi, numframes)))
tri_dir = array(sign(ediff1d(tri, None, 0)), dtype='int')
ampl = .1
lights = mod(cumsum(tri_dir), 3)

# construct a table of speeds to move against each other
spds = [1, 2, 3, 4]
for i in spds:
    for j in spds:
        expnumlights = array(spikelist(expi, numframes) * 255, dtype='int')
        s.add_test(tpts.on, 1, 0,
                   spts.on, 1, 0,
                   window.ref_light, 1, 0,

                   tpts.set_pz, linspace(0, i, numframes), 1,
                   tpts.set_px, tri * ampl, 1,
                   spts.set_pz, linspace(0, j, numframes), 1,
                   window.ref_light, lights, 1,
                   window.ref_color_4, expnumlights, 1,

                   tpts.set_pos, array([0, 0, 0.]), -1,
                   spts.set_pos, array([0, 0, 0.]), -1,
                   window.ref_light, -1, -1,
                   tpts.on, 0, -1,
                   spts.on, 0, -1)
        expi += 1

# move them at the same speed (2), with or without a triangle wave
for i in [0, 1]:
    expnumlights = array(spikelist(expi, numframes) * 255, dtype='int')
    s.add_test(tpts.on, 1, 0,
               spts.on, 1, 0,
               window.ref_light, -1, 0,

               tpts.set_pz, linspace(0, 2, numframes), 1,
               tpts.set_px, tri * ampl * i, 1,
               spts.set_pz, linspace(0, 2, numframes), 1,
               spts.set_px, tri * ampl * i, 1,
               window.ref_light, lights, 1,
               window.ref_color_4, expnumlights, 1,

               tpts.set_pos, array([0, 0, 0.]), -1,
               spts.set_pos, array([0, 0, 0.]), -1,
               window.ref_light, -1, -1,
               tpts.on, 0, -1,
               spts.on, 0, -1)
    expi += 1

# and a bar for tracking
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

# and save it in the scheduler
s.save_exp()
