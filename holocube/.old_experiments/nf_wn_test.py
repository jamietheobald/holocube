# nf exp #3
expi = 1

# how long for the exp?
numframes = 511

# two sets of points to move at different speeds
apts = pts_class(window, 4000, dims=[3, 2, 5], color=(1, 1, 1), pt_size=3)
bpts = pts_class(window, 4000, dims=[3, 2, 5], color=(1, 1, 1), pt_size=3)

wn = mseq.mseq(2, 9)
wn_sig = array(wn, dtype='int')
ampl = .02

lights = mod(cumsum(wn), 3)

# first move them at the same speed, with or without a sawtooth wave
for i in [0, 1]:
    s.add_test(apts.on, 1, 0,
               bpts.on, 1, 0,
               window.ref_light, -1, 0,

               apts.set_pz, linspace(0, 2, numframes), 1,
               apts.inc_px, wn * ampl * i, 1,
               bpts.set_pz, linspace(0, 2, numframes), 1,
               bpts.inc_px, wn * ampl * i, 1,
               window.ref_light, lights, 1,
               # ard.pip,             wn_sig+4,                             1,
               # ard.pip,  array(spikelist(expi, numframes)+6, dtype='int'),1,
               window.ref_color_4, array(spikelist(expi, numframes) * 255, dtype='int'), 1,

               apts.set_pos, array([0, 0, 0.]), -1,
               bpts.set_pos, array([0, 0, 0.]), -1,
               window.ref_light, -1, -1,
               apts.on, 0, -1,
               bpts.on, 0, -1)
    expi += 1

# next move only one set of points as a sawtooth while the speed of the other varies
spds = [1, 2, 4]
for i in spds:
    s.add_test(apts.on, 1, 0,
               bpts.on, 1, 0,
               window.ref_light, -1, 0,

               apts.set_pz, linspace(0, 2, numframes), 1,
               apts.inc_px, wn * ampl, 1,
               bpts.set_pz, linspace(0, i, numframes), 1,
               window.ref_light, lights, 1,
               # ard.pip,             wn_sig+4,                             1,
               # ard.pip,  array(spikelist(expi, numframes)+6, dtype='int'),1, # which trial?
               window.ref_color_4, array(spikelist(expi, numframes) * 255, dtype='int'), 1,

               apts.set_pos, array([0, 0, 0.]), -1,
               bpts.set_pos, array([0, 0, 0.]), -1,
               window.ref_light, -1, -1,
               apts.on, 0, -1,
               bpts.on, 0, -1)
    expi += 1

# then vary the speed of the sawtooth pts while the other stays fixed
spds = [1, 2, 4]
for i in spds:
    s.add_test(apts.on, 1, 0,
               bpts.on, 1, 0,
               window.ref_light, -1, 0,

               apts.set_pz, linspace(0, i, numframes), 1,
               apts.inc_px, wn * ampl, 1,
               bpts.set_pz, linspace(0, 2, numframes), 1,
               window.ref_light, lights, 1,
               # ard.pip,             wn_sig+4,                             1,
               # ard.pip,  array(spikelist(expi, numframes)+6, dtype='int'),1, # which trial?
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
           window.ref_light, -1, -1,
           bar.set_ry, tri, 1,
           # ard.pip,            array(spikelist(expi, 800)+6, dtype='int'), 1,
           window.ref_light, lights, 1,
           window.ref_color_4, expnumlights, 1,
           window.ref_light, -1, -1,
           bar.on, 0, -1)

# and save it in the scheduler
s.save_exp()
