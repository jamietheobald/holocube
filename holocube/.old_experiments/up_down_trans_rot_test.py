# upper and lower cues for rotation and translation
expi = 1

# how long for the exp?
numframes = 800

# a set of points
uppts = pts_class2(window, 4000, dims=[(-3, 3), (0, 3), (-3, 3)], color=.5, pt_size=2)
dnpts = pts_class2(window, 4000, dims=[(-3, 3), (-3, 0), (-3, 3)], color=.5, pt_size=2)

# the motions
tri = arcsin(sin(linspace(0, 8 * pi, numframes)))
tri_dir = array(sign(ediff1d(tri, None, 0)), dtype='int')
tampl = .1
rampl = arcsin(.1 / (.5 ** (
            1 / 3.))) * 180 / pi  # this makes the mean offset of rotation the same as translation in the frontal visual field

lights = mod(cumsum(tri_dir), 3)

# first translation

# upper pts
s.add_test(window.ref_light, -1, 0,
           uppts.on, 1, 0,
           dnpts.on, 1, 0,

           uppts.set_px, tri * tampl, 1,
           window.ref_light, lights, 1,

           window.ref_color_4, array(spikelist(expi, numframes) * 255, dtype='int'), 1,

           uppts.set_pos, array([0, 0, 0.]), -1,
           dnpts.set_pos, array([0, 0, 0.]), -1,
           window.ref_light, -1, -1,
           uppts.on, 0, -1,
           dnpts.on, 0, -1)
expi += 1

# lower pts
s.add_test(window.ref_light, -1, 0,
           uppts.on, 1, 0,
           dnpts.on, 1, 0,

           dnpts.set_px, tri * tampl, 1,
           window.ref_light, lights, 1,

           window.ref_color_4, array(spikelist(expi, numframes) * 255, dtype='int'), 1,

           uppts.set_pos, array([0, 0, 0.]), -1,
           dnpts.set_pos, array([0, 0, 0.]), -1,
           window.ref_light, -1, -1,
           uppts.on, 0, -1,
           dnpts.on, 0, -1)
expi += 1

# both pts
s.add_test(window.ref_light, -1, 0,
           uppts.on, 1, 0,
           dnpts.on, 1, 0,

           uppts.set_px, tri * tampl, 1,
           dnpts.set_px, tri * tampl, 1,
           window.ref_light, lights, 1,

           window.ref_color_4, array(spikelist(expi, numframes) * 255, dtype='int'), 1,

           uppts.set_pos, array([0, 0, 0.]), -1,
           dnpts.set_pos, array([0, 0, 0.]), -1,
           window.ref_light, -1, -1,
           uppts.on, 0, -1,
           dnpts.on, 0, -1)
expi += 1

# next rotation

# upper pts
s.add_test(window.ref_light, -1, 0,
           uppts.on, 1, 0,
           dnpts.on, 1, 0,

           uppts.set_ry, tri * rampl, 1,
           window.ref_light, lights, 1,

           window.ref_color_4, array(spikelist(expi, numframes) * 255, dtype='int'), 1,

           uppts.set_pos, array([0, 0, 0.]), -1,
           dnpts.set_pos, array([0, 0, 0.]), -1,
           window.ref_light, -1, -1,
           uppts.on, 0, -1,
           dnpts.on, 0, -1)
expi += 1

# lower pts
s.add_test(window.ref_light, -1, 0,
           uppts.on, 1, 0,
           dnpts.on, 1, 0,

           dnpts.set_ry, tri * rampl, 1,
           window.ref_light, lights, 1,

           window.ref_color_4, array(spikelist(expi, numframes) * 255, dtype='int'), 1,

           uppts.set_pos, array([0, 0, 0.]), -1,
           dnpts.set_pos, array([0, 0, 0.]), -1,
           window.ref_light, -1, -1,
           uppts.on, 0, -1,
           dnpts.on, 0, -1)
expi += 1

# both pts
s.add_test(window.ref_light, -1, 0,
           uppts.on, 1, 0,
           dnpts.on, 1, 0,

           uppts.set_ry, tri * rampl, 1,
           dnpts.set_ry, tri * rampl, 1,
           window.ref_light, lights, 1,

           window.ref_color_4, array(spikelist(expi, numframes) * 255, dtype='int'), 1,

           uppts.set_pos, array([0, 0, 0.]), -1,
           dnpts.set_pos, array([0, 0, 0.]), -1,
           window.ref_light, -1, -1,
           uppts.on, 0, -1,
           dnpts.on, 0, -1)
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

s.save_exp('up down trans rot test')
