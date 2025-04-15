# oriented bars exp

numframes = 150

bars = bars_class(window, o=0, xlim=[-3, 3], ylim=[-3, 3])

upmotion = zeros((numframes)) + .01
leftmotion = zeros((numframes)) - .01
rightmotion = zeros((numframes)) + .01

expi = 1
for o in [0, pi / 2]:
    for h in [.05, .1, .2]:
        for xdir in [leftmotion, rightmotion]:
            expnumlights = array(spikelist(expi, numframes) * 255, dtype='int')
            s.add_test(bars.orient, o, 0,
                       bars.d_height, h, 0,
                       bars.on, 1, 0,
                       window.ref_light, -1, 0,

                       bars.inc_py, upmotion, 1,
                       bars.inc_px, xdir, 1,
                       window.ref_color_4, expnumlights, 1,

                       window.ref_light, -1, -1,
                       bars.set_pos, zeros(3), -1,
                       bars.on, 0, -1)
            expi += 1

s.save_exp()
