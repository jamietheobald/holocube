# bar test
expi = 1

aa = zeros((800), dtype='int')
aa[100:300] = 255
aa[500:700] = 255

s.add_test(bar.on, 1, 0,
           window.ref_color_1, 127, 0,
           bar.set_ry, arcsin(sin(linspace(0, 4 * pi, 800))) * 40, 1,
           ard.pip, array(spikelist(expi, 800) + 6, dtype='int'), 1,
           window.ref_color_1, aa, 1,
           window.ref_color_1, 127, -1,
           bar.on, 0, -1)

s.save_exp()
