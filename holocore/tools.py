# make some tools for experiments and analysis

import numpy as np


def test_num_flash(test_num, num_frames, dist=10, col_1=255, col_2=96):
    '''take a test length and a number and produce a sequence to send
    to ref.set_ref_color.

    '''
    sl = np.zeros((num_frames), dtype='O')
    for i in range(num_frames):
        sl[i] = (0, 0, 0)
    for i in range(test_num):
        sl[(i + 2) * dist] = (0, col_2, 0)
    sl[0] = (0, col_1, 0)
    sl[-2] = (0, col_1, 0)
    return sl


def test_bin_flash(num, num_frames, dist=10, col_1=255, col_2=96):
    '''Return a pulsing synch pattern, to represent the start, end, and
    slots for a flash sequence, along with a binary flashes sequence
    in reverse order (least significant first) to represent a number.

    '''
    s1 = np.zeros((num_frames), dtype='O')
    s2 = np.zeros((num_frames), dtype='O')
    for i in range(num_frames):
        s1[i] = (0, 0, 0)
        s2[i] = (0, 0, 0)

    # make the synch flashes, usually every 10 frames
    s1[1::dist] = [(0, col_2, 0)]
    s1[1] = (0, col_1, 0)
    s1[-2] = (0, col_1, 0)
    s1[-1] = (0, 0, 0)

    # now the binary flashes
    bnum_rev = np.binary_repr(num)[::-1]
    for i in range(len(bnum_rev)):
        s2[i * 10 + 1] = (0, int(bnum_rev[i]) * col_1, 0)

    return s1, s2


def test_flash(frame_list, num_frames, color=255):
    '''Return a flash sequence with any frames highlighted

    '''
    s1 = np.zeros((num_frames), dtype='O')
    for i in range(num_frames):
        s1[i] = (0, 0, 0)
    for frame in frame_list:
        s1[frame] = (0, color, 0)
    return s1


# msequence from a matlab script

def mseq(baseVal, powerVal, shift=1, whichSeq=1):
    bitNum = baseVal ** powerVal - 1;
    register = np.ones([powerVal]);
    if baseVal == 2:
        if powerVal == 2:
            tap = [[1, 2]]
        elif powerVal == 3:
            tap = [[1, 3],
                   [2, 3]]
        elif powerVal == 4:
            tap = [[1, 4],
                   [3, 4]]
        elif powerVal == 5:
            tap = [[2, 5],
                   [3, 5],
                   [1, 2, 3, 5],
                   [2, 3, 4, 5],
                   [1, 2, 4, 5],
                   [1, 3, 4, 5]]
        elif powerVal == 6:
            tap = [[1, 6],
                   [5, 6],
                   [1, 2, 5, 6],
                   [1, 4, 5, 6],
                   [1, 3, 4, 6],
                   [2, 3, 5, 6]]
        elif powerVal == 7:
            tap = [[1, 7],
                   [6, 7],
                   [3, 7],
                   [4, 7],
                   [1, 2, 3, 7],
                   [4, 5, 6, 7],
                   [1, 2, 5, 7],
                   [2, 5, 6, 7],
                   [2, 3, 4, 7],
                   [3, 4, 5, 7],
                   [1, 3, 5, 7],
                   [2, 4, 6, 7],
                   [1, 3, 6, 7],
                   [1, 4, 6, 7],
                   [2, 3, 4, 5, 6, 7],
                   [1, 2, 3, 4, 5, 7],
                   [1, 2, 4, 5, 6, 7],
                   [1, 2, 3, 5, 6, 7]]
        elif powerVal == 8:
            tap = [[1, 2, 7, 8],
                   [1, 6, 7, 8],
                   [1, 3, 5, 8],
                   [3, 5, 7, 8],
                   [2, 3, 4, 8],
                   [4, 5, 6, 8],
                   [2, 3, 5, 8],
                   [3, 5, 6, 8],
                   [2, 3, 6, 8],
                   [2, 5, 6, 8],
                   [2, 3, 7, 8],
                   [1, 5, 6, 8],
                   [1, 2, 3, 4, 6, 8],
                   [2, 4, 5, 6, 7, 8],
                   [1, 2, 3, 6, 7, 8],
                   [1, 2, 5, 6, 7, 8]]
        elif powerVal == 9:
            tap = [[4, 9],
                   [5, 9],
                   [3, 4, 6, 9],
                   [3, 5, 6, 9],
                   [4, 5, 8, 9],
                   [1, 4, 5, 9],
                   [1, 4, 8, 9],
                   [1, 5, 8, 9],
                   [2, 3, 5, 9],
                   [4, 6, 7, 9],
                   [5, 6, 8, 9],
                   [1, 3, 4, 9],
                   [2, 7, 8, 9],
                   [1, 2, 7, 9],
                   [2, 4, 7, 9],
                   [2, 5, 7, 9],
                   [2, 4, 8, 9],
                   [1, 5, 7, 9],
                   [1, 2, 4, 5, 6, 9],
                   [3, 4, 5, 7, 8, 9],
                   [1, 3, 4, 6, 7, 9],
                   [2, 3, 5, 6, 8, 9],
                   [3, 5, 6, 7, 8, 9],
                   [1, 2, 3, 4, 6, 9],
                   [1, 5, 6, 7, 8, 9],
                   [1, 2, 3, 4, 8, 9],
                   [1, 2, 3, 7, 8, 9],
                   [1, 2, 6, 7, 8, 9],
                   [1, 3, 5, 6, 8, 9],
                   [1, 3, 4, 6, 8, 9],
                   [1, 2, 3, 5, 6, 9],
                   [3, 4, 6, 7, 8, 9],
                   [2, 3, 6, 7, 8, 9],
                   [1, 2, 3, 6, 7, 9],
                   [1, 4, 5, 6, 8, 9],
                   [1, 3, 4, 5, 8, 9],
                   [1, 3, 6, 7, 8, 9],
                   [1, 2, 3, 6, 8, 9],
                   [2, 3, 4, 5, 6, 9],
                   [3, 4, 5, 6, 7, 9],
                   [2, 4, 6, 7, 8, 9],
                   [1, 2, 3, 5, 7, 9],
                   [2, 3, 4, 5, 7, 9],
                   [2, 4, 5, 6, 7, 9],
                   [1, 2, 4, 5, 7, 9],
                   [2, 4, 5, 6, 7, 9],
                   [1, 3, 4, 5, 6, 7, 8, 9],
                   [1, 2, 3, 4, 5, 6, 8, 9]]
        elif powerVal == 10:
            tap = [[3, 10],
                   [7, 10],
                   [2, 3, 8, 10],
                   [2, 7, 8, 10],
                   [1, 3, 4, 10],
                   [6, 7, 9, 10],
                   [1, 5, 8, 10],
                   [2, 5, 9, 10],
                   [4, 5, 8, 10],
                   [2, 5, 6, 10],
                   [1, 4, 9, 10],
                   [1, 6, 9, 10],
                   [3, 4, 8, 10],
                   [2, 6, 7, 10],
                   [2, 3, 5, 10],
                   [5, 7, 8, 10],
                   [1, 2, 5, 10],
                   [5, 8, 9, 10],
                   [2, 4, 9, 10],
                   [1, 6, 8, 10],
                   [3, 7, 9, 10],
                   [1, 3, 7, 10],
                   [1, 2, 3, 5, 6, 10],
                   [4, 5, 7, 8, 9, 10],
                   [2, 3, 6, 8, 9, 10],
                   [1, 2, 4, 7, 8, 10],
                   [1, 5, 6, 8, 9, 10],
                   [1, 2, 4, 5, 9, 10],
                   [2, 5, 6, 7, 8, 10],
                   [2, 3, 4, 5, 8, 10],
                   [2, 4, 6, 8, 9, 10],
                   [1, 2, 4, 6, 8, 10],
                   [1, 2, 3, 7, 8, 10],
                   [2, 3, 7, 8, 9, 10],
                   [3, 4, 5, 8, 9, 10],
                   [1, 2, 5, 6, 7, 10],
                   [1, 4, 6, 7, 9, 10],
                   [1, 3, 4, 6, 9, 10],
                   [1, 2, 6, 8, 9, 10],
                   [1, 2, 4, 8, 9, 10],
                   [1, 4, 7, 8, 9, 10],
                   [1, 2, 3, 6, 9, 10],
                   [1, 2, 6, 7, 8, 10],
                   [2, 3, 4, 8, 9, 10],
                   [1, 2, 4, 6, 7, 10],
                   [3, 4, 6, 8, 9, 10],
                   [2, 4, 5, 7, 9, 10],
                   [1, 3, 5, 6, 8, 10],
                   [3, 4, 5, 6, 9, 10],
                   [1, 4, 5, 6, 7, 10],
                   [1, 3, 4, 5, 6, 7, 8, 10],
                   [2, 3, 4, 5, 6, 7, 9, 10],
                   [3, 4, 5, 6, 7, 8, 9, 10],
                   [1, 2, 3, 4, 5, 6, 7, 10],
                   [1, 2, 3, 4, 5, 6, 9, 10],
                   [1, 4, 5, 6, 7, 8, 9, 10],
                   [2, 3, 4, 5, 6, 8, 9, 10],
                   [1, 2, 4, 5, 6, 7, 8, 10],
                   [1, 2, 3, 4, 6, 7, 9, 10],
                   [1, 3, 4, 6, 7, 8, 9, 10]]
        elif powerVal == 11:
            tap = [[9, 11]]
        elif powerVal == 12:
            tap = [[6, 8, 11, 12]]
        elif powerVal == 13:
            tap = [[9, 10, 12, 13]]
        elif powerVal == 14:
            tap = [[4, 8, 13, 14]]
        elif powerVal == 15:
            tap = [[14, 15]]
        elif powerVal == 16:
            tap = [[4, 13, 15, 16]]
        elif powerVal == 17:
            tap = [[14, 17]]
        elif powerVal == 18:
            tap = [[11, 18]]
        elif powerVal == 19:
            tap = [[14, 17, 18, 19]]
        elif powerVal == 20:
            tap = [[17, 20]]
        elif powerVal == 21:
            tap = [[19, 21]]
        elif powerVal == 22:
            tap = [[21, 22]]
        elif powerVal == 23:
            tap = [[18, 23]]
        elif powerVal == 24:
            tap = [[17, 22, 23, 24]]
        elif powerVal == 25:
            tap = [[22, 25]]
        elif powerVal == 26:
            tap = [[20, 24, 25, 26]]
        elif powerVal == 27:
            tap = [[22, 25, 26, 27]]
        elif powerVal == 28:
            tap = [[25, 28]]
        elif powerVal == 29:
            tap = [[27, 29]]
        elif powerVal == 30:
            tap = [[7, 28, 29, 30]]
        else:
            print('M-sequence %d**%d is not defined' % (baseVal, powerVal))
    elif baseVal == 3:
        if powerVal == 2:
            tap = [[2, 1],
                   [1, 1]]
        elif powerVal == 3:
            tap = [[0, 1, 2],
                   [1, 0, 2],
                   [1, 2, 2],
                   [2, 1, 2]]
        elif powerVal == 4:
            tap = [[0, 0, 2, 1],
                   [0, 0, 1, 1],
                   [2, 0, 0, 1],
                   [2, 2, 1, 1],
                   [2, 1, 1, 1],
                   [1, 0, 0, 1],
                   [1, 2, 2, 1],
                   [1, 1, 2, 1]]
        elif powerVal == 5:
            tap = [[0, 0, 0, 1, 2],
                   [0, 0, 0, 1, 2],
                   [0, 0, 1, 2, 2],
                   [0, 2, 1, 0, 2],
                   [0, 2, 1, 1, 2],
                   [0, 1, 2, 0, 2],
                   [0, 1, 1, 2, 2],
                   [2, 0, 0, 1, 2],
                   [2, 0, 2, 0, 2],
                   [2, 0, 2, 2, 2],
                   [2, 2, 0, 2, 2],
                   [2, 2, 2, 1, 2],
                   [2, 2, 1, 2, 2],
                   [2, 1, 2, 2, 2],
                   [2, 1, 1, 0, 2],
                   [1, 0, 0, 0, 2],
                   [1, 0, 0, 2, 2],
                   [1, 0, 1, 1, 2],
                   [1, 2, 2, 2, 2],
                   [1, 1, 0, 1, 2],
                   [1, 1, 2, 0, 2]]
        elif powerVal == 6:
            tap = [[0, 0, 0, 0, 2, 1],
                   [0, 0, 0, 0, 1, 1],
                   [0, 0, 2, 0, 2, 1],
                   [0, 0, 1, 0, 1, 1],
                   [0, 2, 0, 1, 2, 1],
                   [0, 2, 0, 1, 1, 1],
                   [0, 2, 2, 0, 1, 1],
                   [0, 2, 2, 2, 1, 1],
                   [2, 1, 1, 1, 0, 1],
                   [1, 0, 0, 0, 0, 1],
                   [1, 0, 2, 1, 0, 1],
                   [1, 0, 1, 0, 0, 1],
                   [1, 0, 1, 2, 1, 1],
                   [1, 0, 1, 1, 1, 1],
                   [1, 2, 0, 2, 2, 1],
                   [1, 2, 0, 1, 0, 1],
                   [1, 2, 2, 1, 2, 1],
                   [1, 2, 1, 0, 1, 1],
                   [1, 2, 1, 2, 1, 1],
                   [1, 2, 1, 1, 2, 1],
                   [1, 1, 2, 1, 0, 1],
                   [1, 1, 1, 0, 1, 1],
                   [1, 1, 1, 2, 0, 1],
                   [1, 1, 1, 1, 1, 1]]
        elif powerVal == 7:
            tap = [[0, 0, 0, 0, 2, 1, 2],
                   [0, 0, 0, 0, 1, 0, 2],
                   [0, 0, 0, 2, 0, 2, 2],
                   [0, 0, 0, 2, 2, 2, 2],
                   [0, 0, 0, 2, 1, 0, 2],
                   [0, 0, 0, 1, 1, 2, 2],
                   [0, 0, 0, 1, 1, 1, 2],
                   [0, 0, 2, 2, 2, 0, 2],
                   [0, 0, 2, 2, 1, 2, 2],
                   [0, 0, 2, 1, 0, 0, 2],
                   [0, 0, 2, 1, 2, 2, 2],
                   [0, 0, 1, 0, 2, 1, 2],
                   [0, 0, 1, 0, 1, 1, 2],
                   [0, 0, 1, 1, 0, 1, 2],
                   [0, 0, 1, 1, 2, 0, 2],
                   [0, 2, 0, 0, 0, 2, 2],
                   [0, 2, 0, 0, 1, 0, 2],
                   [0, 2, 0, 0, 1, 1, 2],
                   [0, 2, 0, 2, 2, 0, 2],
                   [0, 2, 0, 2, 1, 2, 2],
                   [0, 2, 0, 1, 1, 0, 2],
                   [0, 2, 2, 0, 2, 0, 2],
                   [0, 2, 2, 0, 1, 2, 2],
                   [0, 2, 2, 2, 2, 1, 2],
                   [0, 2, 2, 2, 1, 0, 2],
                   [0, 2, 2, 1, 0, 1, 2],
                   [0, 2, 2, 1, 2, 2, 2]]
        else:
            print('M-sequence %d**%d is not defined' % (baseVal, powerVal))
    elif baseVal == 5:
        if powerVal == 2:
            tap = [[4, 3],
                   [3, 2],
                   [2, 2],
                   [1, 3]]
        if powerVal == 3:
            tap = [[0, 2, 3],
                   [4, 1, 2],
                   [3, 0, 2],
                   [3, 4, 2],
                   [3, 3, 3],
                   [3, 3, 2],
                   [3, 1, 3],
                   [2, 0, 3],
                   [2, 4, 3],
                   [2, 3, 3],
                   [2, 3, 2],
                   [2, 1, 2],
                   [1, 0, 2],
                   [1, 4, 3],
                   [1, 1, 3]]
        if powerVal == 4:
            tap = [[0, 4, 3, 3],
                   [0, 4, 3, 2],
                   [0, 4, 2, 3],
                   [0, 4, 2, 2],
                   [0, 1, 4, 3],
                   [0, 1, 4, 2],
                   [0, 1, 1, 3],
                   [0, 1, 1, 2],
                   [4, 0, 4, 2],
                   [4, 0, 3, 2],
                   [4, 0, 2, 3],
                   [4, 0, 1, 3],
                   [4, 4, 4, 2],
                   [4, 3, 0, 3],
                   [4, 3, 4, 3],
                   [4, 2, 0, 2],
                   [4, 2, 1, 3],
                   [4, 1, 1, 2],
                   [3, 0, 4, 2],
                   [3, 0, 3, 3],
                   [3, 0, 2, 2],
                   [3, 0, 1, 3],
                   [3, 4, 3, 2],
                   [3, 3, 0, 2],
                   [3, 3, 3, 3],
                   [3, 2, 0, 3],
                   [3, 2, 2, 3],
                   [3, 1, 2, 2],
                   [2, 0, 4, 3],
                   [2, 0, 3, 2],
                   [2, 0, 2, 3],
                   [2, 0, 1, 2],
                   [2, 4, 2, 2],
                   [2, 3, 0, 2],
                   [2, 3, 2, 3],
                   [2, 2, 0, 3],
                   [2, 2, 3, 3],
                   [2, 1, 3, 2],
                   [1, 0, 4, 3],
                   [1, 0, 3, 3],
                   [1, 0, 2, 2],
                   [1, 0, 1, 2],
                   [1, 4, 1, 2],
                   [1, 3, 0, 3],
                   [1, 3, 1, 3],
                   [1, 2, 0, 2],
                   [1, 2, 4, 3],
                   [1, 1, 4, 2]]
        else:
            print('M-sequence %d**%d is not defined' % (baseVal, powerVal))
    elif baseVal == 5:
        if powerVal == 2:
            tap = [[1, 1],
                   [1, 2]]
        else:
            print('M-sequence %d**%d is not defined' % (baseVal, powerVal))

    ms = np.zeros([bitNum])

    if whichSeq == None:
        whichSeq = np.ceil(rand(1) * len(tap))
    else:
        if (whichSeq > len(tap)) or (whichSeq < 1):
            print(' wrapping arround!')
            whichSeq = (whichSeq % len(tap)) + 1

    weights = np.zeros([powerVal])

    if baseVal == 2:
        weights[np.array(tap[whichSeq - 1]) - 1] = 1
    elif baseVal > 2:
        weights = tap[whichSeq - 1]

    for i in np.arange(bitNum):
        ms[i] = (np.dot(weights, register) + baseVal) % baseVal
        register[1:] = register[:-1]
        register[0] = ms[i]

    if not shift == None:
        shift = shift % len(ms)
        ms = np.concatenate([ms[shift:], ms[:shift]])

    if baseVal == 2:
        ms = ms * 2 - 1
    elif baseVal == 3:
        ms[ms == 2] = -1
    elif baseVal == 5:
        ms[ms == 4] = -1
        ms[ms == 2] = -2
    elif baseVal == 9:
        ms[ms == 5] = -1
        ms[ms == 6] = -2
        ms[ms == 7] = -3
        ms[ms == 8] = -4
    else:
        print('Wrong baseVal!')

    return ms
