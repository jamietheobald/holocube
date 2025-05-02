"""Viewport and window classes

Classes for holocore. Window is a subclass of a pyglet window, and
starting it requires a viewport config file, which specifies the
dimensions of the main window, and the locations and projection
details of each viewport.

"""

import pyglet
from pyglet.gl import *
from pyglet.window import key
# from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.math import Mat4, Vec3

import numpy as np
# from os.path import expanduser
import time
import configparser
import ast
import numbers


class Viewport:
    """A class for a single window configuration information.

    """

    def __init__(self, batch, name='vp', coords=None,
                 scale_factors=None, projection='perspective',
                 frustum=None, pan=0.0, tilt=0.0, dutch=0.0):
        """Set up the projection parameters and location on screen
        :param batch:
        :param name:
        :param coords:
        :param scale_factors:
        :param projection:
        :param frustum:
        :param pan:
        :param tilt:
        :param dutch:
        """
        if frustum is None:
            frustum = [-.1, .1, -.1, .1, .1, 2.0]
        if scale_factors is None:
            scale_factors = [1., 1., 1]
        if coords is None:
            coords = [0, 0, 150, 150]

        self.batch = batch
        self.name = name
        self.coords = np.array(coords, dtype='int')  # window left, bottom, width, height
        self.orth_coords = self.coords
        self.scale_factors = scale_factors  # x y and z flips for mirror reflections
        self.pan = pan
        self.tilt = tilt
        self.dutch = dutch
        # calculate the intrinsic camera rotation matrix
        self.cam_rot_mat = self.set_cam_rot_mat()
        # self.rmat = dot(rotmat([0.,1.,0.], azimuth), rotmat([1.,0.,0.], elevation))
        self.projection = projection  # orthographic or perspective projection, or ref
        # self.rotmat = dot(rotmat([0., 1., 0.], pan*pi/180), rotmat([1., 0., 0.], tilt*pi/180))
        self.forward_up = np.array([[0, 0], [0, 1], [-1, 0.]])
        self.ref_vl = None
        if projection.startswith('persp'):
            self.project = 'persp'
            self.view = self.view_perspective
        [left, right, bottom, top, frust_near, frust_far] = frustum
        self.frustum = np.array([left, right, bottom, top, frust_near, frust_far])  # left right bottom top near far

        # set the projection matrix
        self.aspect = (right - left) / (top - bottom)
        self.fov = np.degrees(2 * np.arctan((top - bottom) / (2 * frust_near)))
        self.projection_mat = Mat4.perspective_projection(
            fov=self.fov,
            aspect=self.aspect,
            z_near=frust_near,
            z_far=frust_far)

        self.sprite_pos = 0
        self.clear_color = [0., 0., 0., 1.]

    def set_cam_rot_mat(self):
        """Set the rotation matrix for the viewport from the pan,
        tilt, and dutch angle of the virtual camera, set as intrinsic
        fields before calling this function. Right now this is 4x
        faster than using rotate methods on a Mat4s.

        """
        # to do these only once, get the radians
        pan_rad = np.radians(self.pan)
        tilt_rad = np.radians(self.tilt)
        dutch_rad = np.radians(self.dutch)
        # and each matrix
        sint, cost = np.sin(pan_rad), np.cos(pan_rad)
        pan_mat = np.array([[cost, 0, sint], [0, 1, 0], [-sint, 0, cost]])  # yaw of the camera
        sint, cost = np.sin(tilt_rad), np.cos(tilt_rad)
        tilt_mat = np.array([[1, 0, 0], [0, cost, -sint], [0, sint, cost]])  # pitch of the camera
        sint, cost = np.sin(dutch_rad), np.cos(dutch_rad)
        dutch_mat = np.array([[cost, -sint, 0], [sint, cost, 0], [0, 0, 1]])  # dutch (roll) of the camera

        return dutch_mat @ tilt_mat @ pan_mat

    def view_perspective(self, pos, rot):
        """calculate the view for a given

        """
        # soon I can figure out how to change this to pyglet mats and
        # vecs --- or maybe this is faster?
        view = np.dot(np.dot(rot, self.cam_rot_mat), self.forward_up)

        look = Mat4.look_at(
            position=Vec3(*pos),
            target=Vec3(pos[0] + view[0, 0],
                        pos[1] + view[1, 0],
                        pos[2] + view[2, 0]),
            up=Vec3(view[0, 1],
                    view[1, 1],
                    view[2, 1]))

        return look

    def set_val(self, field, val, increment=False):
        """Set any of the viewport positional and rotational parameters by
        name"""
        match field:
            case 'left':
                self.coords[0] = self.coords[0] * increment + val
            case 'bottom':
                self.coords[1] = self.coords[1] * increment + val
            case 'width':
                self.coords[2] = self.coords[2] * increment + val
            case 'height':
                self.coords[3] = self.coords[3] * increment + val
        # projection params
            case 'fleft':
                self.frustum[0] = self.frustum[0] * increment + val
                print(f'{self.frustum[0]}')
            case 'fright':
                self.frustum[1] = self.frustum[1] * increment + val
            case 'fbottom':
                self.frustum[2] = self.frustum[2] * increment + val
            case 'ftop':
                self.frustum[3] = self.frustum[3] * increment + val
            case 'near':
                self.frustum[4] = self.frustum[4] * increment + val
            case 'far':
                self.frustum[5] = self.frustum[5] * increment + val
            # scaling for mirror reflections
            case 'scalex':
                self.scale_factors[0] *= -1
            case 'scaley':
                self.scale_factors[1] *= -1
            case 'scalez':
                self.scale_factors[2] *= -1
            # cam angles to the viewport
            case 'pan':
                self.pan = self.pan * increment + val
                self.set_cam_rot_mat()
            case 'tilt':
                self.tilt = self.tilt * increment + val
                self.set_cam_rot_mat()
            case 'dutch':
                self.dutch = self.dutch * increment + val
                self.set_cam_rot_mat()
            # background color
            case 'bg':
                self.clear_color[0] = self.clear_color[0] * increment + val[0]
                self.clear_color[1] = self.clear_color[1] * increment + val[1]
                self.clear_color[2] = self.clear_color[2] * increment + val[2]
                self.clear_color[3] = self.clear_color[3] * increment + val[3]

        # set the projection matrix
        [left, right, bottom, top, near, far] = self.frustum
        # # make a matrix from as a symmetrical projection
        # self.aspect = (right - left) / (top - bottom)
        # self.fov = np.degrees(2 * np.arctan((top - bottom) / (2 * near)))
        # self.projection_mat = Mat4.perspective_projection(
        #     fov=self.fov,
        #     aspect=self.aspect,
        #     z_near=near,
        #     z_far=far)

        # make a matrix from the frustum, potentially asymmetrical
        a = (2 * near) / (right - left)
        b = (2 * near) / (top - bottom)
        c = (right + left) / (right - left)
        d = (top + bottom) / (top - bottom)
        e = -(far + near) / (far - near)
        f = -(2 * far * near) / (far - near)
        print(f'{a=}, {b=}, {c=}, {d=}, {e=}, {f=}, ')
        # Row-major order
        self.projection_mat =Mat4(
            a, 0, 0, 0,
            0, b, 0, 0,
            c, d, e, -1,
            0, 0, f, 0
        )
        print(f'{self.projection_mat=}')
        # print(f'{self.projection_mat2=}')

def str_to_key(s):
    """Change a string to an appropriate key tuple, one key, zero
    or more modifiers seperated by spaces, any order, upper or
    lower case mods: shift ctrl alt capslock numlock windows
    command option scrolllock function accel keys: A-Z, 0-9,
    BACKSPACE, TAB, LINEFEED, CLEAR, RETURN, ENTER, PAUSE,
    SCROLLLOCK SYSREQ, ESCAPE, SPACE, HOME, LEFT, UP, RIGHT, DOWN,
    PAGEUP, PAGEDOWN, END, BEGIN DELETE, SELECT, PRINT, EXECUTE,
    INSERT, UNDO, REDO, MENU, FIND, CANCEL, HELP BREAK, NUM_SPACE,
    NUM_TAB, NUM_ENTER, NUM_F1, NUM_F2, NUM_F3, NUM_F4, NUM_HOME
    NUM_LEFT, NUM_UP, NUM_RIGHT, NUM_DOWN, NUM_PRIOR, NUM_PAGE_UP,
    NUM_NEXT NUM_PAGE_DOWN, NUM_END, NUM_BEGIN, NUM_INSERT,
    NUM_DELETE, NUM_EQUAL NUM_MULTIPLY, NUM_ADD, NUM_SEPARATOR,
    NUM_SUBTRACT, NUM_DECIMAL, NUM_DIVIDE NUM_0, NUM_1, NUM_2,
    NUM_3, NUM_4, NUM_5, NUM_6, NUM_7, NUM_8, NUM_9, F1, F2, F3
    F4, F5, F6, F7, F8, F9, F10, F11, F12, F13, F14, F15, F16,
    F17, F18, F19, F20 EXCLAMATION, DOUBLEQUOTE, HASH, POUND,
    DOLLAR, PERCENT, AMPERSAND, APOSTROPHE PARENLEFT, PARENRIGHT,
    ASTERISK, PLUS, COMMA, MINUS, PERIOD, SLASH, COLON SEMICOLON,
    LESS, EQUAL, GREATER, QUESTION, AT, BRACKETLEFT, BACKSLASH
    BRACKETRIGHT, ASCIICIRCUM, UNDERSCORE, GRAVE, QUOTELEFT,
    BRACELEFT, BAR BRACERIGHT, ASCIITILDE

    """
    k, mods = 0, 0
    for ss in s.split(' '):
        ssu = ss.upper()  # constants in key are uppercase
        if 'MOD_' + ssu in key.__dict__:  # turn ctrl indo MOD_CTRL
            mods += key.__dict__['MOD_' + ssu]  # add to mods
        elif len(ssu) == 1 and ssu.isdigit():  # turn 5 into _5
            ssu = '_' + ssu
            k = key.__dict__['_' + ssu]
        elif ssu in key.__dict__:
            k = key.__dict__[ssu]
    return (k, mods)


def dict_key(key):
    """Change a keypress with modifiers to an appropriate tuple for a dictionary key"""
    if isinstance(key, str):  # is this a string
        key = str_to_key(key)  # parse it for key and mods
    elif not hasattr(key, '__iter__'):  # did we provide a (key modifier) tuple?
        key = (key, 0)  # if not, make a tuple with 0 modifier
    elif len(key) > 2:  # are there multiple modifiers?
        key = (key[0], sum(key[1:]))  # if so, add them together
    return tuple(key)  # in case we passed a list, change it to an immutable tuple


class Holocube_Window(pyglet.window.Window):
    """Subclass of pyglet's window, for displaying all the viewports
    with experimental stimuli

    """

    def __init__(self, ):
        # init the window class
        # config buffers double buffering and for antialiasing
        config = pyglet.gl.Config(sample_buffers=1, samples=4, double_buffer=True, depth_size=24)
        style = pyglet.window.Window.WINDOW_STYLE_BORDERLESS
        super().__init__(style=style, config=config)

        # self.disp = pyglet.canvas.Display()    #**** New ****
        # self.screens = self.disp.get_screens()
        # self.w, self.h = self.get_size()
        self.disp = pyglet.display.get_display()
        self.screens = self.display.get_screens()  # lists all available monitors explicitly

        self.moves = []
        self.w, self.h = self.get_size()

        self.moves = []
        self.key_actions = {}  # keys that execute a command, once, multiple times, or indefinitely
        self.frame_actions = []  # additional functions to execute each frame
        self.params = [[], [], [], []]
        self.activate()  # give keyboard focus

        # make the batches for display
        self.batch = pyglet.graphics.Batch()

        # where's the camera?
        self.pos = np.zeros(3)
        self.rot = np.identity(3)  # rotation matrix
        self.frame = 0

        self.add_keypress_action(key.I, self.print_info)
        self.bufm = pyglet.image.get_buffer_manager()

        # new hisory structure, to be declared at the beginning of an experiment
        self.hist_num_frames = 0
        self.hist_num_tests = 0
        self.hist_test_ind = 0
        self.record_fn_ind = 0  # current file number appended
        self.record_data = np.array([0])

        self.bg_color = [0., 0., 0., 1.]

        self.fps_display = pyglet.window.FPSDisplay(self)

        glEnable(GL_SCISSOR_TEST)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_PROGRAM_POINT_SIZE)

        # this is to allow updating all vpblock (view and projection
        # for each viewport) with just one call in windows.py for each
        # shader program in stimuli.py
        self.vpblock = GLuint()
        glGenBuffers(1, self.vpblock)
        # Bind the buffer to the VPBLOCK binding point 0
        glBindBuffer(GL_UNIFORM_BUFFER, self.vpblock)
        glBufferData(GL_UNIFORM_BUFFER, 2 * 16 * 4, None, GL_DYNAMIC_DRAW)  # 2 matrices (4x4) * 4 bytes per float
        glBindBufferBase(GL_UNIFORM_BUFFER, 0, self.vpblock)  # Bind to binding point 0
        glBindBuffer(GL_UNIFORM_BUFFER, 0)  # Unbind

        print('__init__')

    def __hash__(self):
        return hash(id(self))

    def start(self, config_file='viewport.config'):
        """Instantiate the window by loading the config file"""
        self.viewports = []
        self.load_config(config_file)
        if self.project:
            self.set_fullscreen(True, self.screens[self.screen_number])
        else:
            self.set_size(self.w, self.h)
        print(self.screen_number, self.project)
        self.curr_viewport_ind = 0
        self.curr_indicator_ind = 0
        print('start')

    def load_config(self, filename='viewport.config'):
        """Read the configuration file to size and color the window and all
        the viewports
        """
        config = configparser.ConfigParser()
        config.read(filename)
        # grab options for the whole screen
        bg_color = config.get('screen', 'bg_color', fallback='[0.,0.,0.,1.]')
        self.bg_color = ast.literal_eval(bg_color)
        self.project = config.getboolean('screen', 'project', fallback=False)
        self.screen_number = config.getint('screen', 'screen_number', fallback=0)
        self.w = config.getint('screen', 'w_size', fallback=640)
        self.h = config.getint('screen', 'h_size', fallback=320)
        # remove the screen section if it's there
        if 'screen' in config.sections():
            screen = config.pop('screen')
        # cycle through the remaining sections, viewports
        for vp_name in config.sections():
            batch = self.batch
            cfg = config[vp_name]
            name = vp_name
            coords = [cfg.getint('left', 0), cfg.getint('bottom', 0), cfg.getint('width', 100),
                      cfg.getint('height', 100)]
            scale_factors = [cfg.getfloat('scale_x', 1), cfg.getfloat('scale_y', 1), cfg.getfloat('scale_z', 1)]
            pan = cfg.getfloat('pan', 0.0)
            tilt = cfg.getfloat('tilt', 0.0)
            dutch = cfg.getfloat('dutch', 0.0)
            projection = cfg.get('projection', 'perspective')
            frustum = [cfg.getfloat('frustum_left', -.1), cfg.getfloat('frustum_right', .1),
                       cfg.getfloat('frustum_bottom', -.1), cfg.getfloat('frustum_top', .1),
                       cfg.getfloat('frustum_near', .1), cfg.getfloat('frustum_far', 1.)]
            self.add_viewport(batch, name, coords, scale_factors,
                              projection=projection, frustum=frustum,
                              pan=pan, tilt=tilt, dutch=dutch)
        print('load_config')

    def save_config(self, filename='viewport.config'):
        """Save the configuration file"""
        config = configparser.ConfigParser()
        config.add_section('screen')
        config['screen']['bg_color'] = str(self.bg_color)
        config['screen']['project'] = str(self.project)
        config['screen']['screen_number'] = str(self.screen_number)
        config['screen']['w_size'] = str(self.w)
        config['screen']['h_size'] = str(self.h)
        for viewport in self.viewports:
            config.add_section(viewport.name)
            config[viewport.name]['left'] = str(viewport.coords[0])
            config[viewport.name]['bottom'] = str(viewport.coords[1])
            config[viewport.name]['width'] = str(viewport.coords[2])
            config[viewport.name]['height'] = str(viewport.coords[3])
            config[viewport.name]['scale_x'] = str(viewport.scale_factors[0])
            config[viewport.name]['scale_y'] = str(viewport.scale_factors[1])
            config[viewport.name]['scale_z'] = str(viewport.scale_factors[2])
            config[viewport.name]['projection'] = str(viewport.projection)
            if viewport.projection.startswith('persp'):  # perspective projection
                config[viewport.name]['pan'] = str(viewport.pan)
                config[viewport.name]['tilt'] = str(viewport.tilt)
                config[viewport.name]['dutch'] = str(viewport.dutch)
                config[viewport.name]['frustum_left'] = str(viewport.frustum[0])
                config[viewport.name]['frustum_right'] = str(viewport.frustum[1])
                config[viewport.name]['frustum_bottom'] = str(viewport.frustum[2])
                config[viewport.name]['frustum_top'] = str(viewport.frustum[3])
                config[viewport.name]['frustum_near'] = str(viewport.frustum[4])
                config[viewport.name]['frustum_far'] = str(viewport.frustum[5])
            elif viewport.projection.startswith('ref'):  # reference projection
                config[viewport.name]['ref_pt_size'] = str(viewport.ref_pt_size)
                config[viewport.name]['ref_coords'] = ' '.join([str(rc) for rc in viewport.ref_coords])
        # save the file
        with open(filename, 'w') as config_file:
            config.write(config_file)
        print('wrote {}'.format(config_file))

    # def add_vertex_list(self, )

    def viewport_inc_ind(self, val=1, highlight=True):
        # get rid of omeome highlights
        if highlight: self.viewports[self.curr_viewport_ind].clear_color = [0.0, 0.0, 0.0, 1.0]
        # switch the index
        self.curr_viewport_ind = np.mod(self.curr_viewport_ind + val, len(self.viewports))
        # highlight the new viewport
        if highlight: self.viewports[self.curr_viewport_ind].clear_color = [0.2, 0.2, 0.2, 1.0]
        print('selected viewport: ', self.viewports[self.curr_viewport_ind].name)

    def ref_inc_ind(self, val=1, highlight=True):
        # only do anything if we have a current indicator viewport
        if self.viewports[self.curr_viewport_ind].name.startswith('ind'):
            if highlight:
                self.viewports[self.curr_viewport_ind].ref_coords[self.curr_indicator_ind * 3]
            self.curr_indicator_ind = np.mod(self.curr_indicator_ind + val,
                                             self.viewports[self.curr_viewport_ind].num_refs)

    def add_viewport(self, batch, name='vp-0', coords=[0, 0, 150, 150], scale_factors=[1., 1., 1],
                     projection='perspective', frustum=[-.1, .1, -.1, .1, .1, 2.0], ref_pt_size=5,
                     ref_coords=[16.0, 80.0, 0.0, 82.0, 82.0, 0.0, 14.0, 16.0, 0.0, 83.0, 16.0, 0.0],
                     pan=0.0, tilt=0.0, dutch=0.0):
        """Add a new viewport class to the window list of viewports to draw"""
        # must have a unique name
        while any([name == vp.name for vp in self.viewports]):
            if name[-1].isdecimal():
                bname = name[:-1]
                num = name[-1] + 1
            else:
                bname = name
                num = 0
            name = '{}{}'.format(bname, num)
        # add the new viewport to the list
        self.viewports.append(Viewport(batch, name, coords,
                                       scale_factors,
                                       projection=projection,
                                       frustum=frustum,
                                       pan=pan, tilt=tilt, dutch=dutch))
        self.curr_viewport_ind = len(self.viewports) - 1

    def viewport_set_val(self, field, val, increment='set', viewport_ind=None):
        """Set a value for all the viewports (but not refs)"""
        if viewport_ind is None or viewport_ind.startswith('curr'):  # set to the current vp
            viewports = [self.viewports[self.curr_viewport_ind]]
        elif isinstance(viewport_ind, int):  # single integer set to a vp
            viewport = [self.viewports[viewport_ind]]
        elif viewport_ind == 'all':  # set the value for all of them
            viewports = self.viewports
        elif viewport_ind == 'all-ref':  # set the value for all of them but ref windows
            viewports = [vp for vp in self.viewports if not vp.projection.startswith('ref')]
        elif viewport_ind == 'ref':  # set the value for only the ref windows
            viewports = [vp for vp in self.viewports if vp.projection.startswith('ref')]
        if increment.startswith('inc'):
            increment = True
        else:
            increment = False
        for viewport in viewports:
            viewport.set_val(field, val, increment)

    def set_near(self, near):
        self.viewport_set_val('near', near, 'set', 'all-ref')

    def set_far(self, far):
        self.viewport_set_val('far', far, 'set', 'all-ref')

    def set_bg(self, color=[0.0, 0.0, 0.0, 1.0]):
        self.viewport_set_val('bg', color, 'set', 'all-ref')

    def toggle_fullscreen(self):
        """Switch the fullscreen state on the current screen"""
        self.set_fullscreen(not self.fullscreen)

    def toggle_screen(self):
        """Switch the active current display screen"""
        self.screen_number = (self.screen_number + 1) % len(self.screens)
        self.set_fullscreen(self.fullscreen, self.screens[self.screen_number])

    def set_viewport_projection(self, viewport_ind=0, projection=0, half=False):
        """change the projection of this viewport:
        0 - orthographic
        1 - frustum (perspective)
        2 - orthographic for ref window, not the scene as a whole"""
        if projection == 0:
            self.viewports[viewport_ind].draw = self.viewports[viewport_ind].draw_ortho
            self.viewports[viewport_ind].sprite_pos = int((viewport_ind + 1) * 1000)
            self.viewports[viewport_ind].orth_coords = self.viewports[viewport_ind].coords[:].copy()
            if half:
                self.viewports[viewport_ind].orth_coords[2] /= 2
                self.viewports[viewport_ind].orth_coords[3] /= 2
        elif projection == 1:
            self.viewports[viewport_ind].draw = self.viewports[viewport_ind].draw_perspective

    def set_ref(self, ref_ind, color, viewport_ind=-1):
        """Set the color of a ref pt with a three tuple

        """
        # choose the proper viewport---usually there is only one
        indicator_vps = [vp for vp in self.viewports if vp.name.startswith('ind')]
        indicator_viewport = indicator_vps[viewport_ind]

        # if color is a number, make it a tuple
        if isinstance(color, numbers.Real):
            color = (0, int(color), 0)

        # self.viewports[viewport_ind].ref_vl.colors[(ref_ind*3):(ref_ind*3 + 3)] = color
        # set the color
        ri = ref_ind * 3
        indicator_viewport.ref_vl.colors[(ri):(ri + 3)] = color

    # TODO thisis the forest
    def unset_refs(self, viewport_ind=-1):
        """Set all ref indicators off

        """
        # choose the proper viewport---usually there is only one
        indicator_vps = [vp for vp in self.viewports if vp.name.startswith('ind')]
        indicator_viewport = indicator_vps[viewport_ind]

        # set each indicator to 0
        num_values = len(indicator_viewport.ref_vl.colors)
        indicator_viewport.ref_vl.colors = np.zeros(num_values, dtype=int)

    def move_ref(self, ref_ind, ax_ind, viewport_ind=0):
        """Set the pos of a ref pt with a three tuple

        """
        # choose the proper viewport---usually there is only one
        ref_vp = [vp for vp in self.viewports if vp.name.startswith('ind')][viewport_ind]
        # swap the color of the color vertexes
        # ref_vp.ref_colors[(ref_ind * 3):(ref_ind * 3 + 3)] = color

    # functions to set flash patterns for ref lights
    def set_flash_pattern(self, pattern, num_frames=0, ref_ind=0,
                          dist=10, col_1=255, col_2=96,
                          viewport_ind=0):
        """Sets the flash pattern for a reference flasher. the pattern is a
        binary string, or the whole array if flash_pattern is not
        None.

        """
        # choose the proper viewport---usually there is only one
        ref_vp = [vp for vp in self.viewports if vp.name.startswith('ind')][viewport_ind]
        if pattern is None:  # delete the entry
            ref_vp[ref_ind].pop(ref_ind)
        elif isinstance(pattern, np.ndarray):  # an array, just for playing
            ref_vp.flash_patterns[ref_ind] = pattern
        elif isinstance(pattern, str):
            if pattern.startswith('0b'):  # a binary number, just for playing
                flash_pattern = np.zeros(num_frames, dtype='ubyte')
                for i, p in enumerate(pattern):
                    flash_pattern[(i + 1) * dist] = col_1 if pattern[i] in '1b' else col_2
                    ref_vp.flash_patterns[ref_ind] = np.array(flash_pattern)

    def calc_flash_pattern(self, param, num_frames, ref_ind=0,
                           dist=10, col_1=255, col_2=96,
                           viewport_ind=0):
        """Calculate the flash pattern needed to signal current states of the
        window

        """
        if param == 'az':
            num = np.mod(np.arctan2(self.rot[2, 2], self.rot[2, 0]) * 180 / np.pi - 90, 360)
            self.set_flash_pattern(bin(int(num)), num_frames, ref_ind, dist, col_1, col_2, viewport_ind)
        if param == 'el':
            num = 360 - np.mod(
                np.arctan2(self.rot[2, 1], np.sqrt(self.rot[2, 2] ** 2 + self.rot[2, 0] ** 2)) * 180 / np.pi, 360)
            self.set_flash_pattern(bin(int(num)), num_frames, ref_ind, dist, col_1, col_2, viewport_ind)
        if param == 'start-stop':
            pat = np.zeros(num_frames, dtype='ubyte')
            pat[[1, -2]] == col_1
            self.set_flash_pattern(pat, num_frames, ref_ind, dist, col_1, col_2, viewport_ind)

    def flash_patterns(self, frame_ind, viewport_ind=0):
        """Set all the flash patterns in the reference viewport to the current
        frame

        """
        # choose the proper viewport---usually there is only one
        ref_vp = [vp for vp in self.viewports if vp.name.startswith('ind')][viewport_ind]
        for ref_ind in ref_vp.flash_patterns:
            color = ref_vp.flash_patterns[ref_ind][frame_ind]
            ref_vp.ref_vl.colors[(ref_ind * 3):(ref_ind * 3 + 3)] = (0, color, 0)

    ## alter position and heading of the viewpoint ##
    # alter position and heading directly
    def set_pos(self, pos):
        """Set the x, y, z of the viewpoint"""
        self.pos = pos

    def set_rot(self, rot):
        """Set the rotation matrix around the viewpoint"""
        self.rot = rot

    # alter position and heading relative to global axes
    def set_px(self, dis):
        """Set the x position (left and right) of the viewpoint"""
        self.pos[0] = dis

    def set_py(self, dis):
        """Set the y position (up and down) of the viewpoint"""
        self.pos[1] = dis

    def set_pz(self, dis):
        """Set the z position (forward and backward) of the viewpoint"""
        self.pos[2] = dis

    def inc_px(self, dis):
        """Increment the x position (left and right) of the viewpoint"""
        self.pos[0] += dis

    def inc_py(self, dis):
        """Increment the y position (up and down) of the viewpoint"""
        self.pos[1] += dis

    def inc_pz(self, dis):
        """Increment the z position (forward and backward) of the viewpoint"""
        self.pos[2] += dis

    def set_rx(self, ang=0):
        """Set the current heading around global the x axis. This erases
        other rotation, so may produce odd results if the viewer is already
        rotated"""
        sint, cost = np.sin(np.radians(ang)), np.cos(np.radians(ang))
        self.rot = np.array([[1, 0, 0], [0, cost, -sint], [0, sint, cost]])


    def set_ry(self, ang=0):
        """Set the current heading around global the y axis. This erases
        other rotation, so may produce odd results if the viewer is already
        rotated"""
        sint, cost = np.sin(np.radians(ang)), np.cos(np.radians(ang))
        self.rot = np.array([[cost, 0, sint], [0, 1, 0], [-sint, 0, cost]])

    def set_rz(self, ang=0):
        """Set the current heading around global the z axis. This erases
        other rotation, so may produce odd results if the viewer is already
        rotated"""
        sint, cost = np.sin(np.radians(ang)), np.cos(np.radians(ang))
        self.rot = np.array([[cost, -sint, 0], [sint, cost, 0], [0, 0, 1]])

    def inc_rx(self, ang=0):
        """Increment current heading around the global x axis"""
        sint, cost = np.sin(np.radians(ang)), np.cos(np.radians(ang))
        mat = np.array([[1, 0, 0], [0, cost, -sint], [0, sint, cost]])
        np.dot(mat, self.rot, out=self.rot)

    def inc_ry(self, ang=0):
        """Increment current heading around the global y axis"""
        sint, cost = np.sin(np.radians(ang)), np.cos(np.radians(ang))
        mat = np.array([[cost, 0, sint], [0, 1, 0], [-sint, 0, cost]])
        np.dot(mat, self.rot, out=self.rot)

    def inc_rz(self, ang=0):
        """Increment current heading around the global z axis"""
        sint, cost = np.sin(np.radians(ang)), np.cos(np.radians(ang))
        mat = np.array([[cost, -sint, 0], [sint, cost, 0], [0, 0, 1]])
        np.dot(mat, self.rot, out=self.rot)

    # alter position and heading relative to current position and heading
    def inc_slip(self, dis):
        """Move the viewpoint left and right, relative to heading"""
        self.pos += dis * np.dot(self.rot, np.array([1., 0., 0]))

    def inc_lift(self, dis):
        """Move the viewpoint up and down, relative to heading"""
        self.pos += dis * np.dot(self.rot, np.array([0., 1., 0.]))

    def inc_thrust(self, dis):
        """Move the viewpoint forward and backward, relative to heading"""
        self.pos += dis * np.dot(self.rot, np.array([0., 0., 1.]))

    def inc_pitch(self, ang=0):
        """Increment current heading in pitch"""
        sint, cost = np.sin(np.radians(ang)), np.cos(np.radians(ang))
        mat = np.array([[1, 0, 0], [0, cost, -sint], [0, sint, cost]])
        np.dot(self.rot, mat, out=self.rot)

    def inc_yaw(self, ang=0):
        """Increment current heading in yaw"""
        sint, cost = np.sin(np.radians(ang)), np.cos(np.radians(ang))
        mat = np.array([[cost, 0, sint], [0, 1, 0], [-sint, 0, cost]])
        np.dot(self.rot, mat, out=self.rot)

    def inc_roll(self, ang=0):
        """Increment current heading in roll"""
        sint, cost = np.sin(np.radians(ang)), np.cos(np.radians(ang))
        mat = np.array([[cost, -sint, 0], [sint, cost, 0], [0, 0, 1]])
        np.dot(self.rot, mat, out=self.rot)

    # reset to center position and straight heading
    def reset_pos(self):
        """Set pos back to 0,0,0"""
        self.pos = np.zeros((3))

    def reset_rot(self):
        """Set heading back to identity matrix"""
        self.rot = np.identity(3)

    def reset_pos_rot(self):
        """Set pos back to 0,0,0, heading back to identity matrix"""
        self.pos = np.zeros((3))
        self.rot = np.identity(3)

    def reset_frame(self, frame=0):
        self.frame = frame

    def save_cam(self, pos=True, rot=True, bg=True, near=True, far=True):
        """Save the curret camera position and rotation, bg color, near and far settings."""
        if pos: self.pos_saved = self.pos.copy()
        if rot: self.rot_saved = self.rot.copy()
        self.bg_saved, self.near_saved, self.far_saved = [], [], []
        for viewport in self.viewports:
            if bg: self.bg_saved.append(viewport.clear_color.copy())
            if near: self.near_saved.append(viewport.frustum[4])
            if far: self.far_saved.append(viewport.frustum[5])

    def restore_cam(self, pos=True, rot=True, bg=True, near=True, far=True):
        """Save the curret camera position and rotation, bg color, near and far settings."""
        if pos: self.pos = self.pos_saved.copy()
        if rot: self.rot = self.rot_saved.copy()
        for viewport in self.viewports:
            if bg: viewport.set_val('bg', self.bg_saved.pop(0))
            if near: viewport.set_val('near', self.near_saved.pop(0))
            if far: viewport.set_val('far', self.far_saved.pop(0))

    # query position and heading
    def get_pos(self):
        """Get the x, y, z of the viewpoint"""
        return self.pos

    def get_rot(self):
        """Get the rotation matrix around the viewpoint"""
        return self.rot

    # new --- added start to define number of tests and frames at the beginning
    # also now generic to record any data
    def start_record(self, num_tests, num_frames, data_dims):
        record_dims = [num_tests, num_frames]
        record_dims.extend(data_dims)
        self.record_data = np.zeros(record_dims)

    def record(self, test_ind, frame_ind, data):
        self.record_data[test_ind, frame_ind] = data

    def save_record(self, fn=None):
        if fn is None:
            fn = 'data/hc-{}-{:02d}.npy'.format(time.strftime('%Y-%m-%d'), self.record_fn_ind)
        np.save(fn, self.record_data)
        self.record_fn_ind += 1
        print('saved {} - {}'.format(fn, self.record_data.shape))

    def save_png(self, prefix='frame'):
        # set the viewport to grab the whole window
        # (trying to grab only one viewport leads to strange behavior---
        #  only the first one is ever imaged, regardless of how which coords are specified)
        glViewport(0, 0, self.w, self.h)
        self.bufm.get_color_buffer().save('{}_{:06d}.png'.format(prefix, self.frame))
        print('{}_{:06d}.png'.format(prefix, self.frame))
        # self.bufm.get_color_buffer().save(f'{prefix}_{self.frame:06d}.png')

    def add_keypress_action(self, key, action, *args, **kwargs):
        """Add a function to be executed once when a key is pressed"""
        self.add_key_action(1, key, action, *args, **kwargs)

    def add_keyhold_action(self, key, action, *args, **kwargs):
        """Add a function to be executed continuously while a key is pressed"""
        self.add_key_action(np.inf, key, action, *args, **kwargs)

    def add_key_action(self, num_frames, key, action, *args, **kwargs):
        """Add a function to be executed once when a key is pressed"""
        self.key_actions[dict_key(key)] = [num_frames, action, args, kwargs]

    def save_keys(self):
        """Save the current key bindings, to later reset them to this
        state.  This should let you assign keys without having to
        explicitly unbind them later.

        """
        self.key_actions_saved = self.key_actions.copy()

    def restore_keys(self):
        """Restore key bindings to the last saved state.

        """
        self.key_actions = self.key_actions_saved

    def remove_key_action(self, key):
        """Free up a key press combination"""
        key = dict_key(key)
        if key in self.key_actions:
            del (self.key_actions[key])

    def remove_key_actions(self, keys):
        """Free up a set of key press combination"""
        for key in keys:
            self.remove_key_action(key)

    def print_keypress_actions(self):
        items = sorted(self.key_actions.items())
        for keypress, action in items:
            keysymbol = key.symbol_string(keypress[0]).lstrip(' _')
            modifiers = key.modifiers_string(keypress[1]).replace('MOD_', '').replace('|', ' ').lstrip(' ')
            func, args, kwargs = action[0].__name__, action[1], action[2]
            print('{:<10} {:<6} --- {:<30}({}, {})'.format(modifiers, keysymbol, func, args, kwargs))

    def print_info(self):
        """print information about everything

        """
        print(f'pos:\n{self.pos}')
        print(f'rot:\n{self.rot}')
        fps_value = self.fps_display.label.text
        print(f'fps: {fps_value}')
        print(f'bg: {self.bg_color}')
        # print(f'fps = {}\n'.format(pyglet.clock.get_fps()))
        # for vp in self.viewports:
        #     print('viewport - {vp.name}')

    # def print_info(self):
    #     """print information about everything"""
    #     print( 'pos:\n{}'.format(self.pos))
    #     print('rot\n{}'.format(self.rot))
    #     print('fps = {}\n'.format(pyglet.clock.get_fps()))
    #     for vp in self.viewports:
    #         print('viewport - {}'.format(vp.name))

    ##############
    ### ON KEY ###
    ##############
    def on_key_press(self, symbol, modifiers):
        print(f'w {symbol=}, {modifiers=}')
        """Execute functions for a key press event"""
        # close the window (for when it has no visible close box)
        if symbol == key.PAUSE or symbol == key.BREAK:
            print('quitting now...')
            self.close()

        elif (symbol, modifiers) in self.key_actions:
            print(f'{self.key_actions[(symbol, modifiers)]=}')
            self.frame_actions.append(self.key_actions[(symbol, modifiers)])

        # if there were no hits, report which keys were pressed
        else:
            if symbol not in [65507, 65508, 65513, 65514, 65505,
                              65506]:  # if it's not a common modifier pressed on its own
                print('No action for {} {} ({} {})'.format(key.modifiers_string(modifiers), key.symbol_string(symbol),
                                                           modifiers, symbol))  # print whatever it was

    def on_key_release(self, symbol, modifiers):
        """When a key is released, remove its action from the frame_actions list, if it is there"""
        if (symbol, modifiers) in self.key_actions and self.key_actions[symbol, modifiers][0] == np.inf:
            self.frame_actions.remove(self.key_actions[(symbol, modifiers)])

    def update_vp(self, projection_matrix, view_matrix):
        """Updates the UBO with new projection & view matrices."""
        glBindBuffer(GL_UNIFORM_BUFFER, self.vpblock)
        data = np.hstack([projection_matrix, view_matrix]).astype(np.float32)
        glBufferSubData(GL_UNIFORM_BUFFER, 0, data.nbytes, data.ctypes.data)
        glBindBuffer(GL_UNIFORM_BUFFER, 0)  # Unbind

    ###############
    ### ON DRAW ###
    ###############
    def on_draw(self):
        """Each frame, clear the whole area, draw each viewport, and
        execute any held key commands

        """
        self.clear()

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glScissor(0, 0, self.w, self.h)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        for viewport in self.viewports:
            glViewport(*viewport.coords)
            self.projection = viewport.projection_mat
            self.view = viewport.view(self.pos, self.rot)
            self.update_vp(self.projection, self.view)

            # clear the viewport
            glScissor(*viewport.coords)
            glClearColor(*viewport.clear_color)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # draw the viewport
            self.batch.draw()

        # execute any keypress frame action commands
        for num_frames, fun, args, kwargs in self.frame_actions:
            fun(*args, **kwargs)

        # filter the frame_action list to eliminate finished actions
        self.frame += 1
        self.frame_actions = [[num_frames - 1, action, args, kwargs]
                              for num_frames, action, args, kwargs
                              in self.frame_actions if num_frames > 1]


if __name__ == '__main__':
    project = 0
    bg = [1., 1., 1., 1.]
    near, far = .01, 1.

    import holocore.hc as hc

    hc.window.start(project=project, bg_color=bg, near=near, far=far)

    hc.window.add_key_action(key._0, hc.window.abort, True)

    # run pyglet
    pyglet.app.run()
