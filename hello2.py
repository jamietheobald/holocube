import pyglet
from pyglet.gl import *
from pyglet.window import key
from os.path import expanduser
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.math import Mat3, Mat4, Vec3

import numpy as np
import time
import configparser
import ast
import numbers

# GLSL source as python strings to make the shader program
vertex_source = """#version 460 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 color;

out vec3 frag_color;

uniform mat4 projection; 
uniform mat4 view;

void main() {
    vec4 world_position = view * vec4(position, 1.0);  // Ensure `view` is used
    gl_Position = projection * world_position;  // Forces OpenGL to keep `view`
    frag_color = color;
}
"""
#     gl_Position = vec4(position, 1.0);
#     frag_color = color;  // Pass vertex color to fragment shaderif not shader_program.success:
# }

fragment_source = """#version 460 core

in vec3 frag_color;  // Receive color from vertex shader
out vec4 FragColor;

void main() {
    FragColor = vec4(frag_color, 1.0);  // Output interpolated color
}
"""


class Viewport_class():
    '''A class for a single window configuration information.

    '''

    def __init__(self, batch, name='vp', coords=[0, 0, 150, 150],
                 scale_factors=[1., 1., 1],
                 for_ax=[2, -1], up_ax=[1, 1], projection='perspective',
                 frustum=[-.1, .1, -.1, .1, .1, 2.0],
                 pan=0.0, tilt=0.0, dutch=0.0):
        self.batch = batch
        self.name = name
        self.coords = np.array(coords, dtype='int')  # window left, bottom, width, height
        self.orth_coords = self.coords
        self.scale_factors = scale_factors  # x y and z flips for mirror reflections
        self.pan = pan
        self.tilt = tilt
        self.dutch = dutch
        # calculate the intrinsic camera rotation matrix
        self.set_cam_rot_mat()
        # self.rmat = dot(rotmat([0.,1.,0.], azimuth), rotmat([1.,0.,0.], elevation))
        self.projection = projection  # orthographic or perspective projection, or ref
        # self.rotmat = dot(rotmat([0., 1., 0.], pan*pi/180), rotmat([1., 0., 0.], tilt*pi/180))
        self.forward_up = np.array([[0, 0], [0, 1], [-1, 0.]])
        self.ref_vl = None
        # if projection.startswith('persp') :
        #     self.project = 'persp'
        #     self.draw = self.draw_perspective
        left, right, bottom, top, near, far = frustum
        self.frustum = np.array([left, right, bottom, top, near, far])  # left right bottom top near far
        # set the projection matrix
        self.aspect = (right - left) / (top - bottom)
        self.fov = np.degrees(2 * np.arctan((top - bottom) / (2 * near)))
        self.projection = Mat4.perspective_projection(fov=self.fov, aspect=self.aspect, z_near=near, z_far=far)

        self.sprite_pos = 0
        self.clear_color = [0., 0., 0., 1.]

    def set_cam_rot_mat(self):
        '''Set the rotation matrix for the viewport from the pan,
        tilt, and dutch angle of the virtual camera, set as intrinsic
        fields before calling this function. Right now this is 4x
        faster than using rotate methods on a Mat4s.

        '''
        sint, cost = np.sin(self.pan * np.pi / 180), np.cos(self.pan * np.pi / 180)
        pan_mat = np.array([[cost, 0, sint], [0, 1, 0], [-sint, 0, cost]])  # yaw of the camera
        sint, cost = np.sin(self.tilt * np.pi / 180), np.cos(self.tilt * np.pi / 180)
        tilt_mat = np.array([[1, 0, 0], [0, cost, -sint], [0, sint, cost]])  # pitch of the camera
        sint, cost = np.sin(self.dutch * np.pi / 180), np.cos(self.dutch * np.pi / 180)
        dutch_mat = np.array([[cost, -sint, 0], [sint, cost, 0], [0, 0, 1]])  # dutch (roll) of the camera
        self.cam_rot_mat = np.dot(np.dot(dutch_mat, tilt_mat), pan_mat)

    def view(self, pos, rot):
        '''calculate the view for a given 

        '''
        # soon I can figure out how to change this to pyglet mats and
        # vecs
        view = np.dot(np.dot(rot, self.cam_rot_mat), self.forward_up)

        look_at = Mat4.look_at(
            position=Vec3(*pos),
            target=Vec3(pos[0] + view[0, 0],
                        pos[1] + view[1, 0],
                        pos[2] + view[2, 0]),
            up=Vec3(view[0, 1],
                    view[1, 1],
                    view[2, 1]))

        return look_at

    def set_val(self, field, val, increment=False):
        '''Set any of the viewport positional and rotational parameters by
        name'''
        if field == 'left':
            self.coords[0] = self.coords[0] * increment + val
        elif field == 'bottom':
            self.coords[1] = self.coords[1] * increment + val
        elif field == 'width':
            self.coords[2] = self.coords[2] * increment + val
        elif field == 'height':
            self.coords[3] = self.coords[3] * increment + val
        elif field == 'near':
            self.frustum[4] = self.frustum[4] * increment + val
        elif field == 'far':
            self.frustum[5] = self.frustum[5] * increment + val
        elif field == 'scalex':
            self.scale_factors[0] *= -1
        elif field == 'scaley':
            self.scale_factors[1] *= -1
        elif field == 'scalez':
            self.scale_factors[2] *= -1
        elif field == 'pan':
            self.pan = self.pan * increment + val
            self.set_cam_rot_mat()
        elif field == 'tilt':
            self.tilt = self.tilt * increment + val
            self.set_cam_rot_mat()
        elif field == 'dutch':
            self.dutch = self.dutch * increment + val
            self.set_cam_rot_mat()
        elif field == 'bg':
            self.clear_color[0] = self.clear_color[0] * increment + val[0]
            self.clear_color[1] = self.clear_color[1] * increment + val[1]
            self.clear_color[2] = self.clear_color[2] * increment + val[2]
            self.clear_color[3] = self.clear_color[3] * increment + val[3]


class Holocube_window(pyglet.window.Window):
    '''Subclass of pyglet's window, for displaying all the viewports'''

    def __init__(self):
        super().__init__()
        self.label = pyglet.text.Label('screen 1',
                                       font_size=36,
                                       x=self.width // 2,
                                       y=self.height // 2,
                                       anchor_x='center',
                                       anchor_y='center')

        self.disp = pyglet.canvas.get_display()  # **** New ****
        self.screens = self.disp.get_screens()
        self.w, self.h = self.get_size()

        self.moves = []
        self.key_actions = {}  # keys that execute a command, once, multiple times, or indefinitely
        self.frame_actions = []  # additional functions to execute each frame
        self.params = [[], [], [], []]
        self.activate()  # give keyboard focus

        # make the batches for display
        self.batch = pyglet.graphics.Batch()

        # where's the camera?
        self.pos = np.zeros((3))
        self.rot = np.identity(3)  # rotation matrix
        self.frame = 0

        self.add_keypress_action(key.I, self.print_info)
        self.add_keyhold_action(key.LEFT, self.inc_yaw, .05)
        self.add_keyhold_action(key.RIGHT, self.inc_yaw, -.05)
        self.add_keyhold_action(key.UP, self.inc_thrust, -.05)
        self.add_keyhold_action(key.DOWN, self.inc_thrust, .05)

        self.bufm = pyglet.image.get_buffer_manager()

        # new hisory structure, to be declared at the beginning of an experiment
        self.hist_num_frames = 0
        self.hist_num_tests = 0
        self.hist_test_ind = 0
        self.record_fn_ind = 0  # current file number appended
        self.record_data = np.array([0])

        self.bg_color = [0., 1., 0., 1.]

        self.fps_display = pyglet.window.FPSDisplay(self)
        print('__init__')

        glEnable(GL_DEPTH_TEST)  # Enables Depth Testing
        glEnable(GL_SCISSOR_TEST)

        self.vert_shader = Shader(vertex_source, 'vertex')
        self.frag_shader = Shader(fragment_source, 'fragment')
        self.shader_program = ShaderProgram(self.vert_shader, self.frag_shader)

        ###### test pts
        # Generate 100 random points around (0,0,0)
        num_pts = 10000
        positions = np.random.uniform(-1, 1, (num_pts, 3)).astype('f4')  # Shape: (100, 3)
        colors = np.random.uniform(0, 1, (num_pts, 3)).astype('f4')  # RGBA colors
        vertex_format = "3f position 4f color"
        self.vlist = self.shader_program.vertex_list(
            num_pts, GL_POINTS,
            batch=self.batch,
            position=('f', positions.flatten()),
            color=('f', colors.flatten())
        )

    def start(self, config_file='test_viewport.config'):
        '''Instantiate the window by loading the config file

        '''
        self.viewports = []
        self.load_config(config_file)
        if self.project:
            self.set_fullscreen(True, self.screens[self.screen_number])
        else:
            self.set_size(self.w, self.h)
        print(f'{self.screen_number=}, {self.project=}')
        self.curr_viewport_ind = 0
        self.curr_indicator_ind = 0
        print('start')

    def load_config(self, filename='viewport.config'):
        '''Read the configuration file to size and color the window and all
        the viewports
        '''
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
        if 'screen' in config.sections(): screen = config.pop('screen')
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

    def add_viewport(self, batch, name='vp-0', coords=[0, 0, 150, 150], scale_factors=[1., 1., 1],
                     for_ax=[2, -1], up_ax=[1, 1], projection='perspective',
                     frustum=[-.1, .1, -.1, .1, .1, 2.0], ref_pt_size=5,
                     pan=0.0, tilt=0.0, dutch=0.0):
        '''Add a new viewport class to the window list of viewports to
        draw

        '''
        # must have a unique name
        while any([name == vp.name for vp in self.viewports]):
            if name[-1].isdecimal():
                bname = name[:-1]
                num = name[-1] + 1
            else:
                bname = name
                num = 0
            name = f'{bname}{num}'
        # add the new viewport to the list
        self.viewports.append(Viewport_class(batch, name, coords,
                                             scale_factors, for_ax, up_ax,
                                             projection=projection,
                                             frustum=frustum,
                                             pan=pan, tilt=tilt, dutch=dutch))
        self.curr_viewport_ind = len(self.viewports) - 1

    # alter position and heading relative to current position and heading
    def inc_slip(self, dis):
        '''Move the viewpoint left and right, relative to heading'''
        self.pos += dis * np.dot(self.rot, np.array([1., 0., 0]))

    def inc_lift(self, dis):
        '''Move the viewpoint up and down, relative to heading'''
        self.pos += dis * np.dot(self.rot, np.array([0., 1., 0.]))

    def inc_thrust(self, dis):
        '''Move the viewpoint forward and backward, relative to heading'''
        self.pos += dis * np.dot(self.rot, np.array([0., 0., 1.]))

    def inc_pitch(self, ang=0):
        '''Increment current heading in pitch'''
        sint, cost = np.sin(ang), np.cos(ang)
        mat = np.array([[1, 0, 0], [0, cost, -sint], [0, sint, cost]])
        np.dot(self.rot, mat, out=self.rot)

    def inc_yaw(self, ang=0):
        '''Increment current heading in yaw'''
        sint, cost = np.sin(ang), np.cos(ang)
        mat = np.array([[cost, 0, sint], [0, 1, 0], [-sint, 0, cost]])
        np.dot(self.rot, mat, out=self.rot)

    def inc_roll(self, ang=0):
        '''Increment current heading in roll'''
        sint, cost = np.sin(ang), np.cos(ang)
        mat = np.array([[cost, -sint, 0], [sint, cost, 0], [0, 0, 1]])
        np.dot(self.rot, mat, out=self.rot)

    # reset to center position and straight heading
    def reset_pos(self):
        '''Set pos back to 0,0,0'''
        self.pos = np.zeros((3))

    def reset_rot(self):
        '''Set heading back to identity matrix'''
        self.rot = np.identity(3)

    def reset_pos_rot(self):
        '''Set pos back to 0,0,0, heading back to identity matrix'''
        self.pos = np.zeros((3))
        self.rot = np.identity(3)

    def str_to_key(self, s):
        '''Change a string to an appropriate key tuple, one key, zero
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

        '''
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

    def dict_key(self, key):
        '''Change a keypress with modifiers to an appropriate tuple
        for a dictionary key

        '''
        if isinstance(key, str):  # is this a string
            key = self.str_to_key(key)  # parse it for key and mods
        elif not hasattr(key, '__iter__'):  # did we provide a (key modifier) tuple?
            key = (key, 0)  # if not, make a tuple with 0 modifier
        elif len(key) > 2:  # are there multiple modifiers?
            key = (key[0], sum(key[1:]))  # if so, add them together
        return tuple(key)  # in case we passed a list, change it to an immutable tuple

    def add_keypress_action(self, key, action, *args, **kwargs):
        '''Add a function to be executed once when a key is pressed

        '''
        self.add_key_action(1, key, action, *args, **kwargs)

    def add_keyhold_action(self, key, action, *args, **kwargs):
        '''Add a function to be executed continuously while a key is pressed

        '''
        self.add_key_action(np.inf, key, action, *args, **kwargs)

    def add_key_action(self, num_frames, key, action, *args, **kwargs):
        '''Add a function to be executed once when a key is pressed

        '''
        self.key_actions[self.dict_key(key)] = [num_frames, action, args, kwargs]

    def save_keys(self):
        '''Save the current key bindings, to later reset them to this
        state.  This should let you assign keys without having to
        explicitly unbind them later

        '''
        self.key_actions_saved = self.key_actions.copy()

    def restore_keys(self):
        '''Restore key bindings to the last saved state

        '''
        self.key_actions = self.keyhold_actions_saved

    def remove_key_action(self, key):
        '''Free up a key press combination

        '''
        key = self.dict_key(key)
        if key in self.key_actions:
            del (self.key_actions[key])

    def remove_key_actions(self, keys):
        '''Free up a set of key press combination

        '''
        for key in keys:
            self.remove_key_action(key)

    def print_keypress_actions(self):
        '''Print all the assigned keypresses, and the functions they
        execute

        '''
        items = sorted(self.key_actions.items())
        for keypress, action in items:
            keysymbol = key.symbol_string(keypress[0]).lstrip(' _')
            modifiers = key.modifiers_string(keypress[1]).replace('MOD_', '').replace('|', ' ').lstrip(' ')
            func, args, kwargs = action[0].__name__, action[1], action[2]
            print('{:<10} {:<6} --- {:<30}({}, {})'.format(modifiers, keysymbol, func, args, kwargs))

    def print_info(self):
        '''print information about everything

        '''
        print(f'pos:\n{self.pos}')
        print(f'rot:\n{self.rot}')
        fps_value = self.fps_display.label.text
        print(f'fps: {fps_value}')
        # print(f'fps = {}\n'.format(pyglet.clock.get_fps()))
        # for vp in self.viewports:
        #     print('viewport - {vp.name}')

    ##############
    ### ON KEY ###
    ##############
    def on_key_press(self, symbol, modifiers):
        '''Execute functions for a key press event.
        
        Closing the window is hard coded to pause or break.  Other
        commands are in the key_actions dictionary.  Otherwise, if its
        not a modifier key, report that it's not in the list.

        '''
        # close the window (for when it has no visible close box)
        if symbol == key.PAUSE or symbol == key.BREAK:
            print('quitting now...')
            self.close()

        # if the key combination is in the list, append its action to
        # the frame action list
        elif (symbol, modifiers) in self.key_actions:
            self.frame_actions.append(self.key_actions[(symbol, modifiers)])

        # if there were no hits, report which keys were pressed
        else:
            # if it's not a common modifier pressed on its own
            if symbol not in [65507, 65508, 65513, 65514, 65505, 65506]:
                print(
                    f'No action for {key.modifiers_string(modifiers)} {key.symbol_string(symbol)} ({modifiers} {symbol})')

    def on_key_release(self, symbol, modifiers):
        '''When a key is released, remove its action from the
        frame_actions list, if it is there.

        '''
        if (symbol, modifiers) in self.key_actions and self.key_actions[symbol, modifiers][0] == np.inf:
            self.frame_actions.remove(self.key_actions[(symbol, modifiers)])

    ###############
    ### ON DRAW ###
    ###############
    def on_draw(self):
        '''Each frame, clear the whole area, draw each viewport, and
        execute any held key commands

        '''
        # first clear the whole screen with black
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glScissor(0, 0, self.w, self.h)

        # then set the bg_color to clear each viewport
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(*self.bg_color)

        self.clear()
        # self.label.draw()

        with self.shader_program:
            for viewport in self.viewports:
                # coordinates in the window
                glViewport(*viewport.coords)

                # set the shader's projection matrix and camera view
                self.shader_program["projection"] = viewport.projection
                self.shader_program["view"] = viewport.view(self.pos, self.rot)

                # draw the viewport
                self.batch.draw()

        # execute any keypress frame action commands
        for num_frames, fun, args, kwargs in self.frame_actions:
            fun(*args, **kwargs)

        # filter the frame_action list to eliminate finished actions
        self.frame += 1
        self.frame_actions = [[num_frames - 1, action, args, kwargs] for num_frames, action, args, kwargs in
                              self.frame_actions if num_frames > 1]


w = Holocube_window()
w.start()
v = w.viewports[0]

pyglet.app.run()

cr1 = np.array(v.cam_rot_mat).reshape(4, 4)[:3, :3]
view = np.dot(np.dot(w.rot, cr1), v.forward_up)
