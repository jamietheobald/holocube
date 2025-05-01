# import PySpin
import pyglet
from pyglet.gl import *
from pyglet.window import key

import numpy as np


def cm_soapbubble(x, y, alpha=255):
    """soapbubble colors for complex magnitudes and angles.

    x = intensity, lightness, 0 -- 1,

    y = radians

    :return type: array of ints

    """
    r = np.clip(-57.4 * np.sin(y + 4.5) + x * 265.1 + -20.3, 0, 255)
    g = np.clip(20.6 * np.sin(y + -1.5) + x * 253.5 + -14.3, 0, 255)
    b = np.clip(57.1 * np.sin(y + 3.2) + x * 234.6 + -4.3, 0, 255)
    if hasattr(r, '__len__'): alpha = np.resize(alpha, len(r))
    return np.array(np.stack([r, g, b, alpha]), dtype=int)

def soapcolors(x, num, alpha=255):
    """Generates a list of colors that vary continuously in hue,
    but are constant in lightness

    """
    return cm_soapbubble(x, np.linspace(0,np.pi*2, num, endpoint=False), alpha=alpha).T

def soapcolor(hue=(1,6), brightness=.5, alpha=255):
    """Generate a single color that is x of the way through a
    hue circle"""
    return cm_soapbubble(x=brightness, y=2*np.pi*hue[0]/(hue[1]+1), alpha=alpha)

def unpack_action(action):
    if callable(action):
        return action, [], {}

    if isinstance(action, (list, tuple)) and action:
        func = action[0]
        if not callable(func):
            raise ValueError("First element must be a callable")

        rest = action[1:]
        if rest and isinstance(rest[-1], dict):
            *args, kwargs = rest
        else:
            args = rest
            kwargs = {}

        return func, args, kwargs

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
            # ssu = '_' + ssu
            k = key.__dict__['_' + ssu]
        elif ssu in key.__dict__:
            k = key.__dict__[ssu]
    return (k, mods)


def dict_key(key):
    """Change a keypress with modifiers to an appropriate
    tuple for a dictionary key

    """
    if isinstance(key, str):  # is this a string
        key = str_to_key(key)  # parse it for key and mods
    elif not hasattr(key, '__iter__'):  # did we provide a (key modifier) tuple?
        key = (key, 0)  # if not, make a tuple with 0 modifier
    elif len(key) > 2:  # are there multiple modifiers?
        key = (key[0], sum(key[1:]))  # if so, add them together
    return tuple(key)  # in case we passed a list, change it to an immutable tuple


class Button:
    """Make a button for the window that can return whether a click
    occurred inside it or not

    """
    def __init__(self, window, name, action, number=None, bounds=None,
                 color=None, color_num=None, shortcut=None, hold=None):
        self.window = window
        self.name = name
        self.bounds = bounds

        if hold is True or hold=='hold':
            num_frames = np.inf
        elif isinstance(hold, int):
            num_frames = hold
        else:
            num_frames = 1

        func, args, kwargs = unpack_action(action)

        self.button_action = [num_frames, func, args, kwargs]

        self.window.switch_to()

        if (bounds is None) and (number is None):
            number = (20, 3, 1, 1)

        if number is not None:
            button_width = self.window.width//number[1]
            button_height = self.window.height//number[0]
            pad = self.window.button_pad
            left = button_width * (number[3]-1) + pad
            width = button_width - 2*pad
            bottom = self.window.height - button_height * number[2] + pad
            height = button_height - 2*pad

            self.bounds = left, bottom, width, height

        if color is None:
            if color_num is None:
                color = [128, 128, 128, 255]
            else:
                color = soapcolor(color_num)

        self.rect = pyglet.shapes.Rectangle(*self.bounds, color,
                                         batch=self.window.batch)

        text = self.name
        x = left + width//2
        y = bottom + height//2
        fs = 16
        if shortcut is not None:
            text += f' - {shortcut}'
        self.label = pyglet.text.Label(text, x,y,
                                       anchor_x='center', anchor_y='center',
                                       font_size=fs,
                                       batch=self.window.batch)

    def __repr__(self):
        return f'Button - {self.name}'

    def is_in(self, x, y):
        """If x,y is in the boundaries of the button, returns True

        """
        left, bottom, width, height = self.bounds
        x0, x1, y0, y1 = left, left+width, bottom, bottom+height
        return (x0 <= x <= x1) and (y0 <= y <= y1)


class Control_Window(pyglet.window.Window):
    """Subclass of pyglet's window, for displaying experimental
    controls

    """
    def __init__(self):
        self.frame = 0
        config = pyglet.gl.Config(sample_buffers=1, samples=4, double_buffer=True, depth_size=24)
        style = pyglet.window.Window.WINDOW_STYLE_DEFAULT
        super().__init__(style=style, config=config)

        self.disp = pyglet.display.get_display()
        self.screens = self.display.get_screens()  # lists all available monitors explicitly

        self.key_actions = {}  # keys that execute a command, once, multiple times, or indefinitely
        self.frame_actions = []  # additional functions to execute each frame

    #     # make the batches for display
        self.batch = pyglet.graphics.Batch()
        self.activate()  # give keyboard focus

        self.button_list = []

        # items on the experiment menu, shortcuts are number keys
        # each item [name, vertex_list]
        self.mleft = 50
        self.mwidth = 400
        self.mheight = 40
        self.button_pad = 5



    def start(self):
        """start the control window by telling it which scheduler connects to it,
         so we can deliver commands

        """
        # self.scheduler = scheduler


    def add_button(self, name, action, number=None, bounds=None,
                   color=None, color_num=None, shortcut=None):
        """Add a button to the screen and menu check.

        """
        button = Button(self, name, action,
                        number, bounds, color, color_num,
                        shortcut)

        if shortcut:
            self.add_key_action(shortcut, action)
        else:
            sc = ''

        self.button_list.append(button)


    def on_mouse_press(self, x, y, button_code, modifiers, **kwargs):
        """Handle the mouse press by comparing with button_list
        :param

        """
        print(f'mouse_press, ({x=}, {y=}, {button_code=}, {modifiers=})')
        for button in self.button_list:
            if button.is_in(x, y):
                print(button.name)
                self.frame_actions.append(button.button_action)


    def add_key_action(self, key, action, hold=None):
        """Add a function with arguments to a key or key
        combination

        """
        if hold is True or hold=='hold':
            num_frames = np.inf
        elif isinstance(hold, int):
            num_frames = hold
        else:
            num_frames = 1

        func, args, kwargs = unpack_action(action)

        self.key_actions[dict_key(key)] = [num_frames, func, args, kwargs]


    def save_keys(self):
        '''Save the current key bindings, to later reset them to this
        state.  This should let you assign keys without having to
        explicitly unbind them later.

        '''
        self.key_actions_saved = self.key_actions.copy()


    def restore_keys(self):
        '''Restore key bindings to the last saved state.

        '''
        self.key_actions = self.key_actions_saved

    def remove_key_action(self, key):
        '''Free up a key press combination'''
        key = dict_key(key)
        if key in self.key_actions:
            del (self.key_actions[key])

    def remove_key_actions(self, keys):
        '''Free up a set of key press combination'''
        for key in keys:
            self.remove_key_action(key)

    def print_keypress_actions(self):
        items = sorted(self.key_actions.items())
        for keypress, action in items:
            keysymbol = key.symbol_string(keypress[0]).lstrip(' _')
            modifiers = key.modifiers_string(keypress[1]).replace('MOD_', '').replace('|', ' ').lstrip(' ')
            func, args, kwargs = action[0].__name__, action[1], action[2]
            print('{:<10} {:<6} --- {:<30}({}, {})'.format(modifiers, keysymbol, func, args, kwargs))

    ##############
    ### ON KEY ###
    ##############
    def on_key_press(self, symbol, modifiers):
        '''Execute functions for a key press event'''
        # close the window (for when it has no visible close box)
        print(f'c {symbol=}, {modifiers=}')
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
        '''When a key is released, remove its action from the frame_actions list, if it is there'''
        if (symbol, modifiers) in self.key_actions and self.key_actions[symbol, modifiers][0] == np.inf:
            self.frame_actions.remove(self.key_actions[(symbol, modifiers)])


    def on_draw(self):
        self.clear()
        self.batch.draw()

        # execute any keypress frame action commands
        for num_frames, fun, args, kwargs in self.frame_actions:
            fun(*args, **kwargs)

        # filter the frame_action list to eliminate finished actions
        self.frame += 1
        self.frame_actions = [[num_frames - 1, action, args, kwargs]
                              for num_frames, action, args, kwargs
                              in self.frame_actions if num_frames > 1]



