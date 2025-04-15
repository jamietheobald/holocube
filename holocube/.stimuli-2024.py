# classes for various flexible visual stimuli
# lightspeed design 360 projector displays in the order b r g frames
# pip install numpy-stl

import numpy as np
from stl import mesh

import pyglet
from pyglet.gl import *

import scipy.stats


# for rotating the camera view angles
def rotmat(u=[0., 0., 1.], theta=0.0):
    '''Returns a matrix for rotating an arbitrary amount (theta)
    around an arbitrary axis (u, a unit vector).

    '''
    ux, uy, uz = u
    cost, sint = np.cos(theta), np.sin(theta)

    uxu = np.array([[ux * ux, ux * uy, ux * uz],
                    [ux * uy, uy * uy, uz * uy],
                    [ux * uz, uy * uz, uz * uz]])

    ux = np.array([[0, -uz, uy],
                   [uz, 0, -ux],
                   [-uy, ux, 0]])

    return cost * np.identity(3) + sint * ux + (1 - cost) * uxu


class Sprite(pyglet.graphics.Group):
    def __init__(self, window, vp=0, half=False):
        self.window = window
        # self.pos = np.array([vp*1000,0.]) #changed to match change projection in windows.py
        self.pos = np.array([(vp + 1) * 1000, 0.])
        # self.pos = np.array([0,0.])
        self.rot = np.array([0.])
        self.visible = False
        self.wd, self.ht = self.window.viewports[vp].coords[2:]
        if half:
            self.wd = int(self.wd / 2)
            self.ht = int(self.ht / 2)

    def add(self, x=0, y=0):
        self.sprite = pyglet.sprite.Sprite(self.image, x=self.pos[0], y=self.pos[1], batch=self.window.world)
        self.visible = True

    def remove(self):
        self.sprite.delete()
        self.visible = False

    def on(self, onstate):
        if onstate and not self.visible:
            self.add()
        elif not onstate and self.visible:
            self.remove()

    def switch(self, onoff):
        '''A synonym for method on, adding string control'''
        if onoff == 'on' or onoff == True or onoff == 1:
            onstate = True
        elif onoff == 'off' or onoff == False or onoff == 0:
            onstate = False
        else:
            onstate = False

        if onstate and not self.visible:
            self.add()
        elif not onstate and self.visible:
            self.remove()

    def change_vp(self, vp_ind=0):
        '''change the vp_ind by x shifting the entire sprite by 1000*vp_ind'''
        self.sprite.x = int(1000 * vp_ind)

    def animate(self, frames):
        # self.sprite.draw
        pass


class Movable(pyglet.graphics.Group):
    '''Any opengl object that we can move and rotate outside of observer
    motion

    '''

    def __init__(self, window):
        super().__init__()
        self.window = window
        self.parent = None  # required to use set_state
        self.pos = np.array([0, 0, 0.])
        self.rot = np.array([0, 0, 0.])
        self.ind = 0
        self.poss = np.array([[0, 0, 0.]])
        self.visible = False
        self.txtcoords = None
        self.colors = None

    def add(self):
        # are colors specified?
        if self.colors is None: self.colors = np.zeros((3, self.num), dtype='byte') + 255
        if self.txtcoords is None: self.txtcoords = np.zeros((2, self.num), dtype='float')
        self.vl = self.window.world.add(self.num, self.gl_type, self,
                                        ('v3f', self.coords.T.flatten()),
                                        ('c3B/stream', self.colors.T.flatten()),
                                        ('t2f', self.txtcoords.T.flatten()))
        self.visible = True

    def remove(self):
        self.vl.delete()
        self.visible = False

    def on(self, onstate):
        if onstate and not self.visible:
            self.add()
        elif not onstate and self.visible:
            self.remove()

    def switch(self, onoff):
        '''A synonym for method on, adding string control'''
        if onoff == 'on' or onoff == True or onoff == 1:
            onstate = True
        elif onoff == 'off' or onoff == False or onoff == 0:
            onstate = False
        else:
            onstate = False

        if onstate and not self.visible:
            self.add()
        elif not onstate and self.visible:
            self.remove()

    def set_color(self, color=np.array([1, 1, 1])):
        self.vl.colors = take(atleast_1d(np.array(color, dtype='int8')), mod(np.arange(self.num * 3), 3), mode='wrap')

    def set_colorf(self, color=0.0):
        self.vl.colors = np.array(np.repeat(color * 255, self.num * 3), dtype='byte')

    def set_pos(self, pos):
        self.pos[:] = pos

    def set_rot(self, rot):
        self.rot[:] = rot

    def set_px(self, x):
        self.pos[0] = x

    def set_py(self, y):
        self.pos[1] = y

    def set_pz(self, z):
        self.pos[2] = z

    def set_rx(self, x):
        self.rot[0] = x

    def set_ry(self, y):
        self.rot[1] = y

    def set_rz(self, z):
        self.rot[2] = z

    def inc_px(self, x=.01):
        self.pos[0] += x

    def inc_py(self, y=.01):
        self.pos[1] += y

    def inc_pz(self, z=.01):
        self.pos[2] += z

    def inc_rx(self, x=np.pi / 180):
        self.rot[0] += x

    def inc_ry(self, y=np.pi / 180):
        self.rot[1] += y

    def inc_rz(self, z=np.pi / 180):
        self.rot[2] += z

    def update_pos_ind(self, dt=0.0):
        take(self.poss, [self.ind], out=self.pos, mode='wrap')
        self.ind += 1

    def update_ry_func(self, dt=0.0, func=None):
        self.rot[1] = func()

    def subset_inc_px(self, bool_a, x):
        '''move in x direction just some of the vertices'''
        self.coords[0, bool_a] += x
        if self.visible:
            self.vl.vertices[::3] = self.coords[0]

    def subset_inc_py(self, bool_a, y):
        '''move in z direction just some of the vertices'''
        self.coords[1, bool_a] += y
        if self.visible:
            self.vl.vertices[1::3] = self.coords[1]

    def subset_inc_pz(self, bool_a, z):
        '''move in z direction just some of the vertices'''
        self.coords[2, bool_a] += z
        if self.visible:
            self.vl.vertices[2::3] = self.coords[2]

    def subset_set_px(self, bool_a, x_a):
        '''set x coordinate for some vertices '''
        self.coords[0, bool_a] = x_a[bool_a]
        if self.visible:
            self.vl.vertices[0::3] = self.coords[0]

    def subset_set_py(self, bool_a, y_a):
        '''set y coordinate for some vertices '''
        self.coords[1, bool_a] = y_a[bool_a]
        if self.visible:
            self.vl.vertices[1::3] = self.coords[1]

    def subset_set_pz(self, bool_a, z_a):
        '''set z coordinate for some vertices '''
        self.coords[2, bool_a] = y_a[bool_a]
        if self.visible:
            self.vl.vertices[2::3] = self.coords[2]

    def set_coords(self, coords):
        self.coords = coords
        if self.visible:
            self.vl.vertices[:] = self.coords.T.ravel()

    def get_rx(self):
        return self.rot[0]

    def get_ry(self):
        return self.rot[1]

    def get_rz(self):
        return self.rot[2]

    def get_px(self):
        return self.pos[0]

    def get_py(self):
        return self.pos[1]

    def get_pz(self):
        return self.pos[2]

    def set_state(self):
        glRotatef(self.rot[0], 1.0, 0.0, 0.0)
        glRotatef(self.rot[1], 0.0, 1.0, 0.0)
        glRotatef(self.rot[2], 0.0, 0.0, 1.0)
        glTranslatef(*self.pos)

    def unset_state(self):
        glTranslatef(*-self.pos)
        glRotatef(-self.rot[2], 0.0, 0.0, 1.0)
        glRotatef(-self.rot[1], 0.0, 1.0, 0.0)
        glRotatef(-self.rot[0], 1.0, 0.0, 0.0)


class Triangles(Movable):
    '''Any 3d shape made of triangles.

    '''

    def __init__(self, window, coords, color=1., add=False):
        super().__init__(window)
        self.gl_type = GL_TRIANGLES
        self.coords = np.array(coords)
        self.num = self.coords.shape[1]
        self.txtcoords = None
        self.color = color
        if hasattr(color, '__iter__'):
            self.colors = np.array(np.tile(color[:3], self.num), dtype='byte')
        else:
            self.colors = np.array(np.repeat(color * 255, self.num * 3), dtype='byte')
        if add: self.add()


class STL(Triangles):
    '''Get all the vertices from an stl file

    '''

    def __init__(self, window, fn, color=1, scale=1, add=False):
        stl_mesh = mesh.Mesh.from_file(fn)

        xs = stl_mesh.vectors[:, :, 0].ravel()
        ys = stl_mesh.vectors[:, :, 1].ravel()
        zs = stl_mesh.vectors[:, :, 2].ravel()
        super().__init__(window, np.array([xs * scale, ys * scale, zs * scale]), color, add)


class Shape(Movable):
    '''An arbitrary polygon.

    '''

    def __init__(self, window, coords, color=1., add=False):
        super().__init__(window)
        self.gl_type = GL_POLYGON
        self.coords = np.array(coords)
        self.num = self.coords.shape[1]
        self.txtcoords = None
        self.color = color
        if hasattr(color, '__iter__'):
            self.colors = np.array(np.tile(color[:3], self.num), dtype='byte')
        else:
            self.colors = np.array(np.repeat(color * 255, self.num * 3), dtype='byte')
        if add: self.add()


class Horizon(Shape):
    '''A horizon rendered out to some large distance off.

    '''

    def __init__(self, window, depth, dist, color=1., add=False):
        c = np.array([[-dist, -dist, dist, dist], [depth, depth, depth, depth], [-dist, dist, dist, -dist]])
        super(Horizon, self).__init__(window, coords=c, color=color)


class Spherical_segment(Movable):
    '''a spherical segment of any color that can be raised or tilted.
    Making this with phi degrees for the top and bottom polar angles,
    which is 0 at the north pole, and 90 at the south pole.

    '''

    def __init__(self, window, polang_top=0, polang_bot=90, radius=1.,
                 color=0., elres=60, azres=60, add=False):
        super(Spherical_segment, self).__init__(window)
        self.gl_type = GL_QUADS
        self.elres = elres
        self.azres = azres
        self.init_coords(polang_top, polang_bot, radius, color)
        if add: self.add()

    def change_color(self, color):
        '''you have to have added the instance already so it has a vertex
        list. color is 0-255
        '''
        self.colors[:] = color
        self.vl.colors[:] = self.colors

    def init_coords(self, polang_top=0, polang_bot=180, radius=1., color=1.):
        xlist, ylist, zlist = [], [], []
        # change to radians
        phi1 = polang_top * np.pi / 180
        phi2 = polang_bot * np.pi / 180
        # goes a fraction of the way around, so choose num segs based on elres
        els = np.linspace(phi1, phi2, int(ceil(self.elres * (phi2 - phi1) / (2 * np.pi))) + 1)
        # this always goes all the way around
        azs = np.linspace(0, 2 * np.pi, int(self.azres) + 1)
        for eind in range(len(els) - 1):
            n_elev = els[eind]  # northern and southern elevations
            s_elev = els[eind + 1]  # 0 at npole, pi/2 at equator, pi at spole
            n_y = radius * np.cos(n_elev)  # calculate the y coord
            s_y = radius * np.cos(s_elev)
            n_prad = radius * np.sin(n_elev)  # radius projected onto xz plane
            s_prad = radius * np.sin(s_elev)
            y1, y2, y3, y4 = n_y, s_y, s_y, n_y

            for aind in range(len(azs) - 1):
                w_azi = azs[aind]  # western and eastern azimuths
                e_azi = azs[aind + 1]

                x1 = n_prad * np.cos(w_azi)
                x2 = s_prad * np.cos(w_azi)
                x3 = s_prad * np.cos(e_azi)
                x4 = n_prad * np.cos(e_azi)

                z1 = n_prad * np.sin(w_azi)
                z2 = s_prad * np.sin(w_azi)
                z3 = s_prad * np.sin(e_azi)
                z4 = n_prad * np.sin(e_azi)

                xlist.extend([x1, x2, x3, x4])
                ylist.extend([y1, y2, y3, y4])
                zlist.extend([z1, z2, z3, z4])

        self.coords = np.array([xlist, ylist, zlist])
        self.num = len(xlist)
        self.txtcoords = None
        self.colors = np.array(np.repeat(color * 255, self.num * 3), dtype='byte')


# class pts_class(movable_fast_class):
class Points(Movable):
    '''Randomly distributed points in a specified x, y and, z range.

    '''

    def __init__(self, window, num=1000, dims=[(-1, 1), (-1, 1), (-1, 1)],
                 color=1., pt_size=1, add=False):

        super(Points, self).__init__(window)
        self.gl_type = GL_POINTS
        self.pt_size = pt_size
        self.num = num
        self.color = color
        self.colors = np.array(np.repeat(color * 255, self.num * 3), dtype='byte')
        self.dims = np.array(dims)
        if len(self.dims.shape) == 1:
            self.dims = np.array([[-dims[0], dims[0]], [-dims[1], dims[1]], [-dims[2], dims[2]]])
        self.txtcoords = None

        self.init_coords()
        if add: self.add()

    def set_num(self, num):
        self.num = num
        self.colors = np.array(np.repeat(self.color * 255, self.num * 3), dtype='byte')
        self.init_coords()
        self.txtcoords = None

    def remove_subset(self, arr):
        ''' provide bool np.array of vertices to remove '''
        arr = np.array([not i for i in arr])
        self.coords = self.coords[:, arr]
        self.num = self.coords.shape[1]
        self.colors = np.array(np.repeat(self.color * 255, self.num * 3), dtype='byte')
        self.txtcoords = None

    def shuffle(self, shuf=True):
        self.colors = np.array(np.repeat(self.color * 255, self.num * 3), dtype='byte')
        self.init_coords()
        self.txtcoords = None

    def init_coords(self):
        self.coords = np.array([np.random.uniform(self.dims[0][0], self.dims[0][1], self.num),
                                np.random.uniform(self.dims[1][0], self.dims[1][1], self.num),
                                np.random.uniform(self.dims[2][0], self.dims[2][1], self.num)])

    def set_pt_size(self, pt_size):
        self.pt_size = pt_size

    def set_state(self):
        super(Points, self).set_state()
        glPointSize(self.pt_size)

    def unset_state(self):
        super(Points, self).unset_state()
        glPointSize(1)


class pts_trail_class(Movable):
    def __init__(self, window, num=1000, dims=[(-1, 1), (-1, 1), (-1, 1)],
                 color=1., trail_size=10, add=False):

        self.gl_type = GL_POINTS
        self.num = num * trail_size
        self.trail_size = trail_size
        self.color = color
        self.colors = np.array(np.repeat(color * 255, self.num * 3), dtype='byte')
        self.dims = np.array(dims)
        if len(self.dims.shape) == 1:
            self.dims = np.array([[-dims[0], dims[0]], [-dims[1], dims[1]], [-dims[2], dims[2]]])

        self.init_coords()
        if add: self.add()

        super(pts_class2, self).__init__(window)

    def set_num(self, num):
        self.num = num
        self.colors = np.array(np.repeat(self.color * 255, self.num * 3), dtype='byte')
        self.init_coords()
        self.txtcoords = None

    def shuffle(self):
        self.colors = np.array(np.repeat(self.color * 255, self.num * 3), dtype='byte')
        self.init_coords()
        self.txtcoords = None

    def init_coords(self):
        self.coords = np.array([np.random.uniform(self.dims[0][0], self.dims[0][1], self.num),
                                np.random.uniform(self.dims[1][0], self.dims[1][1], self.num),
                                np.random.uniform(self.dims[2][0], self.dims[2][1], self.num)])


class Dot_cohere_cyl(Movable):
    def __init__(self, window, num=1000, color=0.1, z=1, rho=1, pt_size=3,
                 duration=10, regions=[[[-np.pi, np.pi, -1, 1], [1, -.01, .01, -.01, .01]]], add=False):  ###
        super(Dot_cohere_cyl, self).__init__(window)
        self.gl_type = GL_POINTS
        self.pt_size = pt_size
        self.num = num
        self.z = z
        self.rho = rho
        self.color = color
        self.colors = np.array(np.repeat(color * 255, self.num * 3), dtype='byte')
        self.txtcoords = None

        self.duration = duration
        self.regions = regions

        self.init_coords()
        if add: self.add()

    def init_coords(self):
        self.phis = np.random.uniform(-np.pi, np.pi, self.num)
        self.zs = np.random.uniform(-self.z, self.z, self.num)

        self.durations = np.random.randint(0, self.duration, self.num)
        self.motions = np.zeros((self.num, 2))
        for pt_ind in range(self.num):
            self.motions[pt_ind, :] = self.assign_motion(self.phis[pt_ind], self.zs[pt_ind])

        xs, ys = self.rho * np.cos(self.phis), self.rho * np.sin(self.phis)
        self.coords = np.array([xs, self.zs, ys])  # switch for opengl

    def remove_region(self, index=0):
        self.regions.pop(index)

    def add_region(self, reg, index=0):
        self.regions.insert(index, reg)

    def replace_region(self, reg, index=0):
        self.regions[index] = reg

    def replace_p(self, p, index=0):  ###
        self.regions[index][1][0] = p  ###

    def move(self):
        self.phis += self.motions[:, 0]
        self.phis[self.phis > np.pi] -= 2 * np.pi
        self.phis[self.phis < -np.pi] += 2 * np.pi
        self.zs += self.motions[:, 1]
        self.durations -= 1

        for pt_ind in range(self.num):
            # if we drifted out of bounds, or durations got below 0, pick a random pos
            if (not (-self.z < self.zs[pt_ind] < self.z)) or (self.durations[pt_ind] < 0):
                self.phis[pt_ind] = np.random.uniform(-np.pi, np.pi)
                self.zs[pt_ind] = np.random.uniform(-self.z, self.z)
                self.durations[pt_ind] = self.duration
                self.motions[pt_ind, :] = self.assign_motion(self.phis[pt_ind], self.zs[pt_ind])

        self.cyl_to_coords()
        # xs,ys = self.rho*np.cos(self.phis), self.rho*np.sin(self.phis)
        # self.coords = np.array([xs, self.zs, ys]) #switch for opengl

    def assign_motion(self, phi, z):
        for region in self.regions:
            reg, direct = region
            phi0, phi1, z0, z1 = reg
            p, phi_l, phi_h, z_l, z_h = direct
            if (np.random.rand() < p) and z0 < z < z1 and phi0 < phi < phi1:
                return np.random.uniform(phi_l, phi_h), np.random.uniform(z_l, z_h)

    def cyl_to_coords(self):
        xs, ys = self.rho * np.cos(self.phis), self.rho * np.sin(self.phis)
        self.coords = np.array([xs, self.zs, ys])  # switch for opengl
        if self.visible:
            self.vl.vertices = self.coords.T.flatten()

    def set_state(self):
        super(Dot_cohere_cyl, self).set_state()
        glPointSize(self.pt_size)

    def unset_state(self):
        super(Dot_cohere_cyl, self).unset_state()
        glPointSize(1)


class Dot_cohere_sph(Movable):
    def __init__(self, window, num=1000, color=0.1, r=1, pt_size=3,
                 speed=.01, duration=10, add=False):  ###
        '''Initialize the sphere points.

        '''
        super(Dot_cohere_sph, self).__init__(window)
        self.gl_type = GL_POINTS
        self.pt_size = pt_size
        self.num = num
        self.speed = speed
        self.r = r
        self.color = color
        self.colors = np.array(np.repeat(color * 255, self.num * 3), dtype='byte')
        self.txtcoords = None

        self.duration = duration
        self.regions = []

        self.init_coords()
        if add: self.add()

    def add_region(self, azimuth=0, elevation=0, radius=.5,
                   flow_azimuth=90, flow_elevation=0, speed=None,
                   coherence=1):
        '''Add a region implemented by a dictionary, where points flow
        non-randomly.

        '''
        if speed is not None:
            speed = self.speed

        theta = (elevation + 90) * np.pi / 180
        phi = (azimuth - 90) * np.pi / 180
        vec = self.sph_to_cart(theta, phi)

        theta = (flow_elevation + 90) * np.pi / 180
        phi = (flow_azimuth - 90) * np.pi / 180
        flow_vec = self.sph_to_cart(theta, phi)

        region = {'azimuth': azimuth, 'elevation': elevation, 'vec': vec, 'radius': radius,
                  'flow_azimuth': azimuth, 'flow_elevation': elevation, 'flow_vec': flow_vec,
                  'speed': speed, 'coherence': coherence}

        self.regions.append(region)

    def remove_region(self, ind=-1):
        '''Remove one of the regions.

        '''
        self.regions.pop(ind)

    def update_coherence(self, reg_ind, coherence):
        '''Change coherence in a region.

        '''
        self.regions[reg_ind]['coherence'] = coherence

    def update_speed(self, reg_ind, speed):
        '''Change dot speed in a region.

        '''
        self.regions[reg_ind]['speed'] = speed

    def update_radius(self, reg_ind, radius):
        '''Change radius of a region.

        '''
        self.regions[reg_ind]['radius'] = radius

    def update_region(self, reg_ind, azimuth=None, elevation=None):
        '''Change location of a region.

        '''
        if azimuth is not None:
            self.regions[reg_ind]['azimuth'] = azimuth
        if elevation is None:
            self.regions[reg_ind]['elevation'] = elevation

        theta = (self.regions[reg_ind]['elevation'] + 90) * np.pi / 180
        phi = (self.regions[reg_ind]['azimuth'] - 90) * np.pi / 180
        self.regions[reg_ind]['vec'] = self.sph_to_cart(theta, phi)

    def update_flow(self, reg_ind, flow_azimuth=None, flow_elevation=None):
        '''Change flow direction of coherent dots in a region.

        '''
        if flow_azimuth is not None:
            self.regions[reg_ind]['flow_azimuth'] = flow_azimuth
        if flow_elevation is None:
            self.regions[reg_ind]['flow_elevation'] = flow_elevation

        theta = (self.regions[reg_ind]['flow_elevation'] + 90) * np.pi / 180
        phi = (self.regions[reg_ind]['flow_azimuth'] - 90) * np.pi / 180
        self.regions[reg_ind]['flow_vec'] = self.sph_to_cart(theta, phi)

    def gc_distances(self, target_pt):
        '''Return the great-circle distance between a unit vector and the
        coordinate points.

        '''
        return np.arccos(np.dot(target_pt, self.coords))

    def rotate(self):
        '''Rodrigues' rotation formula, takes a position vector v, a unit
        rotation vector, and an angle, and returns the new position.

        '''
        cosangs, sinangs = np.cos(self.speeds), np.sin(self.speeds)

        # the first term scales the vector down
        t1 = self.coords * cosangs

        # the second skews it (via vector addition) toward the new
        # rotational position.
        t2 = np.cross(self.uvecs, self.coords, axis=0) * sinangs

        # The third term re-adds the height (relative to k that was
        # lost by the first term.
        t3 = self.uvecs * np.sum(self.uvecs * self.coords, axis=0) * (1 - cosangs)

        self.coords = t1 + t2 + t3

    def sph_to_cart(self, theta, phi):
        '''Spherical to cartesion coordinates, with phi as the azimuthal
        angle, and theta as the angle from the north pole.

        '''
        x = self.r * np.sin(theta) * np.cos(phi)
        y = self.r * np.sin(theta) * np.sin(phi)
        # use phi just to make the expression broadcast to the proper length
        z = self.r * np.cos(theta) + 0 * phi
        # rearrange for opengl
        return np.array([x, -z, y])

    def rand_pts(self, num):
        '''Make random points on the sphere, uniformly distributed.

        '''
        phi = np.random.uniform(-np.pi, np.pi, num)
        # arccos needed to make distribution uniform:
        theta = np.arccos(np.random.uniform(-1, 1, num))
        x = self.r * np.sin(theta) * np.cos(phi)
        y = self.r * np.sin(theta) * np.sin(phi)
        # use 0*phi just to make the expression broadcast to the proper length
        z = self.r * np.cos(theta) + 0 * phi
        # in opengl, y is z, and z goes in the negative direction
        return np.array((x, -z, y))

    def assign_uvecs(self, inds, target_pts):
        '''Assign unit rotation vectors to each coordinate point. They must be
        perpendicular to the coordinate point and some other point we
        are targeting on the sphere.

        '''
        self.tp = target_pts
        self.inds = inds
        vs = np.cross(self.coords[:, inds], target_pts, axis=0)
        vs = vs / np.linalg.norm(vs, axis=0)
        self.uvecs[:, inds] = vs

    def init_coords(self):
        '''Initial sphere of dot positions, their durations, speeds, and
        rotational motion vectors.

        '''
        # initial positions
        self.coords = np.zeros((3, self.num))
        self.coords = self.rand_pts(self.num)

        # initial directions
        self.uvecs = np.zeros((3, self.num))
        self.assign_uvecs(np.arange(self.num), self.rand_pts(self.num))

        # durations, how many frames left for each, temporary dot
        self.durations = np.random.randint(0, self.duration, self.num)

        # speeds---since the direction vectors can point anywhere, we don't need negative speeds
        self.speeds = np.random.uniform(self.speed, self.speed, self.num)

    def move(self):
        '''Move every point on the sphere by rotating along its uvec

        '''
        # decrement the durations
        self.durations -= 1
        inds = np.where(self.durations <= 0)[0]

        # make new positions where old pts have expired
        self.coords[:, inds] = self.rand_pts(len(inds))
        # start with random directions
        self.assign_uvecs(inds, self.rand_pts(len(inds)))
        self.durations[inds] = self.duration
        self.speeds[inds] = np.random.uniform(self.speed, self.speed, len(inds))

        # check each region, and update uvec direction if needed
        for region in self.regions:
            # which points are close to the target area?
            inds = np.where(self.gc_distances(region['vec']) < region['radius'])[0]
            # and only some fraction of those flow coherently
            inds = inds[np.where(np.random.rand(len(inds)) < region['coherence'])]

            # change the uvecs for those inds
            if len(inds) > 0:
                self.assign_uvecs(inds, np.tile(region['flow_vec'], (len(inds), 1)).T)
                self.speeds[inds] = np.random.uniform(region['speed'], region['speed'], len(inds))

        # rotate all points along their uvecs, according to speed
        self.rotate()

        if self.visible:
            self.vl.vertices = self.coords.T.flatten()

    def set_state(self):
        super(Dot_cohere_sph, self).set_state()
        glPointSize(self.pt_size)

    def unset_state(self):
        super(Dot_cohere_sph, self).unset_state()
        glPointSize(1)


class Incoherent_flower(Dot_cohere_sph):
    '''Define flower shape, size, motion, and coherence, then background
    motion and coherence

    '''

    def __init__(self, window, num=1000, color=0.1, r=1, pt_size=3,
                 speed=.01, duration=10, add=False):
        super(Incoherent_flower, self).__init__(window, num, color, r, pt_size,
                                                speed, duration, add)

    def move_flower(self, ind, d_az):
        '''Move the flower region in azimuth

        '''
        new_az = self.regions[ind]['azimuth'] + d_az
        self.update_region(ind, azimuth=new_az, elevation=0)
        new_flow_dir = new_az + 90 * np.sign(d_az)
        self.update_flow(ind, flow_azimuth=new_flow_dir, flow_elevation=0)

    def rotate(self):
        '''Rodrigues' rotation formula, takes a position vector v, a unit
        rotation vector, and an angle, and returns the new position.

        '''
        cosangs, sinangs = np.cos(self.speeds), np.sin(self.speeds)

        # the first term scales the vector down
        t1 = self.coords * cosangs

        # the second skews it (via vector addition) toward the new
        # rotational position.
        t2 = np.cross(self.uvecs, self.coords, axis=0) * sinangs

        # The third term re-adds the height (relative to k that was
        # lost by the first term.
        t3 = self.uvecs * np.sum(self.uvecs * self.coords, axis=0) * (1 - cosangs)

        self.coords = t1 + t2 + t3

    def move(self):
        '''Move every point on the sphere by rotating along its uvec

        '''
        # decrement the durations
        self.durations -= 1
        inds = np.where(self.durations <= 0)[0]

        # make new positions where old pts have expired
        self.coords[:, inds] = self.rand_pts(len(inds))
        # start with random directions
        self.assign_uvecs(inds, self.rand_pts(len(inds)))
        self.durations[inds] = self.duration
        self.speeds[inds] = np.random.uniform(self.speed, self.speed, len(inds))

        # check each region, and update uvec direction if needed
        for region in self.regions:
            # which points are close to the target area?
            inds = np.where(self.gc_distances(region['vec']) < region['radius'])[0]
            # and only some fraction of those flow coherently
            inds = inds[np.where(np.random.rand(len(inds)) < region['coherence'])]

            # change the uvecs for those inds
            if len(inds) > 0:
                self.assign_uvecs(inds, np.tile(region['flow_vec'], (len(inds), 1)).T)

        # rotate all points along their uvecs, according to speed
        self.rotate()

        if self.visible:
            self.vl.vertices = self.coords.T.flatten()

    def set_state(self):
        super(Incoherent_flower, self).set_state()
        glPointSize(self.pt_size)

    def unset_state(self):
        super(Incoherent_flower, self).unset_state()
        glPointSize(1)


class lines_class(Movable):
    def __init__(self, window, num=1000, dims=[(-1, 1), (-1, 1), (-1, 1)],
                 color=1., ln_width=1, add=False):

        self.gl_type = GL_LINES
        self.ln_width = ln_width
        self.num = num * 2
        self.color = color
        self.colors = np.array(np.repeat(color * 255, self.num * 3), dtype='byte')
        self.dims = np.array(dims)
        if len(self.dims.shape) == 1:
            self.dims = np.array([[-dims[0], dims[0]], [-dims[1], dims[1]], [-dims[2], dims[2]]])

        self.txtcoords = None
        super(lines_class, self).__init__(window)

        self.init_coords()
        if add: self.add()

    def init_coords(self):
        self.coords = np.array([np.random.uniform(self.dims[0][0], self.dims[0][1], self.num),
                                np.random.uniform(self.dims[1][0], self.dims[1][1], self.num),
                                np.random.uniform(self.dims[2][0], self.dims[2][1], self.num)])

    def set_ln_width(self, ln_width):
        self.ln_width = ln_width

    def set_state(self):
        super(lines_class, self).set_state()
        glLineWidth(self.ln_width)

    def unset_state(self):
        super(lines_class, self).unset_state()
        glLineWidth(1)


class Bars(Shape):
    '''A simple rotating bar.

    '''

    def __init__(self, window, width=.15, height=2, dist=.8, color=1.,
                 add=False):
        xl, xr = -width / 2., width / 2.
        yb, yt = -height / 2., height / 2.
        z = -dist
        coords = np.array([[xl, xl, xr, xr], [yb, yt, yt, yb], [z, z, z, z]])
        super().__init__(window, coords, color)

    def set_width_height(self, wh):
        width, height = wh
        xl, xr = -width / 2., width / 2.
        yb, yt = -height / 2., height / 2.
        self.coords[:2] = np.array([[xl, xl, xr, xr], [yb, yt, yt, yb]])

    def set_top(self, yt):
        self.coords[1, 1:3] = yt

    def set_bot(self, yb):
        self.coords[1, 0:4:3] = yb

    def set_dist(self, dist):
        self.coords[2] = [-dist, -dist, -dist, -dist]

    def set_top_ang(self, ang):
        x = self.coords[2, 0]
        y = tan(ang * np.pi / 180) * abs(x)
        self.set_top(y)

    def set_bot_ang(self, ang):
        x = self.coords[2, 0]
        y = tan(ang * np.pi / 180) * abs(x)
        self.set_bot(y)


class cbar_class(Movable):
    '''A multicolored rotating bar.'''

    def __init__(self, window, width=.15, height=2, dist=.8, color=(1, 1, 1),
                 add=False):
        Movable.__init__(self, window)

        self.width = width
        self.height = height
        self.num = 12
        self.gl_type = GL_QUADS

        xl, xr = -width / 2., width / 2.
        yb, yt = -height / 2., height / 2.
        z = -dist

        self.coords = np.array(
            [[xl / 4, xl / 4, xr / 4, xr / 4, 2 * xl / 3, 2 * xl / 3, 2 * xr / 3, 2 * xr / 3, xl, xl, xr, xr],
             [yb, yt, yt, yb, yb, yt, yt, yb, yb, yt, yt, yb],
             [z + .02, z + .02, z + .02, z + .02, z + .01, z + .01, z + .01, z + .01, z, z, z, z]])
        self.colors = np.array([[255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255],
                                [255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255],
                                [255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255]])

        if add: self.add()


class cbarr_class(Movable):
    '''A multicolored rotating bar---reverse colors.

    '''

    def __init__(self, window, width=.15, height=2, dist=.8, color=(1, 1, 1),
                 add=False):
        Movable.__init__(self, window)

        self.width = width
        self.height = height
        self.num = 12
        self.gl_type = GL_QUADS

        xl, xr = -width / 2., width / 2.
        yb, yt = -height / 2., height / 2.
        z = -dist

        self.coords = np.array(
            [[xl / 4, xl / 4, xr / 4, xr / 4, 2 * xl / 3, 2 * xl / 3, 2 * xr / 3, 2 * xr / 3, xl, xl, xr, xr],
             [yb, yt, yt, yb, yb, yt, yt, yb, yb, yt, yt, yb],
             [z + .02, z + .02, z + .02, z + .02, z + .01, z + .01, z + .01, z + .01, z, z, z, z]])
        self.colors = np.array([[0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0],
                                [0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0],
                                [0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0]])

        if add: self.add()


class bars_class(Movable):
    '''Oriented bar np.array in the frontal visual field.

    '''

    def __init__(self, window, dist=.25, w=.02, h=.15, o=np.pi / 4, z=-0.8,
                 xlim=[-1., 1.], ylim=[-1., 1.], color=1, add=False):
        Movable.__init__(self, window)

        self.dist = dist
        self.width = w
        self.height = h
        self.orientation = o
        self.z = z
        self.xlim = xlim
        self.ylim = ylim

        self.add_bar_field()
        self.make_coord_list()

        self.num = len(self.bars) * 4
        self.gl_type = GL_QUADS

        self.colors = np.array(np.repeat(color * 255, self.num * 3), dtype='byte')

        if add: self.add()

    def bar_pts(self, x, y, z, w, h, o):
        # get the 4 vertices of a bar given x and y of its center
        xl, xr, yb, yt = -h / 2., h / 2., -w / 2., w / 2.

        pts = np.array([[xl, xl, xr, xr], [yb, yt, yt, yb], [z, z, z, z]])
        pts = np.dot(rotmat([0, 0, -1.], -o), pts)
        pts[0] += x
        pts[1] += y
        return pts

    def add_bar_field(self):
        self.bars = []
        for i in np.arange(self.xlim[0], self.xlim[1], self.dist):
            for j in np.arange(self.ylim[0], self.ylim[1], self.dist):
                self.bars.append([i, j, self.z, self.width, self.height, self.orientation])

    def orient(self, o):
        self.orientation = o
        self.add_bar_field()
        self.make_coord_list()

    def d_height(self, h):
        self.height = h
        self.add_bar_field()
        self.make_coord_list()

    def make_coord_list(self):
        num_bars = len(self.bars)
        self.coords = np.zeros([3, 4 * num_bars])
        for i in np.arange(num_bars):
            b = self.bars[i]
            pts = self.bar_pts(b[0], b[1], b[2], b[3], b[4], b[5])
            self.coords[0, i * 4:(i + 1) * 4] = pts[0]
            self.coords[1, i * 4:(i + 1) * 4] = pts[1]
            self.coords[2, i * 4:(i + 1) * 4] = pts[2]


class Disks(Movable):
    '''discs'''

    def __init__(self, window, radius, color=(1, 1, 1), num_pts=60, add=False):
        Movable.__init__(self, window)
        a = np.linspace(0, 2 * np.pi, num_pts, endpoint=False)
        self.coords = np.array([radius * np.cos(a), radius * np.sin(a), -np.ones(num_pts)])
        self.num = num_pts
        self.colors = np.tile(color, num_pts) * 255
        self.gl_type = GL_POLYGON
        self.txtcoords = None
        if add: self.add


class disk_class(Movable):
    '''discs'''

    def __init__(self, window, radius, color=(1, 1, 1), num_pts=60, add=False):
        Movable.__init__(self, window)
        a = np.linspace(0, 2 * np.pi, num_pts, endpoint=False)
        self.coords = np.array([radius * np.cos(a), radius * np.sin(a), -np.ones(num_pts)])
        self.num = num_pts
        self.colors = np.tile(color, num_pts) * 255
        self.gl_type = GL_POLYGON
        self.txtcoords = None
        if add: self.add


class sphere_lines_class(Movable):
    '''A class of lines circling a sphere for calibration.

    '''

    def __init__(self, window, num_lines=12, num_segs=50, add=False):

        Movable.__init__(self, window)

        self.colors = None

        self.line_angs = np.linspace(-np.pi, np.pi, num_lines + 2)[1:-1]
        self.seg_angs = np.linspace(0, 2 * np.pi, num_segs + 1)

        self.gl_type = GL_LINES

        self.num = num_lines * num_segs * 2 * 3

        self.coords = np.zeros([3, self.num])
        self.init_coords()

        if add: self.add()

    def init_coords(self):
        ind = 0

        for line_ang in self.line_angs:
            for s in np.arange(len(self.seg_angs) - 1):
                cseg_ang = self.seg_angs[s]
                nseg_ang = self.seg_angs[s + 1]

                r = np.cos(line_ang)

                # around z
                # first coord of a segment
                self.coords[0, ind] = r * np.cos(cseg_ang)
                self.coords[1, ind] = r * np.sin(cseg_ang)
                self.coords[2, ind] = np.sin(line_ang)
                ind += 1

                # second part
                self.coords[0, ind] = r * np.cos(nseg_ang)
                self.coords[1, ind] = r * np.sin(nseg_ang)
                self.coords[2, ind] = np.sin(line_ang)
                ind += 1

                # around y
                # first coord of a segment
                self.coords[0, ind] = r * np.cos(cseg_ang)
                self.coords[1, ind] = np.sin(line_ang)
                self.coords[2, ind] = r * np.sin(cseg_ang)
                ind += 1

                # second part
                self.coords[0, ind] = r * np.cos(nseg_ang)
                self.coords[1, ind] = np.sin(line_ang)
                self.coords[2, ind] = r * np.sin(nseg_ang)
                ind += 1

                # around x
                # first coord of a segment
                self.coords[0, ind] = np.sin(line_ang)
                self.coords[1, ind] = r * np.sin(cseg_ang)
                self.coords[2, ind] = r * np.cos(cseg_ang)
                ind += 1

                # second part
                self.coords[0, ind] = np.sin(line_ang)
                self.coords[1, ind] = r * np.sin(nseg_ang)
                self.coords[2, ind] = r * np.cos(nseg_ang)
                ind += 1


class Tree(Movable):
    def __init__(self, window, distance=3, num_sides=12, num_levels=3, color=0.0, add=False):
        self.gl_type = GL_QUADS
        self.color = color
        self.num_sides = num_sides
        self.num_levels = num_levels
        angs = np.linspace(0, 2 * np.pi, self.num_sides + 1)
        self.cos_angs = np.cos(angs)
        self.sin_angs = np.sin(angs)
        super(Tree, self).__init__(window)
        self.init_coords(distance=distance)
        if add: self.add()

    def add_branch(self, x1, y1, z1, x2, y2, z2, rad):
        '''Add all the coords for a slanted cylinder section of a branch.'''
        xstart, ystart, zstart = x1 + rad * self.cos_angs, np.repeat(y1, self.num_sides), z1 + rad * self.sin_angs
        xend, yend, zend = x2 + rad * self.cos_angs, np.repeat(y2, self.num_sides), z2 + rad * self.sin_angs
        for side in range(self.num_sides):
            ind1, ind2 = side, mod(side + 1, self.num_sides)
            self.xlist.extend([xstart[ind1], xstart[ind2], xend[ind2], xend[ind1]])
            self.ylist.extend([ystart[ind1], ystart[ind2], yend[ind2], yend[ind1]])
            self.zlist.extend([zstart[ind1], zstart[ind2], zend[ind2], zend[ind1]])

    def add_tree(self, level=0, pstart_pt=np.array([0, -1, 0.]), pend_pt=np.array([0, 0, 0.]), lens=(1, .2, .7),
                 angs=(30., 15.), nbrs=3):
        # an np.array of vectors representing start and stop pts of branches
        start_pt = pend_pt  # new start_pt
        # make a random branch length
        # lens[0] is mean length, lens[1] is std, lens[2] is decay with branch level
        ln = lens[2] ** level * lens[0] + lens[2] ** level * np.random.randn() * lens[1]

        # make a random angle
        ang = (angs[0] + 90 + angs[1] * np.random.randn()) * np.pi / 180  # random angle
        if level == 0: ang = np.pi / 2.
        nx, ny, nz = ln * np.cos(ang), ln * np.sin(ang), 0.
        raz = 2 * np.pi * np.random.rand()  # swing around by a random azimuth
        nx, nz = nx * np.cos(raz), nx * np.sin(raz)

        # angle it to the existing branch
        stem = pend_pt - pstart_pt
        xyang, xzang = np.arctan2(stem[1], stem[0]) - np.pi / 2, np.arctan2(stem[2], stem[0])
        nx, ny, nz = nx * np.cos(xyang) - ny * np.sin(xyang), nx * np.sin(xyang) + ny * np.cos(
            xyang), nz  # xy rot (along z)
        nx, ny, nz = nx * np.cos(xyang) + nz * np.sin(xyang), ny, -nx * np.sin(xyang) + nz * np.cos(
            xyang)  # xz rot (along y)
        end_pt = np.array([nx, ny, nz])
        # and translate
        end_pt += start_pt

        # if poisson(nlevs)+1>lev:
        if level < self.num_levels:
            for i in np.arange(nbrs):
                self.add_branch(start_pt[0], start_pt[1], start_pt[2], end_pt[0], end_pt[1], end_pt[2],
                                0.05 / (level + 1))
                self.add_tree(level + 1, start_pt, end_pt, lens, angs, nbrs)

    def init_coords(self, distance=0):
        self.xlist, self.ylist, self.zlist = [], [], []
        self.add_tree(level=0)

        self.coords = np.array([self.xlist, self.ylist, self.zlist])
        self.coords[2] -= distance
        self.num = len(self.xlist)
        self.txtcoords = None
        self.colors = np.array(np.repeat(self.color * 255, self.num * 3), dtype='byte')


class Forest(Movable):
    def __init__(self, window, numtrees=50, add=False):
        self.gl_type = GL_QUADS
        self.color = .5

        self.init_coords(numtrees)

        if add: self.add()
        super(Forest, self).__init__(window)

    def init_coords(self, numtrees):

        numsides = 12
        angs = np.linspace(0, 2 * np.pi, numsides + 1)
        cosangs = np.cos(angs)
        sinangs = np.sin(angs)
        xlist, ylist, zlist = [], [], []

        # first the ground
        grnd = -.2
        xlist.extend([-100, 100, 100, -100])
        zlist.extend([-100, -100, 100, 100])
        ylist.extend([grnd, grnd, grnd, grnd])

        # now the trunks
        for i in range(numtrees):
            diameter = np.random.uniform(.02, .2)
            height = np.random.uniform(.4, 10)
            x, z = np.random.uniform(-10, 10, 2)
            for j in range(numsides):
                xlist.extend([x + diameter * cosangs[j],
                              x + diameter * cosangs[j],
                              x + diameter * cosangs[j + 1],
                              x + diameter * cosangs[j + 1]])
                zlist.extend([z + diameter * sinangs[j],
                              z + diameter * sinangs[j],
                              z + diameter * sinangs[j + 1],
                              z + diameter * sinangs[j + 1]])
                ylist.extend([grnd, grnd + height, grnd + height, grnd])

        self.coords = np.array([xlist, ylist, zlist])
        self.num = len(xlist)
        self.colors = np.array(np.repeat(self.color * 255, self.num * 3), dtype='byte')
        self.colors[:12] = 255


class grating_class(Sprite):
    '''Moving gratings and plaids in a single window.  Fast means put
    different values in the blue, red and green sequential channels.

    '''

    def __init__(self, window, vp=0, rate=120., add=False, fast=True, half=False):

        Sprite.__init__(self, window, vp=vp, half=half)

        # self.wd, self.ht = self.window.vpcoords[0,2], self.window.vpcoords[0,3]
        self.indices = indices((self.wd, self.ht))
        h, v = np.meshgrid(np.linspace(-np.pi / 4, np.pi / 4, self.wd), np.linspace(-np.pi / 4, np.pi / 4, self.ht))
        self.center_dists = np.sqrt(h ** 2 + v ** 2)
        self.atans = np.arctan2(h, v)

        self.rate = rate
        self.fast = fast
        self.gratings = []

        if add: self.add()

    def add_grating(self, sf=.1, tf=1, c=1, o=0, phi_i=0., sd=None, sdb=None, maxframes=500):
        frames = []
        data = np.zeros((self.ht, self.wd, 3), dtype='ubyte')

        # gaussian mask
        if sd:
            mask_ss = scipy.stats.norm.pdf(self.center_dists, 0, sd)
            mask_ss /= mask_ss.max()  # normalize
        else:
            mask_ss = np.ones([self.ht, self.wd])

        if sdb:
            mask_ssb = scipy.stats.norm.pdf(self.center_dists, 0, sdb)
            mask_ssb /= mask_ssb.max()  # normalize
        else:
            mask_ssb = np.ones([self.ht, self.wd])

        # spatial
        phi_ss = 2 * np.pi * sf * np.cos(self.atans + (o - np.pi / 2)) * self.center_dists

        # temporal
        if hasattr(tf, '__iter__'):
            tf_np.array = np.array(tf)  # is it changing?
        else:
            tf_np.array = np.repeat(tf, min(abs(self.rate / tf), maxframes))  # or constant?

        nframes = len(tf_np.array)
        phi_ts = np.cumsum(-2 * np.pi * tf_np.array / float(self.rate))

        for f in np.arange(nframes):
            if self.fast:  # different frames in each color (projected without color wheel)
                if f == 0:
                    prev_phi_t = 0  # first frame?
                else:
                    prev_phi_t = phi_ts[f - 1]
                phi_ts_b, phi_ts_r, phi_ts_g = np.linspace(prev_phi_t, phi_ts[f], 4)[1:]
            else:  # or grays
                phi_ts_b, phi_ts_r, phi_ts_g = np.repeat(phi_ts[f], 3)

            # blue channel --- we don't need sub_phi_ts[0] since it was displayed last time
            lum = mask_ssb * 127 * (1 + mask_ss * c * np.sin(phi_ss + phi_ts_b + phi_i))
            data[:, :, 2] = lum[:, :]
            # red channel
            lum = mask_ssb * 127 * (1 + mask_ss * c * np.sin(phi_ss + phi_ts_r + phi_i))
            data[:, :, 0] = lum[:, :]
            # green channel
            lum = mask_ssb * 127 * (1 + mask_ss * c * np.sin(phi_ss + phi_ts_g + phi_i))
            data[:, :, 1] = lum[:, :]
            # make each frame and append the image to the frames list
            frames.append(pyglet.image.ImageData(self.wd, self.ht, 'RGB', data.tostring()))

        # make the animation with all the frames and append it to the list of playable, moving gratings
        self.gratings.append(pyglet.image.Animation.from_image_sequence(frames, 1. / self.rate))

    def add_grating_fast(self, sf=.1, tf=1, c=1, o=0, phi_i=0., sd=None, sdb=None, num_frames=120):
        num_frames *= 3
        frames = []

        # gaussian mask
        if sd:
            mask_ss = scipy.stats.norm.pdf(self.center_dists, 0, sd)
            mask_ss /= mask_ss.max()  # normalize
        else:
            mask_ss = np.ones([self.ht, self.wd])

        if sdb:
            mask_ssb = scipy.stats.norm.pdf(self.center_dists, 0, sdb)
            mask_ssb /= mask_ssb.max()  # normalize
        else:
            mask_ssb = np.ones([self.ht, self.wd])

        sf = np.resize(sf, (num_frames))
        tf = np.resize(tf, (num_frames))
        c = np.resize(c, (num_frames))
        o = np.resize(o, (num_frames))
        phi_ss_np.array = 2 * np.pi * sf[newaxis, newaxis, :] * np.cos(
            self.atans[:, :, newaxis] + (o[newaxis, newaxis, :] - np.pi / 2)) * self.center_dists[:, :, newaxis]
        phi_ts_np.array = np.cumsum(-2 * np.pi * tf / float(self.rate * 3))

        lum = np.array(127 * (1 + mask_ss[:, :, newaxis] * c[newaxis, newaxis, :] * np.sin(
            phi_ss_np.array + phi_ts_np.array[newaxis, newaxis, :] + phi_i)), dtype='ubyte')
        frames = [pyglet.image.ImageData(self.wd, self.ht, 'RGB', lum[:, :, (f + 2, f, f + 1)].tostring()) for f in
                  np.arange(0, num_frames, 3)]

        # make the animation with all the frames and append it to the list of playable, moving gratings
        self.gratings.append(pyglet.image.Animation.from_image_sequence(frames, 1. / self.rate))

    def add_radial(self, sf=1., tf=1., c=1., phi_i=0, sd=None, sdb=None, maxframes=500):
        '''Add a circular radiating grating'''
        frames = []
        data = np.zeros((self.ht, self.wd, 3), dtype='ubyte')

        # gaussian mask
        if sd:
            mask_ss = scipy.stats.norm.pdf(self.center_dists, 0, sd)
            mask_ss /= mask_ss.max()  # normalize
        else:
            mask_ss = np.ones([self.ht, self.wd])

        if sdb:
            mask_ssb = scipy.stats.norm.pdf(self.center_dists, 0, sdb)
            mask_ssb /= mask_ssb.max()  # normalize
        else:
            mask_ssb = np.ones([self.ht, self.wd])

        # spatial
        # phi_ss = 2*pi*sf*np.cos(self.atans + (o-pi/2))*self.center_dists
        phi_ss = 2 * np.pi * sf * self.center_dists

        # temporal
        if hasattr(tf, '__iter__'):
            tf_np.array = np.array(tf)  # is it changing?
        else:
            tf_np.array = np.repeat(tf, min(abs(self.rate / tf), maxframes))  # or constant?

        nframes = len(tf_np.array)
        phi_ts = np.cumsum(-2 * np.pi * tf_np.array / float(self.rate))

        for f in np.arange(nframes):
            if self.fast:  # different frames in each color (projected without color wheel)
                if f == 0:
                    prev_phi_t = 0  # first frame?
                else:
                    prev_phi_t = phi_ts[f - 1]
                phi_ts_b, phi_ts_r, phi_ts_g = np.linspace(prev_phi_t, phi_ts[f], 4)[1:]
            else:  # or grays
                phi_ts_b, phi_ts_r, phi_ts_g = np.repeat(phi_ts[f], 3)

            # blue channel --- we don't need sub_phi_ts[0] since it was displayed last time
            lum = mask_ssb * 127 * (1 + mask_ss * c * np.sin(phi_ss + phi_ts_b + phi_i))
            data[:, :, 2] = lum[:, :]
            # red channel
            lum = mask_ssb * 127 * (1 + mask_ss * c * np.sin(phi_ss + phi_ts_r + phi_i))
            data[:, :, 0] = lum[:, :]
            # green channel
            lum = mask_ssb * 127 * (1 + mask_ss * c * np.sin(phi_ss + phi_ts_g + phi_i))
            data[:, :, 1] = lum[:, :]
            # make each frame and append the image to the frames list
            frames.append(pyglet.image.ImageData(self.wd, self.ht, 'RGB', data.tostring()))

        # make the animation with all the frames and append it to the list of playable, moving gratings
        self.gratings.append(pyglet.image.Animation.from_image_sequence(frames, 1. / self.rate))

    def add_plaid(self, sf1=.1, tf1=1, c1=1, o1=0, phi_i1=0.,
                  sf2=.1, tf2=1, c2=1, o2=0, phi_i2=0., sd=None, sdb=None, maxframes=500):
        frames = []
        data = np.zeros((self.ht, self.wd, 3), dtype='byte')

        # gaussian mask
        if sd:
            mask_ss = scipy.stats.norm.pdf(self.center_dists, 0, sd)
            mask_ss /= mask_ss.max()  # normalize
        else:
            mask_ss = np.ones([self.ht, self.wd])

        if sdb:
            mask_ssb = scipy.stats.norm.pdf(self.center_dists, 0, sdb)
            mask_ssb /= mask_ssb.max()  # normalize
        else:
            mask_ssb = np.ones([self.ht, self.wd])

        # spatial
        phi_ss1 = 2 * np.pi * sf1 * np.cos(self.atans + (o1 - np.pi / 2)) * self.center_dists
        phi_ss2 = 2 * np.pi * sf2 * np.cos(self.atans + (o2 - np.pi / 2)) * self.center_dists

        # temporal
        if hasattr(tf1, '__iter__'):
            lentf1 = len(tf1)
        else:
            lentf1 = self.rate / tf1
        if hasattr(tf2, '__iter__'):
            lentf2 = len(tf2)
        else:
            lentf2 = self.rate / tf2
        if hasattr(tf1, '__iter__'):
            tf1_np.array = np.array(tf1)  # is it changing?
        else:
            tf1_np.array = np.repeat(tf1, min(max(lentf1, lentf2), maxframes))  # or constant?
        if hasattr(tf2, '__iter__'):
            tf2_np.array = np.array(tf2)  # is it changing?
        else:
            tf2_np.array = np.repeat(tf2, min(max(lentf1, lentf2), maxframes))  # or constant?

        nframes = max(len(tf1_np.array), len(tf2_np.array))
        phi_t1s = np.cumsum(-2 * np.pi * tf1_np.array / float(self.rate))
        phi_t2s = np.cumsum(-2 * np.pi * tf2_np.array / float(self.rate))

        for f in np.arange(nframes):
            if self.fast:  # different frames in each color (projected without color wheel)
                if f == 0:
                    prev_phi_t1 = prev_phi_t2 = 0  # first frame?
                else:
                    prev_phi_t1 = phi_t1s[f - 1]
                    prev_phi_t2 = phi_t2s[f - 1]
                phi_t1s_b, phi_t1s_r, phi_t1s_g = np.linspace(prev_phi_t1, phi_t1s[f], 4)[1:]
                phi_t2s_b, phi_t2s_r, phi_t2s_g = np.linspace(prev_phi_t2, phi_t2s[f], 4)[1:]
            else:  # or grays
                phi_t1s_b, phi_t1s_r, phi_t1s_g = np.repeat(phi_t1s[f], 3)
                phi_t2s_b, phi_t2s_r, phi_t2s_g = np.repeat(phi_t2s[f], 3)

            # blue channel --- we don't need sub_phi_ts[0] since it was displayed last time
            lum = mask_ssb * 63 * (2 + mask_ss * (c1 * np.sin(phi_ss1 + phi_t1s_b + phi_i1) + \
                                                  c2 * np.sin(phi_ss2 + phi_t2s_b + phi_i2)))
            data[:, :, 2] = lum[:, :]
            # red channel
            lum = mask_ssb * 63 * (2 + mask_ss * (c1 * np.sin(phi_ss1 + phi_t1s_r + phi_i1) + \
                                                  c2 * np.sin(phi_ss2 + phi_t2s_r + phi_i2)))
            data[:, :, 0] = lum[:, :]
            # green channel
            lum = mask_ssb * 63 * (2 + mask_ss * (c1 * np.sin(phi_ss1 + phi_t1s_g + phi_i1) + \
                                                  c2 * np.sin(phi_ss2 + phi_t2s_g + phi_i2)))
            data[:, :, 1] = lum[:, :]

            # make each frame and append the image to the frames list
            frames.append(pyglet.image.ImageData(self.wd, self.ht, 'RGB', data.tostring()))

        # make the animation with all the frames and append it to the list of playable, moving gratings
        self.gratings.append(pyglet.image.Animation.from_image_sequence(frames, 1. / self.rate))

    def choose_grating(self, num):
        '''Assign one of the already generated gratings as the current
        display grating'''
        self.image = self.gratings[num]


class Sphere_segment(Movable):
    '''a y-axis-centered spherical segment defined by an inner and outer
    angles_(in degrees measured from the y-axis). Outer angle should
    be larger than inner , both smaller than 90 degrees.

    '''

    def __init__(self, window, sphere_rad=0.5, outer_ang=0.5,
                 inner_ang=0, color=1., add=False):

        self.gl_type = GL_QUADS
        self.color = color
        self.sphere_rad = sphere_rad
        self.outer_ang = float(outer_ang)
        self.inner_ang = float(inner_ang)
        self.init_coords()
        if add: self.add()
        super(Sphere_segment, self).__init__(window)

    def set_angs(self, outer_ang, inner_ang):
        self.inner_ang = inner_ang
        self.outer_ang = outer_ang
        self.init_coords()

    def set_color(self, color):
        self.color = color
        self.init_coords()

    def change_color(self, color):
        '''you have to have added the instance
        already so it has a vertex list. color is 0-255'''
        self.colors[:] = color
        self.vl.colors[:] = self.colors

    def init_coords(self):
        num_azis = 180
        num_elevs = 180
        xlist, ylist, zlist = [], [], []
        elevs = np.linspace(-np.pi / 2, np.pi / 2, num_elevs + 1)
        azis = np.linspace(0, 2 * np.pi, num_azis + 1)
        rad_a = np.sin(self.outer_ang / 180 * np.pi) * self.sphere_rad
        rad_b = np.sin(self.inner_ang / 180 * np.pi) * self.sphere_rad
        lower_elevation = np.cos(self.outer_ang / 180 * np.pi) * self.sphere_rad
        upper_elevation = np.cos(self.inner_ang / 180 * np.pi) * self.sphere_rad
        elevs = np.linspace(lower_elevation, upper_elevation, num_elevs)
        for elev_ind, elev_val in enumerate(elevs[:-1]):
            lower_y = elev_val
            upper_y = elevs[elev_ind + 1]
            lower_rad = np.sin(np.arccos(lower_y / self.sphere_rad)) * self.sphere_rad
            upper_rad = np.sin(np.arccos(upper_y / self.sphere_rad)) * self.sphere_rad
            y1, y2, y3, y4 = lower_y, upper_y, upper_y, lower_y
            for azi_ind in range(num_azis):
                cur_azi = azis[azi_ind]
                next_azi = azis[azi_ind + 1]
                x1 = lower_rad * np.cos(cur_azi) * self.sphere_rad
                x2 = upper_rad * np.cos(cur_azi) * self.sphere_rad
                x3 = upper_rad * np.cos(next_azi) * self.sphere_rad
                x4 = lower_rad * np.cos(next_azi) * self.sphere_rad

                z1 = lower_rad * np.sin(cur_azi) * self.sphere_rad
                z2 = upper_rad * np.sin(cur_azi) * self.sphere_rad
                z3 = upper_rad * np.sin(next_azi) * self.sphere_rad
                z4 = lower_rad * np.sin(next_azi) * self.sphere_rad

                xlist.extend([x1, x2, x3, x4])
                ylist.extend([y1, y2, y3, y4])
                zlist.extend([z1, z2, z3, z4])

        self.coords = np.array([xlist, ylist, zlist])
        self.num = len(xlist)
        self.txtcoords = None
        self.colors = np.array(np.repeat(self.color * 255, self.num * 3), dtype='byte')


class Grating_cylinder(Sprite):
    '''Moving gratings and plaids in a single window.  Fast means put
    different values in the blue, red and green sequential
    channels.

    '''

    def __init__(self, window, vp=0, rate=120., center=[0, 0],
                 add=False, fast=False, half=False):

        Sprite.__init__(self, window, vp=vp, half=half)

        # self.wd, self.ht = self.window.vpcoords[0,2], self.window.vpcoords[0,3]
        self.indices = indices((self.wd, self.ht))
        # h, v = np.meshgrid(np.linspace(-np.pi/4, np.pi/4, self.wd), np.linspace(-np.pi/4, np.pi/4, self.ht))
        h, v = np.meshgrid(np.arctan2(np.linspace(-1, 1, self.wd), 1), np.linspace(-np.pi / 4, np.pi / 4, self.ht))
        self.center_dists = np.sqrt((h + center[0]) ** 2 + (v + center[1]) ** 2)
        self.atans = np.arctan2(h + center[0], v + center[1])

        self.rate = rate
        self.fast = fast
        self.gratings = []
        self.vp = vp

        if add: self.add()

    def add_grating(self, sf=.1, tf=1, c=1, o=0, phi_i=0.,
                    sd=None, sdb=None, mask_reflect=False,
                    dots=False, dot_speed=[0, 0, 0.], maxframes=500):
        frames = []
        data = np.zeros((self.ht, self.wd, 3), dtype='ubyte')

        # gaussian mask
        if sd:
            mask_ss = scipy.stats.norm.pdf(self.center_dists, 0, sd)
            mask_ss /= mask_ss.max()  # normalize
        else:
            mask_ss = np.ones([self.ht, self.wd])

        if sdb:
            mask_ssb = scipy.stats.norm.pdf(self.center_dists, 0, sdb)
            mask_ssb /= mask_ssb.max()  # normalize
        else:
            mask_ssb = np.ones([self.ht, self.wd])

        if mask_reflect:
            half = int(self.wd / 2)
            mask_ss[:, -half:] = mask_ss[:, :half][:, ::-1]
            mask_ssb[:, -half:] = mask_ssb[:, :half][:, ::-1]

        # spatial
        phi_ss = 2 * np.pi * sf * np.cos(self.atans + (o - np.pi / 2)) * self.center_dists

        # temporal
        if hasattr(tf, '__iter__'):
            tf_np.array = np.array(tf)  # is it changing?
        else:
            tf_np.array = np.repeat(tf, min(abs(self.rate / tf), maxframes))  # or constant?

        nframes = len(tf_np.array)
        phi_ts = np.cumsum(-2 * np.pi * tf_np.array / float(self.rate))

        for f in np.arange(nframes):
            if self.fast:  # different frames in each color (projected without color wheel)
                if f == 0:
                    prev_phi_t = 0  # first frame?
                else:
                    prev_phi_t = phi_ts[f - 1]
                phi_ts_b, phi_ts_r, phi_ts_g = np.linspace(prev_phi_t, phi_ts[f], 4)[1:]
            else:  # or grays
                phi_ts_b, phi_ts_r, phi_ts_g = np.repeat(phi_ts[f], 3)

            # blue channel --- we don't need sub_phi_ts[0] since it was displayed last time
            lum = mask_ssb * 127 * (1 + mask_ss * c * np.sin(phi_ss + phi_ts_b + phi_i))
            data[:, :, 2] = lum[:, :]
            # red channel
            lum = mask_ssb * 127 * (1 + mask_ss * c * np.sin(phi_ss + phi_ts_r + phi_i))
            data[:, :, 0] = lum[:, :]
            # green channel
            lum = mask_ssb * 127 * (1 + mask_ss * c * np.sin(phi_ss + phi_ts_g + phi_i))
            data[:, :, 1] = lum[:, :]
            # add the dots
            if dots:  # if we are rendering dots
                self.relposs = dots.coords - self.window.pos[:, newaxis] - (f * np.array(dot_speed))[:, newaxis]
                self.dists = np.sqrt((self.relposs ** 2).np.sum(0))
                self.azs = np.arctan2(self.relposs[2], self.relposs[0])
                self.els = np.arctan2(self.relposs[1], self.dists)
                for i in range(len(self.dists)):  # grab them one by one
                    # if self.dists[i] < self.window.far: #if this one is in range
                    if self.dists[i] < 2:  # if this one is in range
                        # if (-np.pi/4 < self.els[i] <= np.pi/4) and not (-3*np.pi/4 < self.azs[i] <= np.pi/4):
                        if (-np.pi / 4 < self.els[i] <= np.pi / 4):
                            y = int((tan(self.els[i]) + 1) / 2 * self.ht)
                            if self.vp == 3 and -np.pi / 4 < self.azs[i] <= np.pi / 4:
                                x = int((tan(self.azs[i]) + 1) / 2 * self.wd)
                                if mask_ssb[y, x] < .1:
                                    try:
                                        data[y - 1:y + 1, x - 1:x + 1, :] = 255
                                    except:
                                        data[y, x, :] = 255
                            elif self.vp == 0 and -3 * np.pi / 4 < self.azs[i] <= -np.pi / 4:
                                x = int((tan(self.azs[i] - np.pi / 2) + 1) / 2 * self.wd)
                                if mask_ssb[y, x] < .1:
                                    try:
                                        data[y - 1:y + 1, x - 1:x + 1, :] = 255
                                    except:
                                        data[y, x, :] = 255
                            elif self.vp == 1 and (
                                    3 * np.pi / 4 < self.azs[i] <= np.pi or -np.pi < self.azs[i] <= -3 * np.pi / 4):
                                x = int((tan(self.azs[i] - np.pi) + 1) / 2 * self.wd)
                                if mask_ssb[y, x] < .1:
                                    try:
                                        data[y - 1:y + 1, x - 1:x + 1, :] = 255
                                    except:
                                        data[y, x, :] = 255

            # make each frame and append the image to the frames list
            frames.append(pyglet.image.ImageData(self.wd, self.ht, 'RGB', data.tostring()))

        # make the animation with all the frames and append it to the list of playable, moving gratings
        self.gratings.append(pyglet.image.Animation.from_image_sequence(frames, 1. / self.rate))

    def add_radial(self, sf=1., tf=1., c=1., phi_i=0, sd=None, sdb=None, maxframes=500):
        '''Add a circular radiating grating'''
        frames = []
        data = np.zeros((self.ht, self.wd, 3), dtype='ubyte')

        # gaussian mask
        if sd:
            mask_ss = scipy.stats.norm.pdf(self.center_dists, 0, sd)
            mask_ss /= mask_ss.max()  # normalize
        else:
            mask_ss = np.ones([self.ht, self.wd])

        if sdb:
            mask_ssb = scipy.stats.norm.pdf(self.center_dists, 0, sdb)
            mask_ssb /= mask_ssb.max()  # normalize
        else:
            mask_ssb = np.ones([self.ht, self.wd])

        # spatial
        # phi_ss = 2*np.pi*sf*np.cos(self.atans + (o-np.pi/2))*self.center_dists
        phi_ss = 2 * np.pi * sf * self.center_dists

        # temporal
        if hasattr(tf, '__iter__'):
            tf_np.array = np.array(tf)  # is it changing?
        else:
            tf_np.array = np.repeat(tf, min(abs(self.rate / tf), maxframes))  # or constant?

        nframes = len(tf_np.array)
        phi_ts = np.cumsum(-2 * np.pi * tf_np.array / float(self.rate))

        for f in np.arange(nframes):
            if self.fast:  # different frames in each color (projected without color wheel)
                if f == 0:
                    prev_phi_t = 0  # first frame?
                else:
                    prev_phi_t = phi_ts[f - 1]
                phi_ts_b, phi_ts_r, phi_ts_g = np.linspace(prev_phi_t, phi_ts[f], 4)[1:]
            else:  # or grays
                phi_ts_b, phi_ts_r, phi_ts_g = np.repeat(phi_ts[f], 3)

            # blue channel --- we don't need sub_phi_ts[0] since it was displayed last time
            lum = mask_ssb * 127 * (1 + mask_ss * c * np.sin(phi_ss + phi_ts_b + phi_i))
            data[:, :, 2] = lum[:, :]
            # red channel
            lum = mask_ssb * 127 * (1 + mask_ss * c * np.sin(phi_ss + phi_ts_r + phi_i))
            data[:, :, 0] = lum[:, :]
            # green channel
            lum = mask_ssb * 127 * (1 + mask_ss * c * np.sin(phi_ss + phi_ts_g + phi_i))
            data[:, :, 1] = lum[:, :]
            # make each frame and append the image to the frames list
            frames.append(pyglet.image.ImageData(self.wd, self.ht, 'RGB', data.tostring()))

        # make the animation with all the frames and append it to the list of playable, moving gratings
        self.gratings.append(pyglet.image.Animation.from_image_sequence(frames, 1. / self.rate))

    def add_plaid(self, sf1=.1, tf1=1, c1=1, o1=0, phi_i1=0.,
                  sf2=.1, tf2=1, c2=1, o2=0, phi_i2=0., sd=None, sdb=None, maxframes=500):
        frames = []
        data = np.zeros((self.ht, self.wd, 3), dtype='byte')

        # gaussian mask
        if sd:
            mask_ss = scipy.stats.norm.pdf(self.center_dists, 0, sd)
            mask_ss /= mask_ss.max()  # normalize
        else:
            mask_ss = np.ones([self.ht, self.wd])

        if sdb:
            mask_ssb = scipy.stats.norm.pdf(self.center_dists, 0, sdb)
            mask_ssb /= mask_ssb.max()  # normalize
        else:
            mask_ssb = np.ones([self.ht, self.wd])

        # spatial
        phi_ss1 = 2 * np.pi * sf1 * np.cos(self.atans + (o1 - np.pi / 2)) * self.center_dists
        phi_ss2 = 2 * np.pi * sf2 * np.cos(self.atans + (o2 - np.pi / 2)) * self.center_dists

        # temporal
        if hasattr(tf1, '__iter__'):
            lentf1 = len(tf1)
        else:
            lentf1 = self.rate / tf1
        if hasattr(tf2, '__iter__'):
            lentf2 = len(tf2)
        else:
            lentf2 = self.rate / tf2
        if hasattr(tf1, '__iter__'):
            tf1_np.array = np.array(tf1)  # is it changing?
        else:
            tf1_np.array = np.repeat(tf1, min(max(lentf1, lentf2), maxframes))  # or constant?
        if hasattr(tf2, '__iter__'):
            tf2_np.array = np.array(tf2)  # is it changing?
        else:
            tf2_np.array = np.repeat(tf2, min(max(lentf1, lentf2), maxframes))  # or constant?

        nframes = max(len(tf1_np.array), len(tf2_np.array))
        phi_t1s = np.cumsum(-2 * np.pi * tf1_np.array / float(self.rate))
        phi_t2s = np.cumsum(-2 * np.pi * tf2_np.array / float(self.rate))

        for f in np.arange(nframes):
            if self.fast:  # different frames in each color (projected without color wheel)
                if f == 0:
                    prev_phi_t1 = prev_phi_t2 = 0  # first frame?
                else:
                    prev_phi_t1 = phi_t1s[f - 1]
                    prev_phi_t2 = phi_t2s[f - 1]
                phi_t1s_b, phi_t1s_r, phi_t1s_g = np.linspace(prev_phi_t1, phi_t1s[f], 4)[1:]
                phi_t2s_b, phi_t2s_r, phi_t2s_g = np.linspace(prev_phi_t2, phi_t2s[f], 4)[1:]
            else:  # or grays
                phi_t1s_b, phi_t1s_r, phi_t1s_g = np.repeat(phi_t1s[f], 3)
                phi_t2s_b, phi_t2s_r, phi_t2s_g = np.repeat(phi_t2s[f], 3)

            # blue channel --- we don't need sub_phi_ts[0] since it was displayed last time
            lum = mask_ssb * 63 * (2 + mask_ss * (c1 * np.sin(phi_ss1 + phi_t1s_b + phi_i1) + \
                                                  c2 * np.sin(phi_ss2 + phi_t2s_b + phi_i2)))
            data[:, :, 2] = lum[:, :]
            # red channel
            lum = mask_ssb * 63 * (2 + mask_ss * (c1 * np.sin(phi_ss1 + phi_t1s_r + phi_i1) + \
                                                  c2 * np.sin(phi_ss2 + phi_t2s_r + phi_i2)))
            data[:, :, 0] = lum[:, :]
            # green channel
            lum = mask_ssb * 63 * (2 + mask_ss * (c1 * np.sin(phi_ss1 + phi_t1s_g + phi_i1) + \
                                                  c2 * np.sin(phi_ss2 + phi_t2s_g + phi_i2)))
            data[:, :, 1] = lum[:, :]

            # make each frame and append the image to the frames list
            frames.append(pyglet.image.ImageData(self.wd, self.ht, 'RGB', data.tostring()))

        # make the animation with all the frames and append it to the list of playable, moving gratings
        self.gratings.append(pyglet.image.Animation.from_image_sequence(frames, 1. / self.rate))

    def choose_grating(self, num):
        '''Assign one of the already generated gratings as the current
        display grating'''
        self.image = self.gratings[num]


class Bar_array(Sprite):
    '''Moving np.array of bars in a single window.  Fast means put different
    values in the blue, red and green sequential channels.

    '''

    def __init__(self, window, vp=0, rate=120., add=False, fast=False):
        Sprite.__init__(self, window, vp=0)

        # self.wd, self.ht = self.window.vpcoords[0,2], self.window.vpcoords[0,3]
        self.indices = indices((self.ht, self.wd))
        # self.sr_indices = self.indices.copy()
        # self.sr_indices[0] *= np.pi/2./self.ht
        # self.sr_indices[1] *= np.pi/2./self.wd
        h, v = np.meshgrid(np.linspace(-np.pi / 4, np.pi / 4, self.wd), np.linspace(-np.pi / 4, np.pi / 4, self.ht))
        self.center_dists = np.sqrt(h ** 2 + v ** 2)
        self.atans = np.arctan2(h, v)

        self.rate = rate
        self.fast = fast
        self.animations = []

        if add: self.add()

    def in_bar(self, testpts, x0, y0, wd, ht, ang):
        # points for the lines, crossing the middle of the bar, at zero, no rot
        zpts = np.array([[-wd, wd, 0, 0],
                         [0, 0, -ht, ht],
                         [1, 1, 1, 1.]])
        # rotate and translate
        cosang, sinang = np.cos(ang), np.sin(ang)
        tmat = np.array([[cosang, sinang, x0],
                         [-sinang, cosang, y0],
                         [0, 0, 1.]])
        pts = np.dot(tmat, zpts)
        # assign vars
        x1a, x1b, x2a, x2b, y1a, y1b, y2a, y2b = pts[:2].flatten()
        xc, yc = testpts
        # distances to each line, a and b are line endpoints, c is the pt
        dln1 = abs((x1b - x1a) * (y1a - yc) - (x1a - xc) * (y1b - y1a)) / np.sqrt((x1b - x1a) ** 2 + (y1b - y1a) ** 2)
        dln2 = abs((x2b - x2a) * (y2a - yc) - (x2a - xc) * (y2b - y2a)) / np.sqrt((x2b - x2a) ** 2 + (y2b - y2a) ** 2)
        # a test point that is less than width to the vertical line
        # and less than height to the horizontal line (before rotation)
        # is in the bar
        return logical_and(dln1 < wd, dln2 < ht)

    def add_bars(self, wd, ht, dist, ori=0, vel=[0, .1], color=255,
                 staggered=False, sd=None, num_frames=None):
        frames = []
        pnt_wd = wd / (np.pi / 2) * self.wd
        pnt_ht = ht / (np.pi / 2) * self.ht

        # gaussian mask
        if sd:
            mask_ss = scipy.stats.norm.pdf(self.center_dists, 0, sd)
            mask_ss /= mask_ss.max()  # normalize
        else:
            mask_ss = np.ones([self.wd, self.ht])

        # temporal
        if hasattr(vel[0], '__iter__'):  # is it changing?
            vel_np.array = np.array([vel[0, :], vel[1, :]])
            if num_frames == None: num_frames = len(vel[0])
        else:  # or constant?
            vel_np.array = np.array([np.repeat(vel[0], num_frames), np.repeat(vel[1], num_frames)])

        dx, dy = 0., 0.
        for f in np.arange(num_frames):
            data = np.zeros((self.ht, self.wd, 3), dtype='ubyte')

            if self.fast:  # different frames in each color (projected without color wheel)
                for cframe in [2, 0, 1]:
                    dx += vel_np.array[0, f] / 3.
                    dy += vel_np.array[1, f] / 3.
                    for xa in np.arange(-np.pi / 4 + mod(dx, dist * 2), np.pi / 4, dist * 2):  # place each bar
                        pnt_xa = (xa + np.pi / 4) / (np.pi / 2) * self.wd
                        for ya in np.arange(-np.pi / 4 + mod(dy, dist), np.pi / 4, dist):
                            pnt_ya = (ya + np.pi / 4) / (np.pi / 2) * self.ht
                            data[self.in_bar(self.indices, pnt_xa, pnt_ya, pnt_wd, pnt_ht, ori), cframe] = color
                    for xa in np.arange(-np.pi / 4 + mod(dx, dist * 2) + dist, np.pi / 4, dist * 2):  # place each bar
                        pnt_xa = (xa + np.pi / 4) / (np.pi / 2) * self.wd
                        if staggered:
                            offset = dist / 2
                        else:
                            offset = 0
                        for ya in np.arange(-np.pi / 4 + mod(dy, dist) + offset, np.pi / 4, dist):
                            pnt_ya = (ya + np.pi / 4) / (np.pi / 2) * self.ht
                            data[self.in_bar(self.indices, pnt_xa, pnt_ya, pnt_wd, pnt_ht, ori), cframe] = color



            else:  # or grays
                dx += vel_np.array[0, f]
                dy += vel_np.array[1, f]
                for xa in np.arange(-np.pi / 4 + mod(dx, dist), np.pi / 4, dist):  # place each bar
                    for ya in np.arange(-np.pi / 4 + mod(dy, dist), np.pi / 4, dist):
                        # data[:,:,:] = self.in_bar(self.sr_indices, xa, ya, wd, ht, ori)[:,:,newaxis]*color*mask_ss[:,:,newaxis]
                        data[self.in_bar(self.indices, pnt_xa, pnt_ya, pnt_wd, pnt_ht, ori), :] = color
                        # data *= mask_ss*color
            lum = data[:, :, :] * mask_ss[:, :, newaxis]
            data[:, :, :] = lum[:, :, :]

            # make each frame and append the image to the frames list
            frames.append(pyglet.image.ImageData(self.wd, self.ht, 'RGB', data.tostring()))

        # make the animation with all the frames and append it to the list of playable, moving gratings
        self.animations.append(pyglet.image.Animation.from_image_sequence(frames, 1. / self.rate))

    def choose_animation(self, num):
        '''Assign one of the already generated bar stims as the current
        display animation.'''
        self.image = self.animations[num]


class kinetogram_class(Sprite):
    '''Moving dots with some degree of coherence.  Fast means put different
    values in the blue, red and green sequential channels.

    '''

    def __init__(self, window, vp=0, rate=120., add=False, fast=False):
        Sprite.__init__(self, window, vp)

        # self.wd, self.ht = self.window.vpcoords[0,2], self.window.vpcoords[0,3]
        self.indices = indices((self.wd, self.ht))
        # self.sr_indices = self.indices.copy()
        # self.sr_indices[0] *= np.pi/2./self.wd
        # self.sr_indices[1] *= np.pi/2./self.ht
        h, v = np.meshgrid(np.linspace(-np.pi / 4, np.pi / 4, self.wd), np.linspace(-np.pi / 4, np.pi / 4, self.ht))
        self.center_dists = np.sqrt(h ** 2 + v ** 2)
        self.atans = np.arctan2(h, v)

        self.rate = rate
        self.fast = fast
        self.animations = []

        if add: self.add()

    def add_kinetogram(self, radius, x, y, density=10, coherence=.5, sd=None,
                       duration=3, dotsize=2, velocity=[1., 0.],
                       dot_color=1, bg_color=0, num_frames=100):
        velocity = np.array(velocity)
        speed = norm(velocity)
        wd, ht = self.wd, self.ht
        # gaussian mask
        if sd:
            mask_ss = scipy.stats.norm.pdf(self.center_dists, 0, sd)
            mask_ss /= mask_ss.max()  # normalize
        else:
            mask_ss = np.ones([wd, ht])

        # density dots/sr * 4*np.pi sr/sphere * 1 sphere/6 surface = density*2*np.pi/3 dots/surface
        numtot = 2 * np.pi * density / 3.
        numcoh, numran = int(round(coherence * numtot)), int(round((1 - coherence) * numtot))
        cohpts, ci = np.random.rand(numcoh, 2) * np.array([wd, ht]), np.random.randint(0, duration, (numcoh))
        ranpts, ri = np.random.rand(numran, 2) * np.array([wd, ht]), np.random.randint(0, duration, (numran))
        ranspd, ranang = np.random.normal(speed, speed / 10., (numran)), np.random.uniform(0, 2 * np.pi, (numran))
        ranvel = ranspd.repeat(2).reshape((numran, 2))
        ranvel[:, 0] *= np.cos(ranang)
        ranvel[:, 1] *= np.sin(ranang)

        lum = np.zeros((wd, ht, num_frames * 3), dtype='ubyte') + bg_color
        for i in np.arange(num_frames * 3):
            # move all the coherently moving pts by a given velocity
            cohpts += velocity
            # a fraction have expired, and are moved to a random location
            ci = mod(ci + 1, duration)
            cohpts[ci == 0] = np.random.rand(len(cohpts[ci == 0]), 2) * np.array([wd, ht])
            # wrap points over the edges
            mod(cohpts, [wd, ht], cohpts)
            # fill in the points of the frame
            lum[np.array(cohpts[:, 0], dtype='ubyte'), np.array(cohpts[:, 1], dtype='ubyte'), i] = dot_color

            ranpts += ranvel
            ri = mod(ri + 1, duration)
            ranpts[ri == 0] = np.random.rand(len(ranpts[ri == 0]), 2) * np.array([wd, ht])
            ranspd, ranang = np.random.normal(speed, speed / 10., (len(ri) - count_nonzero(ri))), np.random.uniform(0,
                                                                                                                    2 * np.pi,
                                                                                                                    (
                                                                                                                                len(ri) - count_nonzero(
                                                                                                                            ri)))
            ranvel[ri == 0, 0] = ranspd * np.cos(ranang)
            ranvel[ri == 0, 1] = ranspd * np.sin(ranang)
            mod(ranpts, [wd, ht], ranpts)
            # fill in the points of the frame
            lum[np.array(ranpts[:, 0], dtype='ubyte'), np.array(ranpts[:, 1], dtype='ubyte'), i] = dot_color

        # make a list of images
        if self.fast:
            frames = [
                pyglet.image.ImageData(wd, ht, 'RGB', lum[:, :, (f + 2, f, f + 1)].transpose([1, 0, 2]).tostring()) for
                f in np.arange(0, num_frames * 3, 3)]
        else:
            frames = [pyglet.image.ImageData(wd, ht, 'RGB', lum[:, :, (f, f, f)].transpose([1, 0, 2]).tostring()) for f
                      in np.arange(0, num_frames * 3, 3)]

        # make the animation with all the frames and append it to the list of playable, moving gratings
        self.animations.append(pyglet.image.Animation.from_image_sequence(frames, 1. / self.rate))

    def add_kinetogram2(self, radius, x, y, density=10, coherence=.5, sd=None,
                        duration=3, dotsize=2, velocity=[1., 0.],
                        dot_color=1, bg_color=0, num_frames=100):
        velocity = np.array(velocity)
        speed = norm(velocity)
        wd, ht = self.wd, self.ht

        # gaussian mask
        if sd:
            mask_ss = scipy.stats.norm.pdf(self.center_dists, 0, sd)
        else:
            mask_ss = np.ones([wd, ht])
        mask_ss /= mask_ss.max()  # normalize

        # from the density argument, dots/sr, get dots to display on surface, dots/surface 
        # density dots/sr * 4*np.pi sr/sphere * 1 sphere/6 surface = density*2*np.pi/3 dots/surface
        numtot = int(2 * np.pi * density / 3.)
        numcoh = int(round(coherence * numtot))
        numran = numtot - numcoh

        # make the points, and put them in random states of duration
        pts = np.random.rand(numtot, 2) * np.array([wd, ht])
        durs = np.random.randint(0, duration, (numtot))
        spds = np.random.normal(speed, speed / 10., (numtot))
        lum = np.zeros((wd, ht, num_frames * 3), dtype='ubyte') + bg_color

        cohpts, ci = np.random.rand(numcoh, 2) * np.array([wd, ht]),
        ranpts, ri = np.random.rand(numran, 2) * np.array([wd, ht]), np.random.randint(0, duration, (numran))
        ranspd, ranang = np.random.normal(speed, speed / 10., (numran)), np.random.uniform(0, 2 * np.pi, (numran))
        ranvel = ranspd.repeat(2).reshape((numran, 2))
        ranvel[:, 0] *= np.cos(ranang)
        ranvel[:, 1] *= np.sin(ranang)

        for i in np.arange(num_frames * 3):
            # move all the coherently moving pts by a given velocity
            cohpts += velocity
            # a fraction have expired, and are moved to a random location
            ci = mod(ci + 1, duration)
            cohpts[ci == 0] = np.random.rand(len(cohpts[ci == 0]), 2) * np.array([wd, ht])
            # wrap points over the edges
            mod(cohpts, [wd, ht], cohpts)
            # fill in the points of the frame
            lum[np.array(cohpts[:, 0], dtype='ubyte'), np.array(cohpts[:, 1], dtype='ubyte'), i] = dot_color

            ranpts += ranvel
            ri = mod(ri + 1, duration)
            ranpts[ri == 0] = np.random.rand(len(ranpts[ri == 0]), 2) * np.array([wd, ht])
            ranspd, ranang = np.random.normal(speed, speed / 10., (len(ri) - count_nonzero(ri))), np.random.uniform(0,
                                                                                                                    2 * np.pi,
                                                                                                                    (
                                                                                                                                len(ri) - count_nonzero(
                                                                                                                            ri)))
            ranvel[ri == 0, 0] *= np.cos(ranang)
            ranvel[ri == 0, 1] *= np.sin(ranang)
            mod(ranpts, [wd, ht], ranpts)
            # fill in the points of the frame
            lum[np.array(ranpts[:, 0], dtype='ubyte'), np.array(ranpts[:, 1], dtype='ubyte'), i] = dot_color

        # make a list of images
        if self.fast:
            frames = [pyglet.image.ImageData(wd, ht, 'RGB', lum[:, :, (f + 2, f, f + 1)].tostring()) for f in
                      np.arange(0, num_frames * 3, 3)]
        else:
            frames = [pyglet.image.ImageData(wd, ht, 'RGB', lum[:, :, (f, f, f)].tostring()) for f in
                      np.arange(0, num_frames * 3, 3)]

        # make the animation with all the frames and append it to the list of playable, moving gratings
        self.animations.append(pyglet.image.Animation.from_image_sequence(frames, 1. / self.rate))

    def choose_animation(self, num):
        '''Assign one of the already generated bar stims as the current
        display animation.'''
        self.image = self.animations[num]


class Movable_grating(Movable):
    '''A grating that can appear anywhere in perspective projection

    '''

    def __init__(self, window, coords, rate=120., xres=64, yres=64, fast=True,
                 sf=.1, tf=1, c=1, o=0, phi_i=0., sd=None, maxframes=500,
                 add=False):
        super(Movable_grating, self).__init__(window)
        # self.gl_type = GL_POLYGON
        self.gl_type = GL_QUADS
        self.coords = np.array(coords)
        self.num = self.coords.shape[1]
        self.txtcoords = np.array([[0, 1, 1, 0], [0, 0, 1, 1.]])
        self.colors = np.zeros((4, self.num), dtype='byte') + 255
        self.colors[3, :] = 255

        self.xres, self.yres = xres, yres

        # self.wd, self.ht = self.window.vpcoords[0,2], self.window.vpcoords[0,3]
        self.indices = np.indices((self.xres, self.yres))
        h, v = np.meshgrid(np.linspace(-np.pi / 4, np.pi / 4, self.xres), np.linspace(-np.pi / 4, np.pi / 4, self.yres))
        self.center_dists = np.sqrt(h ** 2 + v ** 2)
        self.atans = np.arctan2(h, v)

        self.rate = rate
        self.fast = fast
        self.gratings = []
        self.add_grating(sf, tf, c, o, phi_i, sd, maxframes)
        if add: self.add()
        self.frame_ind = 0

    def next_frame(self, inc=1):
        self.frame_ind = np.mod(self.frame_ind + inc, self.num_frames)

    def add_grating(self, sf=.3, tf=1, c=1, o=0, phi_i=0., sd=None, maxframes=500):
        frames = []
        # data = np.zeros((self.xres, self.yres, 3), dtype='ubyte')
        data = np.zeros((self.xres, self.yres, 4), dtype='ubyte')  # include the alpha channel

        # gaussian mask
        if sd:
            mask_ss = scipy.stats.norm.pdf(self.center_dists, 0, sd)
            mask_ss /= mask_ss.max()  # normalize
        else:
            mask_ss = np.ones([self.xres, self.yres])

        # spatial
        phi_ss = 2 * np.pi * sf * np.cos(self.atans + (o - np.pi / 2)) * self.center_dists

        # temporal
        if hasattr(tf, '__iter__'):
            tf_np.array = np.array(tf)  # is it changing?
        else:
            tf_array = np.repeat(tf, min(abs(self.rate / tf), maxframes))  # or constant?

        nframes = len(tf_array)
        self.num_frames = nframes
        phi_ts = np.cumsum(-2 * np.pi * tf_array / float(self.rate))

        for f in np.arange(nframes):
            if self.fast:  # different frames in each color (projected without color wheel)
                if f == 0:
                    prev_phi_t = 0  # first frame?
                else:
                    prev_phi_t = phi_ts[f - 1]
                phi_ts_b, phi_ts_r, phi_ts_g = np.linspace(prev_phi_t, phi_ts[f], 4)[1:]
            else:  # or grays
                phi_ts_b, phi_ts_r, phi_ts_g = np.repeat(phi_ts[f], 3)

            # blue channel --- we don't need sub_phi_ts[0] since it was displayed last time
            lum = 127 * (1 + c * np.sin(phi_ss + phi_ts_b + phi_i))
            data[:, :, 2] = lum[:, :]
            # red channel
            lum = 127 * (1 + c * np.sin(phi_ss + phi_ts_r + phi_i))
            data[:, :, 0] = lum[:, :]
            # green channel
            lum = 127 * (1 + c * np.sin(phi_ss + phi_ts_g + phi_i))
            data[:, :, 1] = lum[:, :]
            data[:, :, 3] = mask_ss * 255
            # data[:,:,3] = 0
            # make each frame and append the image to the frames list
            frames.append(pyglet.image.ImageData(self.yres, self.xres, 'RGBA', data.tostring()))

        # make the animation with all the frames and append it to the list of playable, moving gratings
        self.ani = pyglet.image.Animation.from_image_sequence(frames, 1. / self.rate)
        self.tbin = pyglet.image.atlas.TextureBin()
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        # self.ani.add_to_texture_bin(self.tbin)
        # self.texture_id = self.tbin.atlases[0].texture.id
        # self.gratings.append( pyglet.image.Animation.from_image_sequence(frames, 1./self.rate) )

    def add(self):
        if self.txtcoords is None: self.txtcoords = np.zeros((2, self.num), dtype='float')
        self.vl = self.window.world.add(self.num, self.gl_type, self,
                                        ('v3f', self.coords.T.flatten()),
                                        ('c4B/stream', self.colors.T.flatten()),
                                        ('t2f', self.txtcoords.T.flatten()))
        self.visible = True

    def set_state(self):
        super(Movable_grating, self).set_state()
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBindTexture(GL_TEXTURE_2D, self.ani.frames[self.frame_ind].image.get_texture().id)

    def unset_state(self):
        super(Movable_grating, self).unset_state()
        glDisable(GL_TEXTURE_2D)  # skipping this causes all the other objects to oscillate with the texture!
        glDisable(GL_BLEND)


class Movable_kinetogram(Movable):
    '''A grating that can appear anywhere in perspective projection

    '''

    def __init__(self, window, coords,  # coords of the display square in 3d
                 rate=120., xres=64, yres=64, fast=True,
                 density=10, coherence=.5, duration=3, dotsize=2, velocity=[1., 0.],
                 dot_color=1, bg_color=0, num_frames=100, sd=None,
                 add=False):
        super(Movable_kinetogram, self).__init__(window)

        # self.gl_type = GL_POLYGON
        self.gl_type = GL_QUADS
        self.coords = np.array(coords)
        self.num = self.coords.shape[1]
        self.txtcoords = np.array([[0, 1, 1, 0], [0, 0, 1, 1.]])
        self.colors = np.zeros((4, self.num), dtype='byte') + 255
        self.colors[3, :] = 255

        self.xres, self.yres = xres, yres

        # self.wd, self.ht = self.window.vpcoords[0,2], self.window.vpcoords[0,3]
        self.indices = indices((self.xres, self.yres))
        h, v = np.meshgrid(np.linspace(-np.pi / 4, np.pi / 4, self.xres), np.linspace(-np.pi / 4, np.pi / 4, self.yres))
        self.center_dists = np.sqrt(h ** 2 + v ** 2)
        self.atans = np.arctan2(h, v)

        self.rate = rate
        self.fast = fast
        self.num_frames = num_frames
        self.add_kinetogram(density, coherence, duration, dotsize, velocity,
                            dot_color, bg_color, num_frames, sd)
        if add: self.add()
        self.frame_ind = 0

    def next_frame(self, inc=1):
        self.frame_ind = mod(self.frame_ind + inc, self.num_frames)

    def add_kinetogram(self, density=10, coherence=.5, duration=3, dotsize=2, velocity=[1., 0.],
                       dot_color=1, bg_color=0, num_frames=100, sd=None):
        num = 3 if self.fast else 1
        frames = []
        lum = np.zeros((self.xres, self.yres, num))
        data = np.zeros((self.xres, self.yres, 4), dtype='ubyte')  # include the alpha channel

        # gaussian mask
        if sd:
            mask_ss = scipy.stats.norm.pdf(self.center_dists, 0, sd)
            mask_ss /= mask_ss.max()  # normalize
        else:
            mask_ss = np.ones([self.xres, self.yres])

        velocity = np.array(velocity)
        x, y = velocity
        speed = norm(velocity)

        # from the density argument, dots/sr, get dots to display on surface, dots/surface 
        # density dots/sr * 4*np.pi sr/sphere * 1 sphere/6 surface = density*2*np.pi/3 dots/surface
        numtot = int(2 * np.pi * density / 3.)
        numcoh = int(round(coherence * numtot))
        numran = numtot - numcoh

        # make velocities for all the non-coherent pts, distributed equally around the coherent velocities
        ranvels = np.array([[x * np.cos(a) - y * np.sin(a), x * np.sin(a) + y * np.cos(a)] for a in
                            np.linspace(0, 2 * np.pi, numran + 2)[1:-1]])
        # make the velocities for adding
        velocities = np.zeros((2, numtot))
        velocities[:, :numran] = ranvels.T
        velocities[:, numran:] = velocity[:, None]

        # how many frames to make multiple of duration
        nframes = int(ceil(num_frames / duration)) * duration
        # x and y pts with a time index
        pts = np.zeros((2, numtot, nframes))
        # fill in the random skips (all align at first)
        pts[0, :, ::duration] = np.random.uniform(0, self.xres, (numtot, nframes // duration))
        pts[1, :, ::duration] = np.random.uniform(0, self.yres, (numtot, nframes // duration))
        # now add the motions, after each random skip
        for i in range(1, duration):
            pts[:, :, i::duration] = pts[:, :, 0::duration] + i * velocities[:, :, None]
        pts[0] %= self.xres  # wrap out of bound values
        pts[1] %= self.yres  #
        # now shift the columns, every in turn by one more, resseting at dot duration, so pt skips don't align
        for i in range(numtot):
            pts[:, i, :] = roll(pts[:, i, :], i % duration, 1)

        for f in np.arange(num_frames):
            # reset the frame to background color
            lum[:, :, :] = bg_color
            # fill in the points with dot_color
            for ff in range(num):
                lum[pts[0, :, f].astype('int'), pts[1, :, f].astype('int'), ff] = int(dot_color)
            # put the frames in the right order for my projectors, 2, 0, 1 (blue, red, green)
            data[:, :, :3] = roll(lum * 255, 1, 2)  # and muliply by 255
            # add the mask
            data[:, :, 3] = mask_ss * 255
            # append to the animation
            frames.append(pyglet.image.ImageData(self.yres, self.xres, 'RGBA', data.tostring()))
        # make the animation with all the frames and append it to the list of playable, moving kinetograms
        self.ani = pyglet.image.Animation.from_image_sequence(frames, 1. / self.rate)
        self.tbin = pyglet.image.atlas.TextureBin()
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)

    def add(self):
        if self.txtcoords is None: self.txtcoords = np.zeros((2, self.num), dtype='float')
        self.vl = self.window.world.add(self.num, self.gl_type, self,
                                        ('v3f', self.coords.T.flatten()),
                                        ('c4B/stream', self.colors.T.flatten()),
                                        ('t2f', self.txtcoords.T.flatten()))
        self.visible = True

    def set_state(self):
        super(Movable_kinetogram, self).set_state()
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBindTexture(GL_TEXTURE_2D, self.ani.frames[self.frame_ind].image.get_texture().id)

    def unset_state(self):
        super(Movable_kinetogram, self).unset_state()
        glDisable(GL_TEXTURE_2D)  # skipping this causes all the other objects to oscillate with the texture!
        glDisable(GL_BLEND)


class Quad_image(Movable):
    '''Makes a quad mesh and maps an image onto it---possibly
    animated. Images seem to have to have resolutions that are a power of
    two
    
    '''

    def __init__(self, window, rate=120., xres=64, yres=64, fast=True,
                 dist=1., left=-np.pi, right=np.pi, bottom=-np.pi,
                 top=np.pi, xdivs=10, ydivs=10, image=None, add=False):

        super(Quad_image, self).__init__(window)
        self.gl_type = GL_QUADS
        self.num = 4 * xdivs * ydivs
        self.colors = np.zeros((4, self.num), dtype='byte') + 255
        self.colors[3, :] = 255
        self.coords = np.zeros([3, self.num])
        self.txtcoords = np.zeros([2, self.num])
        self.xext = right - left
        self.yext = top - bottom
        i = 0
        xangs, yangs = np.linspace(left, right, xdivs + 1), np.linspace(bottom, top, ydivs + 1)
        sin_x, cos_x, sin_y, cos_y = np.sin(xangs), np.cos(xangs), np.sin(yangs), np.cos(yangs)
        xtxts, ytxts = np.linspace(0, 1, xdivs + 1), np.linspace(0, 1, ydivs + 1)
        for xang_ind in range(xdivs):
            for yang_ind in range(ydivs):
                xi, yi = xang_ind, yang_ind
                self.coords[:, i] = [dist * sin_x[xi] * cos_y[yi], dist * sin_y[yi], -dist * cos_x[xi] * cos_y[yi]]
                self.txtcoords[:, i] = [xtxts[xang_ind], ytxts[yang_ind]]
                xi, yi = xang_ind, yang_ind + 1
                self.coords[:, i + 1] = [dist * sin_x[xi] * cos_y[yi], dist * sin_y[yi], -dist * cos_x[xi] * cos_y[yi]]
                self.txtcoords[:, i + 1] = [xtxts[xang_ind], ytxts[yang_ind + 1]]
                xi, yi = xang_ind + 1, yang_ind + 1
                self.coords[:, i + 2] = [dist * sin_x[xi] * cos_y[yi], dist * sin_y[yi], -dist * cos_x[xi] * cos_y[yi]]
                self.txtcoords[:, i + 2] = [xtxts[xang_ind + 1], ytxts[yang_ind + 1]]
                xi, yi = xang_ind + 1, yang_ind
                self.coords[:, i + 3] = [dist * sin_x[xi] * cos_y[yi], dist * sin_y[yi], -dist * cos_x[xi] * cos_y[yi]]
                self.txtcoords[:, i + 3] = [xtxts[xang_ind + 1], ytxts[yang_ind]]
                i += 4
        self.frame_ind = 0
        self.xres, self.yres = xres, yres
        self.data = np.zeros((self.xres, self.yres, 4), dtype='ubyte')  # include the alpha channel
        self.num_frames = 1

    def gradient_image(self, start=0, stop=255):
        self.data[:, :, :3] = np.linspace(start, stop, len(self.data))[None, :, None]
        self.data[:, :, 3] = 255  # alpha channel
        self.image = pyglet.image.ImageData(self.yres, self.xres, 'RGBA', self.data.tostring())

    def sin_image(self, cycles=5, c=1, phi_i=0.):
        self.data[:, :, :3] = 127 + 127 * c * np.sin(phi_i + cycles * np.linspace(0, self.xext, len(self.data)))[None,
                                              :, None]
        self.data[:, :, 3] = 255  # alpha channel
        self.image = pyglet.image.ImageData(self.yres, self.xres, 'RGBA', self.data.tostring())

    def load_image(self, filename):
        self.image = pyglet.image.load(filename)

    def set_image(self, data):
        '''set an image with a xres by yres by 4 data array'''
        self.data[:, :, :] = data
        self.image = pyglet.image.ImageData(self.yres, self.xres, 'RGBA', self.data.tostring())

    def next_frame(self, inc=1):
        self.frame_ind = mod(self.frame_ind + inc, self.num_frames)

    def add(self):
        if self.txtcoords is None: self.txtcoords = np.zeros((2, self.num), dtype='float')
        self.vl = self.window.world.add(self.num, self.gl_type, self,
                                        ('v3f', self.coords.T.flatten()),
                                        ('c4B/stream', self.colors.T.flatten()),
                                        ('t2f', self.txtcoords.T.flatten()))
        self.visible = True

    def set_state(self):
        super(Quad_image, self).set_state()
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBindTexture(GL_TEXTURE_2D, self.image.get_texture().id)

    def unset_state(self):
        super(Quad_image, self).unset_state()
        glDisable(GL_TEXTURE_2D)  # skipping this causes all the other objects to oscillate with the texture!
        glDisable(GL_BLEND)
