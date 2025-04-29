"""
Classes for flexible visual stimuli based on opengl primitives

import stl requires: pip install numpy-stl

"""

# lightspeed design 360 projector displays in the order b r g Frames

import pyglet
from pyglet.gl import *
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.math import Mat4, Vec3
# from pyglet.graphics.vertexdomain import VertexList

import numpy as np
import stl
import numbers
import scipy.stats

# Detect pyglet version
PYGLET_VERSION = pyglet.version
PYGLET_MAJOR = int(PYGLET_VERSION.split('.')[0])

#
# vertex_source_i = """#version 330 core
# layout (location = 0) in vec3 vertices;
# void main() {
#     gl_Position = vec4(vertices, 1.0);
# }"""
#
# fragment_source_i = """#version 330 core
# out vec4 FragColor;
# void main() {
#     FragColor = vec4(1.0);
# }
# """

vertex_source_c = """#version 330 core

layout (location = 0) in vec3 vertices;
layout (location = 1) in vec4 colors;

out vec4 frag_color;

uniform vpBlock {
    uniform mat4 projection;
    uniform mat4 view;
} viewport;

uniform mat4 model;

void main() {
    gl_Position = viewport.projection * viewport.view * model * vec4(vertices, 1.0);
    gl_PointSize = 2.0;
    frag_color = colors;
}
"""

fragment_source_c = """#version 330 core

in vec4 frag_color;

out vec4 FragColor;

void main() {
    FragColor = vec4(frag_color);  // Just use color
}"""

vertex_source_t = """#version 330 core

layout (location = 0) in vec3 vertices;
layout (location = 1) in vec2 tex_coords;

out vec2 TexCoords;

uniform vpBlock {
    uniform mat4 projection;
    uniform mat4 view;
} viewport;

uniform mat4 model;

void main() {
    gl_Position = viewport.projection * viewport.view * model * vec4(vertices, 1.0);
    TexCoords = tex_coords;
}
"""

fragment_source_t = """#version 330 core
in vec2 TexCoords;
out vec4 FragColor;
uniform sampler2D texture1;

void main()
{
    FragColor = texture(texture1, TexCoords);
}
"""

vertex_source_l = """#version 330 core

layout (location = 0) in vec3 vertices;
layout (location = 1) in vec4 colors;
layout (location = 2) in vec3 normals;

out vec4 frag_color;
out vec3 frag_normal;
out vec3 frag_pos;

uniform vpBlock {
    uniform mat4 projection;
    uniform mat4 view;
} viewport;

uniform mat4 model;

void main() {
    vec4 world_pos = model * vec4(vertices, 1.0);
    gl_Position = viewport.projection * viewport.view * world_pos;

    frag_color = colors;

    // Normal transformation (ignore scaling for now)
    frag_normal = mat3(transpose(inverse(model))) * normals;
    frag_pos = world_pos.xyz;
    }
"""

fragment_source_l = """#version 330 core

in vec4 frag_color;
in vec3 frag_normal;
in vec3 frag_pos;

out vec4 FragColor;

uniform vec3 light_pos;
uniform float ambient_strength;  // e.g., 0.2

void main() {
    // Normalize vectors
    vec3 norm = normalize(frag_normal);
    vec3 light_dir = normalize(light_pos - frag_pos);  // Direction to the light

    // Diffuse shading (Lambertian)
    float diff = max(dot(norm, light_dir), 0.0);

    // Ambient and diffuse contributions
    vec3 ambient = ambient_strength * frag_color.rgb;
    vec3 diffuse = diff * frag_color.rgb;

    vec3 final_color = ambient + diffuse;

    FragColor = vec4(final_color, frag_color.a);
}
"""



def cm_soapbubble(x, y, alpha=1.0):
    """soapbubble colors for complex magnitudes and angles.

    x = intensity, lightness, 0 -- 1,

    y = radians

    """
    r = np.clip(-57.4 * np.sin(y + 4.5) + x * 265.1 + -20.3, 0, 255)
    g = np.clip(20.6 * np.sin(y + -1.5) + x * 253.5 + -14.3, 0, 255)
    b = np.clip(57.1 * np.sin(y + 3.2) + x * 234.6 + -4.3, 0, 255)
    if hasattr(r, '__len__'): alpha = np.resize(alpha, len(r))
    return np.stack([r / 255, g / 255, b / 255, alpha])


def rotate_points_to_normal(points, n_target):
    """Reorient 3D points from facing (0,0,1) to a given normal vector."""

    n_target = np.array(n_target) / np.linalg.norm(n_target)  # Normalize target normal
    z_axis = np.array([0, 0, 1])  # Current "up" vector

    if np.allclose(n_target, z_axis):  # If already aligned, return unchanged
        return points
    elif np.allclose(n_target, -z_axis):  # If negatively aligned,
        rotation_axis = np.array([0, 1, 0.])
        theta = np.pi
    else:
        # Compute rotation axis (cross product)
        rotation_axis = np.cross(z_axis, n_target)
        rotation_axis /= np.linalg.norm(rotation_axis)  # Normalize

        # Compute rotation angle (dot product)
        cos_theta = np.dot(z_axis, n_target)  # cos(θ) = dot(v1, v2)
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clamp for numerical stability

    # Rodrigues' rotation formula components
    K = np.array([
        [0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0]
    ])

    I = np.eye(3)  # Identity matrix
    R = I + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)  # Rodrigues' formula

    # Apply rotation matrix to all points
    return R @ points  # Rotates the entire set of points


def azel_to_cart(az, el, radius=1.):
    """Convert azimuth and elevation to cartesion, x,y,z coordinates"""

    az = np.radians(az)
    el = np.radians(el)

    x =  radius * np.cos(el) * np.sin(az)
    z = -radius * np.cos(el) * np.cos(az)
    y =  radius * np.sin(el)

    return np.array([x,y,z])


def cart_to_azel(x, y, z):
    """Convert x,y,z coordinates to azimuth and elevation"""
    az = np.degrees(np.arctan2(-z, x))
    az = -az + 90
    el = np.degrees(np.arctan2(y, np.sqrt(x**2 + z**2)))

    return np.array([az, el])



class Movable(pyglet.graphics.Group):
    """Any opengl object that we can move and rotate outside of
    observer motion. 

    """
    def __init__(self, window, gl_type, verts, vert_inds=None, order=0):
        """
        Set the basic parameters for any movable object for the 3d enironment.

        window: the window in which the object gets displayed, has a batch property

        gl_type: points, lines, or triangles

        verts: all the vertexes, in 3d, in the form np.array([[xs], [ys], [zs]])

        vert_inds: using indexes makes vertex listing less redundant, as there is
          no need to repeat coordinates

        order: if a group needs to appear in front or behind, lower numbers
          render first, so higher numbers appear in front

        """

        self._order = order

        super().__init__(order=order)

        self.window = window
        self.parent = None  # required to use set_state
        self.pos = np.array([0, 0, 0.])
        self.rot = np.array([0, 0, 0.])
        self.ind = 0
        self.poss = np.array([[0, 0, 0.]])

        self.gl_type = gl_type
        self.verts = np.array(verts, dtype='f4')
        self.num = self.verts.shape[1]

        # if we have no vertex list, make a simple one
        if vert_inds is None:
            self.vert_inds = np.arange(self.num)[::-1]
        else:
            self.vert_inds = vert_inds

        self.time = 0.0
        self.dt = 0.0

        self.visible = False

    def remove(self):
        self.vl.delete()
        self.visible = False

    def on(self, onstate):
        if onstate and not self.visible:
            self.add()
        elif not onstate and self.visible:
            self.remove()

    def switch(self, onoff):
        """A synonym for method on, adding string control"""
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

    def set_verts(self, xs, ys, zs):
        """Set the vertex list by a method, so we don't accidentially make it
        a float64."""
        self.verts = np.asarray([xs, ys, zs], dtype='f4')

    def set_pos(self, pos):
        self.pos[:] = pos

    def set_rot(self, rot):
        self.rot[:] = np.radians(rot)

    def set_px(self, x):
        self.pos[0] = x

    def set_py(self, y):
        self.pos[1] = y

    def set_pz(self, z):
        self.pos[2] = z

    def set_rx(self, x):
        self.rot[0] = np.radians(x)

    def set_ry(self, y):
        self.rot[1] = np.radians(y)

    def set_rz(self, z):
        self.rot[2] = np.radians(z)

    def inc_px(self, x=.01):
        self.pos[0] += x

    def inc_py(self, y=.01):
        self.pos[1] += y

    def inc_pz(self, z=.01):
        self.pos[2] += z

    def inc_rx(self, x=np.pi / 180):
        self.rot[0] += np.radians(x)

    def inc_ry(self, y=np.pi / 180):
        self.rot[1] += np.radians(y)

    def inc_rz(self, z=np.pi / 180):
        self.rot[2] += np.radians(z)

    def update_pos_ind(self, dt=0.0):
        np.take(self.poss, [self.ind], out=self.pos, mode='wrap')
        self.ind += 1

    def update_ry_func(self, dt=0.0, func=None):
        self.rot[1] = func()

    def subset_inc_px(self, bool_a, x):
        """move in x direction just some of the vertices"""
        self.verts[0, bool_a] += x
        if self.visible:
            self.vl.vertices[::3] = self.verts[0]

    def subset_inc_py(self, bool_a, y):
        """move in z direction just some of the vertices"""
        self.verts[1, bool_a] += y
        if self.visible:
            self.vl.vertices[1::3] = self.verts[1]

    def subset_inc_pz(self, bool_a, z):
        """move in z direction just some of the vertices"""
        self.verts[2, bool_a] += z
        if self.visible:
            self.vl.vertices[2::3] = self.verts[2]

    def subset_set_px(self, bool_a, x_a):
        """set x coordinate for some vertices """
        self.verts[0, bool_a] = x_a[bool_a]
        if self.visible:
            self.vl.vertices[0::3] = self.verts[0]

    def subset_set_py(self, bool_a, y_a):
        """set y coordinate for some vertices """
        self.verts[1, bool_a] = y_a[bool_a]
        if self.visible:
            self.vl.vertices[1::3] = self.verts[1]

    def subset_set_pz(self, bool_a, z_a):
        """set z coordinate for some vertices """
        self.verts[2, bool_a] = z_a[bool_a]
        if self.visible:
            self.vl.vertices[2::3] = self.verts[2]

    def get_rx(self):
        return np.degrees(self.rot[0])

    def get_ry(self):
        return np.degrees(self.rot[1])

    def get_rz(self):
        return np.degrees(self.rot[2])

    def get_px(self):
        return self.pos[0]

    def get_py(self):
        return self.pos[1]

    def get_pz(self):
        return self.pos[2]

    def update_time(self, dt):
        """Updates the elapsed time, so any time dependent indexes
        progress over time.

        """
        self.dt = dt
        self.time += dt

    def reset_time(self):
        self.time = 0

    def set_state(self):
        glEnable(GL_DEPTH_TEST)
        rot_mat_x = Mat4.from_rotation(angle=self.rot[0], vector=Vec3(1, 0, 0))
        self.rmx = rot_mat_x
        rot_mat_y = Mat4.from_rotation(angle=self.rot[1], vector=Vec3(0, 1, 0))
        self.rmy = rot_mat_y
        rot_mat_z = Mat4.from_rotation(angle=self.rot[2], vector=Vec3(0, 0, 1))
        self.rmz = rot_mat_z
        trans_mat = Mat4.from_translation(Vec3(*self.pos))

        self.shader_program.use()
        self.shader_program["model"] = trans_mat @ rot_mat_z @ rot_mat_y @ rot_mat_x

    def unset_state(self):
        self.shader_program["model"] = Mat4()
        self.shader_program.stop()




class Movable_Color(Movable):
    """Movables that are colored, without a texture.

    """

    def __init__(self, window, gl_type, verts, vert_inds=None,
                 colors=None, add=False):
        super().__init__(window, gl_type, verts, vert_inds)

        self.colors = colors

        vs = vertex_source_c
        fs = fragment_source_c

        self.vert_shader = Shader(vs, 'vertex')
        self.frag_shader = Shader(fs, 'fragment')
        self.shader_program = ShaderProgram(self.vert_shader, self.frag_shader)

        # this is to allow updating all vpblock (view and projection
        # for each viewport) with just one call in windows.py
        block_name = 'vpBlock'
        block_index = glGetUniformBlockIndex(self.shader_program.id, block_name.encode('utf-8'))
        glUniformBlockBinding(self.shader_program.id, block_index, 0)

        if add: self.add()

    def add(self):
        """Use the shader program to add the vertex list to the
        window's batch. This version also requires an index list,
        which will generate automatically if not provided

        """
        self.window.switch_to()

        # if we don't have a proper color array yet, set one
        if not (isinstance(self.colors, np.ndarray) and self.colors.shape == (4, self.num)):
            self.update_colors()

        self.vl = self.shader_program.vertex_list_indexed(
            count=self.num,
            mode=self.gl_type,
            indices=self.vert_inds,
            batch=self.window.batch,
            group=self,
            vertices=('f', self.verts.T.flatten()),
            colors=('f', self.colors.T.flatten())
        )

        self.visible = True


    def update_colors(self, color=None):
        """Take the self.colors attribute and reassign it to an array
        of the length 4 * self.num

        """
        if color is not None:
            self.colors = color

        # do we have a single number? if so, set to gray
        if isinstance(self.colors, str) and self.colors == 'ring':
            angs = np.linspace(0, 2 * np.pi, self.num, endpoint=False)
            self.colors = cm_soapbubble(.5, angs)
        elif isinstance(self.colors, str) and self.colors == 'ring0':
            angs = np.linspace(0, 2 * np.pi, self.num, endpoint=True)
            self.colors = cm_soapbubble(.5, angs)
            self.colors[:, -1] = [0, 0, 0, 1.]
        elif isinstance(self.colors, str) and self.colors == 'ring1':
            angs = np.linspace(0, 2 * np.pi, self.num, endpoint=True)
            self.colors = cm_soapbubble(.5, angs)
            self.colors[:, -1] = [1, 1, 1, 1.]
        elif isinstance(self.colors, numbers.Real):
            self.colors = np.array([[*[self.colors] * 3, 1.0]] * self.num).T
        # if we have a 3 tuple of numbers, 1 color for all verts
        elif hasattr(self.colors, '__len__') and \
                len(self.colors) == 3 and \
                isinstance(self.colors[0], numbers.Real):
            self.colors = np.array([[*self.colors, 1.0]] * self.num)
        # if we have a 4 tuple of numbers, 1 color for all verts
        elif hasattr(self.colors, '__len__') and \
                len(self.colors) == 4 and \
                isinstance(self.colors[0], numbers.Real):
            self.colors = np.array([[*self.colors]] * self.num)
        # if we have list of 3 tuples
        elif hasattr(self.colors, '__len__') and \
                hasattr(self.colors[0], '__len__') and \
                len(self.colors[0]) == 3:
            self.colors = np.array([[*c, 1.0] for c in self.colors]).T
        # if we have list of 4 tuples
        elif hasattr(self.colors, '__len__') and \
                hasattr(self.colors[0], '__len__') and \
                len(self.colors[0]) == 4:
            self.colors = np.array([[*c] for c in self.colors]).T
        # if None or something else, use white
        else:
            self.colors = np.array([[1.0] * self.num] * 4)

    def __eq__(self, other):
        return id(self) == id(other)

    def __hash__(self):
        return hash(id(self))
        # return hash((self._order, self.parent))

    def set_state(self):
        super().set_state()

    def unset_state(self):
        super().unset_state()


class Movable_Texture(Movable):
    """Movables that are textured, not colored.

    """

    def __init__(self, window, gl_type, verts, vert_inds=None,
                 texture=None, tex_coords=None, add=False):
        super().__init__(window, gl_type, verts, vert_inds)

        self.texture = texture

        self.tex_coords = tex_coords
        if self.tex_coords is None:
            self.tex_coords = np.array([[0., 1., 1., 0.], [0., 0., 1., 1.]])

        vs = vertex_source_t
        fs = fragment_source_t
        self.vert_shader = Shader(vs, 'vertex')
        self.frag_shader = Shader(fs, 'fragment')
        self.shader_program = ShaderProgram(self.vert_shader, self.frag_shader)

        # this is to allow updating all vpblock (view and projection
        # for each viewport) with just one call in windows.py
        block_name = 'vpBlock'
        block_index = glGetUniformBlockIndex(self.shader_program.id, block_name.encode('utf-8'))
        glUniformBlockBinding(self.shader_program.id, block_index, 0)

        if add: self.add()

    def add(self):
        """Use the shader program to add the vertex list to the
        window's batch. This version also requires an index list,
        which will generate automatically if not provided

        """
        self.window.switch_to()

        self.vl = self.shader_program.vertex_list_indexed(
            count=self.num,
            mode=self.gl_type,
            indices=self.vert_inds,
            batch=self.window.batch,
            group=self,
            vertices=('f', self.verts.T.flatten()),
            tex_coords=('f', self.tex_coords.T.flatten())
        )

        self.visible = True

    def set_state(self):
        glDepthMask(GL_TRUE)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(self.texture.target, self.texture.id)

        super().set_state()

    def unset_state(self):
        super().unset_state()

    def __hash__(self):
        return hash(id(self))
        # return hash((self.texture.target, self.texture.id, self.order, self.parent))

    def __eq__(self, other):
        return id(self) == id(other)

    def __repr__(self):
        return '%s' % (self.__class__.__name__)


class Movable_Animation(Movable):
    """Movables that are textured with multiple frames.

    """

    def __init__(self, window, gl_type, verts, vert_inds=None,
                 # ani=None,
                 tex_coords=None, add=False):
        super().__init__(window, gl_type, verts, vert_inds)

        self.tex_coords = tex_coords
        if self.tex_coords is None:
            self.tex_coords = np.array([[0., 1., 1., 0.], [0., 0., 1., 1.]], dtype='f4')

        vs = vertex_source_t
        fs = fragment_source_t
        self.vert_shader = Shader(vs, 'vertex')
        self.frag_shader = Shader(fs, 'fragment')
        self.shader_program = ShaderProgram(self.vert_shader, self.frag_shader)

        vs = vertex_source_c
        fs = fragment_source_c
        self.vert_shader_init = Shader(vs, 'vertex')
        self.frag_shader_init = Shader(fs, 'fragment')
        self.dummy_shader = ShaderProgram(self.vert_shader_init, self.frag_shader_init)

        # this is to allow updating all vpblock (view and projection
        # for each viewport) with just one call in windows.py
        block_name = 'vpBlock'
        block_index = glGetUniformBlockIndex(self.shader_program.id, block_name.encode('utf-8'))
        glUniformBlockBinding(self.shader_program.id, block_index, 0)

        if add: self.add()

    def add(self):
        """Use the shader program to add the vertex list to the
        window's batch. This version also requires an index list,
        which will generate automatically if not provided

        """
        self.window.switch_to()

        self.vl = self.shader_program.vertex_list_indexed(
            count=self.num,
            mode=self.gl_type,
            indices=self.vert_inds,
            batch=self.window.batch,
            group=self,
            vertices=('f', self.verts.T.flatten()),
            tex_coords=('f', self.tex_coords.T.flatten())
        )

        self.visible = True

        self.time = 0.0
        pyglet.clock.schedule(self.update_time)

    def set_state(self):
        glActiveTexture(GL_TEXTURE0)

        frame_index = int((self.time * self.rate) % len(self.ani.frames))
        curr_frame = self.ani.frames[frame_index].image.get_texture()

        self.vl.tex_coords[0:2] = curr_frame.tex_coords[0:2]
        self.vl.tex_coords[2:4] = curr_frame.tex_coords[3:5]
        self.vl.tex_coords[4:6] = curr_frame.tex_coords[6:8]
        self.vl.tex_coords[6:8] = curr_frame.tex_coords[9:11]

        glBindTexture(curr_frame.target, curr_frame.id)

        rot_mat_x = Mat4.from_rotation(angle=self.rot[0], vector=Vec3(1, 0, 0))
        self.rmx = rot_mat_x
        rot_mat_y = Mat4.from_rotation(angle=self.rot[1], vector=Vec3(0, 1, 0))
        self.rmy = rot_mat_y
        rot_mat_z = Mat4.from_rotation(angle=self.rot[2], vector=Vec3(0, 0, 1))
        self.rmz = rot_mat_z
        trans_mat = Mat4.from_translation(Vec3(*self.pos))

        self.shader_program.use()
        self.shader_program['texture1'] = 0
        self.shader_program["model"] = trans_mat @ rot_mat_z @ rot_mat_y @ rot_mat_x

    def unset_state(self):
        self.shader_program.stop()

    def __hash__(self):
        return hash(id(self))
        # return hash((self.texture.target, self.texture.id, self.order, self.parent))

    def __eq__(self, other):
        return id(self) == id(other)

    def __repr__(self):
        return f'{self.__class__.__name__=}'



class Movable_Lighted(Movable):
    """A movable object where vertexes have normals, and the environment both
    directional and diffuse lighting.

    """

    def __init__(self, window, gl_type, verts, vert_inds=None,
                 normals=None, colors=None, add=False):
        super().__init__(window, gl_type, verts, vert_inds)

        self.normals = normals
        self.colors = colors

        vs = vertex_source_l
        fs = fragment_source_l

        self.vert_shader = Shader(vs, 'vertex')
        self.frag_shader = Shader(fs, 'fragment')
        self.shader_program = ShaderProgram(self.vert_shader, self.frag_shader)

        # this is to allow updating all vpblock (view and projection
        # for each viewport) with just one call in windows.py
        block_name = 'vpBlock'
        block_index = glGetUniformBlockIndex(self.shader_program.id, block_name.encode('utf-8'))
        glUniformBlockBinding(self.shader_program.id, block_index, 0)

        if add: self.add()

    def add(self):
        """Use the shader program to add the vertex list to the
        window's batch. This version also requires an index list,
        which will generate automatically if not provided

        """
        self.window.switch_to()

        if not (isinstance(self.colors, np.ndarray) and self.colors.shape == (4, self.num)):
            self.set_colors()

        self.vl = self.shader_program.vertex_list_indexed(
            count=self.num,
            mode=self.gl_type,
            indices=self.vert_inds,
            batch=self.window.batch,
            group=self,
            vertices=('f', self.verts.T.flatten()),
            colors=('f', self.colors.T.flatten()),
            # normals = ('normals3f/static', self.normals.T.flatten())
            normals = ('f', self.normals.T.flatten())
        )

        self.visible = True

    def set_colors(self):
        """Take the self.colors attribute and reassign it to an array
        of the length 4 * self.num

        """
        # do we have a single number? if so, set to gray
        if isinstance(self.colors, str) and self.colors == 'ring':
            angs = np.linspace(0, 2 * np.pi, self.num, endpoint=False)
            self.colors = cm_soapbubble(.5, angs)
        elif isinstance(self.colors, str) and self.colors == 'ring0':
            angs = np.linspace(0, 2 * np.pi, self.num, endpoint=True)
            self.colors = cm_soapbubble(.5, angs)
            self.colors[:, -1] = [0, 0, 0, 1.]
        elif isinstance(self.colors, str) and self.colors == 'ring1':
            angs = np.linspace(0, 2 * np.pi, self.num, endpoint=True)
            self.colors = cm_soapbubble(.5, angs)
            self.colors[:, -1] = [1, 1, 1, 1.]
        elif isinstance(self.colors, numbers.Real):
            self.colors = np.array([[*[self.colors] * 3, 1.0]] * self.num).T
        # if we have a 3 tuple of numbers, 1 color for all verts
        elif hasattr(self.colors, '__len__') and \
                len(self.colors) == 3 and \
                isinstance(self.colors[0], numbers.Real):
            self.colors = np.array([[*self.colors, 1.0]] * self.num)
        # if we have a 4 tuple of numbers, 1 color for all verts
        elif hasattr(self.colors, '__len__') and \
                len(self.colors) == 4 and \
                isinstance(self.colors[0], numbers.Real):
            self.colors = np.array([[*self.colors]] * self.num)
        # if we have list of 3 tuples
        elif hasattr(self.colors, '__len__') and \
                hasattr(self.colors[0], '__len__') and \
                len(self.colors[0]) == 3:
            self.colors = np.array([[*c, 1.0] for c in self.colors]).T
        # if we have list of 4 tuples
        elif hasattr(self.colors, '__len__') and \
                hasattr(self.colors[0], '__len__') and \
                len(self.colors[0]) == 4:
            self.colors = np.array([[*c] for c in self.colors]).T
        # if None or something else, use white
        else:
            self.colors = np.array([[1.0] * self.num] * 4)

    def set_state(self):
        super().set_state()
        self.shader_program["light_pos"] = self.light_pos
        self.shader_program["ambient_strength"] = self.ambient_strength

    def unset_state(self):
        super().unset_state()

    def __hash__(self):
        return hash(id(self))
        # return hash((self.texture.target, self.texture.id, self.order, self.parent))

    def __eq__(self, other):
        return id(self) == id(other)

    def __repr__(self):
        return f'{self.__class__.__name__=}'




class Points(Movable_Color):
    """Points of a given size, which can be randomly distributed in a
    volume

    """

    def __init__(self, window, num=1000, colors=None, pt_size=1,
                 verts=None, extent=None, add=False):
        """This is here"""
        # super(Points, self).__init__(window)
        self.pt_size = pt_size
        self.num = num
        # self.color = color
        # self.colors = np.array(np.repeat(color, self.num * 4), dtype=np.uint8)
        self.tex_coords = None
        self.extent = extent

        # if the verts are specified, assign them
        if verts is not None:
            self.set_verts(*verts)
        else:
            self.shuffle()

        verts = self.verts

        super().__init__(
            window=window,
            gl_type=GL_POINTS,
            verts=verts,
            vert_inds=None,
            colors=colors,
            add=add)
        # super().__init__(window, GL_POINTS, self.verts)

        if add: self.add()

    def shuffle(self):
        """Randomly relocate each point throughout the allowed exent

        """
        e = self.extent
        if e is None:
            e = 1

        # Case 1: Single number → spherical volume
        if isinstance(e, (int, float, np.integer, np.floating)):
            rad = e
            r = np.random.uniform(0, 1, self.num) ** (1 / 3) * rad
            phi = np.random.uniform(-np.pi, np.pi, self.num)
            theta = np.arccos(np.random.uniform(-1, 1, self.num))
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta) + 0 * phi  # Ensure broadcast shape
            # self.verts = np.array((x, -z, y))  # OpenGL: y→z, z→-y
            self.set_verts(x, -z, y)

        # Case 2: List or tuple of 2 numbers → cube volume
        elif (
                isinstance(e, (list, tuple)) and len(e) == 2 and
                all(isinstance(v, (int, float)) for v in e)
        ):
            x1, x2 = e
            # self.verts = np.random.uniform(x1, x2, [3, self.num])
            self.set_verts(*np.random.uniform(x1, x2, [3, self.num]))
        # Case 3: 3-element list of 2-element sublists → rectangular cuboid
        elif (
                isinstance(e, (list, tuple)) and len(e) == 3 and
                all(isinstance(sub, (list, tuple)) and len(sub) == 2 and all(isinstance(v, (int, float)) for v in sub)
                    for sub in e)
        ):
            (x1, x2), (x3, x4), (x5, x6) = e
            # self.verts = np.random.uniform([x1, x3, x5], [x2, x4, x6], [self.num, 3]).T
            self.set_verts(*np.random.uniform([x1, x3, x5], [x2, x4, x6], [self.num, 3]).T)

    def set_pt_size(self, pt_size):
        self.pt_size = pt_size

    def set_state(self):
        super().set_state()
        ### I need to rewrite the shader to get point size working
        # glPointSize(self.pt_size)




class Lines(Movable):
    """Lines class for gl_lines, with pairs of endpoints.

    """

    def __init__(self, window, num=100, color=1., ln_width=1,
                 verts=None, add=False):
        super().__init__(window)
        self.gl_type = GL_LINES
        self.ln_width = ln_width
        self.num = num * 2
        self.color = color
        self.colors = np.array(np.repeat(color, self.num * 4), dtype=np.uint8)
        self.tex_coords = None
        if verts is None:
            self.verts = np.random.uniform(-1, 1, [3, self.num])
            self.set_verts(*np.random.uniform(-1, 1, [3, self.num]))
        else:
            self.verts = verts
            self.set_verts(*verts)

        if add: self.add()

    def loop(self, r=1, norm=(0, 1, 0)):
        normal = np.array(norm, dtype=np.float64)
        normal = normal / np.linalg.norm(normal)  # Normalize the normal vector

        if np.allclose(normal, [0, 0, 1]):  # If normal is along z-axis, pick x-axis as reference
            u = np.array([1, 0, 0])
        else:
            u = np.cross(normal, [0, 0, 1])  # Cross with z-axis
            u /= np.linalg.norm(u)  # Normalize u
        v = np.cross(normal, u)
        angs = np.linspace(0, 2 * np.pi, self.num // 2, endpoint=False)
        circle = np.array([
            r * np.cos(angs)[:, np.newaxis] * u +
            r * np.sin(angs)[:, np.newaxis] * v]).sum(axis=0)  # Sum component-wise

        # self.verts = np.roll(circle.repeat(2, 0).T, -1, 1)
        self.set_verts(*np.roll(circle.repeat(2, 0).T, -1, 1))

class Sphere_Lines(Movable_Color):
    """Perpendicularly oreinted circles to outline a sphere of any radius.

    """

    def __init__(self, window, num_latitudes=5, num_verts=64, rad=1, colors=None, add=False):
        self.num = num_verts * num_latitudes * 3

        ind_list = []
        vert_list = [[], [], []]

        inds = np.roll(np.arange(num_verts).repeat(2), -1)
        curr_ind = 0
        angs = np.linspace(0, 2 * np.pi, num_verts, endpoint=False)
        sina, cosa, const = np.sin(angs), np.cos(angs), 0 * angs
        for offset in np.linspace(-rad, rad, num_latitudes + 2)[1:-1]:
            r = np.sin(np.pi * (offset + rad) / (2 * rad))
            # x plane circle
            vert_list[0].extend(const + offset)
            vert_list[1].extend(r * sina)
            vert_list[2].extend(r * cosa)
            ind_list.extend(inds + curr_ind)
            curr_ind += num_verts

            # y plane circle
            vert_list[0].extend(r * sina)
            vert_list[1].extend(const + offset)
            vert_list[2].extend(r * cosa)
            ind_list.extend(inds + curr_ind)
            curr_ind += num_verts

            # z plane circle
            vert_list[0].extend(r * sina)
            vert_list[1].extend(r * cosa)
            vert_list[2].extend(const + offset)
            ind_list.extend(inds + curr_ind)
            curr_ind += num_verts

        super().__init__(
            window=window,
            gl_type=GL_LINES,
            verts=np.array(vert_list),
            vert_inds=np.array(ind_list),
            colors=colors,
            add=add)

        # self.verts = np.array(vert_list)
        # self.vert_inds = np.array(ind_list)


class Triangles(Movable):
    """Any 3d shape made of triangles.
    
    """

    def __init__(self, window, verts=None, color=1., add=False):
        super().__init__(window)
        self.gl_type = GL_TRIANGLES
        self.tex_coords = None

        if verts is None:
            # self.verts = np.array([[-1, np.sin(np.pi * 2 / 3), np.sin(np.pi * 2 / 3)],
            #                        [0, 0, 0],
            #                        [-1, np.cos(np.pi * 2 / 3), np.cos(np.pi * 2 / 3)]])
            self.setverts(*np.array([[-1, np.sin(np.pi * 2 / 3), np.sin(np.pi * 2 / 3)],
                                     [0, 0, 0],
                                     [-1, np.cos(np.pi * 2 / 3), np.cos(np.pi * 2 / 3)]]))
            self.num = 3
        else:
            # self.verts = np.array(verts)
            self.set_verts(*np.array(verts))
            self.num = self.verts.shape[1]

        self.color = color
        if hasattr(color, '__iter__'):
            self.colors = np.array(np.tile(color[:3], self.num), dtype=np.uint8)
        else:
            self.colors = np.array(np.repeat(color, self.num * 3), dtype=np.uint8)

        if add: self.add()


class Square(Movable_Color):
    """Square made with indexed add.

    """

    def __init__(self, window, size, colors=None, tex_coords=None, add=False):
        # self.num = 4
        c = size / 2.
        verts = np.array([[-c, c, c, -c], [-c, -c, c, c], [0, 0, 0, 0.]])
        vert_inds = [0, 1, 2, 2, 3, 0]

        super().__init__(
            window=window,
            gl_type=GL_TRIANGLES,
            verts=verts,
            vert_inds=vert_inds,
            colors=colors,
            add=add)


class Bar(Movable_Color):
    """A simple bar for rotating.

    """

    def __init__(self, window, width=.15, height=2, dist=0.9,
                 color=1.0, stripes=1, add=False):

        verts = [[], [], []]
        vert_inds = []
        colors = []
        for ind, l in enumerate(np.linspace(-width / 2, width / 2, stripes, endpoint=False)):
            r = l + width / stripes
            verts[0].extend([l, r, r, l])
            verts[1].extend([-height, -height, height, height])
            verts[2].extend([-dist] * 4)

            vert_inds.extend(np.array([0, 1, 2, 2, 3, 0]) + 4 * ind)

            if hasattr(color, '__iter__') and len(color) > ind:
                if hasattr(color[ind], '__iter__') and len(color) == 4:
                    colors.extend(color[ind] * 4)
                elif hasattr(color[ind], '__iter__') and len(color) == 3:
                    colors.extend([color[ind][0], color[ind][1], color[ind][2], 1.0] * 4)
                else:
                    colors.extend([color[ind], color[ind], color[ind], 1.0] * 4)
            else:
                colors.extend([color, color, color, 1.0] * 4)

        colors = np.array(colors).reshape(-1, 4).T

        super().__init__(
            window=window,
            gl_type=GL_TRIANGLES,
            verts=verts,
            vert_inds=vert_inds,
            colors=colors,
            add=add)


class Regular_Polygon(Movable_Color):
    """Any polygon defined by x,y planer coordinates

    """

    def __init__(self, window, num_sides=3, rad=1, init_rot=0,
                 init_pos=[0, 0, 0], init_ori=[0, 0, 1], colors=1.,
                 add=False):
        angs = np.linspace(0, 2 * np.pi, num_sides + 1) + init_rot
        rads = np.array([*np.repeat(rad, num_sides), 0])  # rad num_sides then 0
        xs = rads * np.cos(angs)
        ys = rads * np.sin(angs)
        zs = np.zeros(num_sides + 1)
        verts = np.array([xs, ys, zs])

        self._order = 0
        verts = rotate_points_to_normal(verts, init_ori)
        verts += np.array(init_pos)[:, None]

        vert_ind_list = []
        for s in range(num_sides):
            vert_ind_list.extend([num_sides, s, (s + 1) % num_sides])
        vert_inds = np.array(vert_ind_list, dtype='int')

        super().__init__(
            window=window,
            gl_type=GL_TRIANGLES,
            verts=verts,
            vert_inds=vert_inds,
            colors=colors,
            add=add)


class Regular_Star_Polygon(Movable_Color):
    """Any polygon defined by x,y planer coordinates

    """

    def __init__(self, window, num_verts=3, turning_num=2, rad1=1, rad2=None, init_rot=0,
                 init_pos=[0, 0, 0], init_ori=[0, 0, 1], colors=1.,
                 add=False):

        angs = np.linspace(0, 2 * np.pi, num_verts * 2 + 1) + init_rot
        self.angs = angs
        if rad2 is None:
            p, q = num_verts, turning_num
            d1 = (1 - np.cos(2 * np.pi * q / p)) ** 2
            d2 = (1 - np.cos(4 * np.pi * (q - 1) / p))
            n1 = (1 - np.cos(2 * np.pi * q / p)) ** 2 * (np.sin(np.pi * q / p) + np.sin(np.pi * (q - 2) / p)) ** 2 * (
                        np.cos(2 * np.pi * q / p) + 1)
            n2 = 8 * np.sin(np.pi / p) ** 2 * np.sin(np.pi * q / p) ** 2 * np.sin(2 * np.pi * q / p) ** 2 * np.sin(
                np.pi * (q - 1) / p) ** 2
            rad2 = rad1 * np.sqrt((n1 + n2) / (d1 * d2))

        rads = np.array([*np.tile([rad1, rad2], num_verts), 0])  # rad num_sides then 0
        self.rads = rads
        xs = rads * np.cos(angs)
        ys = rads * np.sin(angs)
        zs = np.zeros(num_verts * 2 + 1)
        verts = np.array([xs, ys, zs])

        self._order = 0

        verts = rotate_points_to_normal(verts, init_ori)
        verts += np.array(init_pos)[:, None]

        vert_ind_list = []
        for s in range(num_verts * 2):
            vert_ind_list.extend([num_verts * 2, s, (s + 1) % (num_verts * 2)])
        vert_inds = np.array(vert_ind_list, dtype='int')

        super().__init__(
            window=window,
            gl_type=GL_TRIANGLES,
            verts=verts,
            vert_inds=vert_inds,
            colors=colors,
            add=add)


class Image_File(Movable_Texture):
    """A square that binds a texture from an image file.

    """

    def __init__(self, window, size, im_file, init_rot=0,
                 init_pos=[0, 0, 0], init_ori=[0, 0, 1], add=False):
        texture_image = pyglet.image.load(im_file)
        texture = texture_image.get_texture()

        tex_coords = np.array([[0., 1., 1., 0.], [0., 0., 1., 1.]])

        c = size / 2.
        verts = np.array([[-c, c, c, -c], [-c, -c, c, c], [0, 0, 0, 0.]])
        self._order = 0

        verts = rotate_points_to_normal(verts, init_ori)
        verts += np.array(init_pos)[:, None]

        vert_inds = [0, 1, 2, 2, 3, 0]

        super().__init__(
            window=window,
            gl_type=GL_TRIANGLES,
            verts=verts,
            vert_inds=vert_inds,
            texture=texture,
            tex_coords=tex_coords,
            add=add)


class Grating(Movable_Animation):
    """A grating that can appear anywhere in perspective projection

    """

    def __init__(self, window, rate=60., xres=64, yres=64,
                 sf=.1, tf=1, c=1, o=0, phi_i=0., sd=None, maxframes=500,
                 edge_size=1, init_pos=[0, 0, 0], init_ori=[0, 0, 1], add=False):

        gl_type = GL_QUADS
        tex_coords = np.array([[0., 1., 1., 0.], [0., 0., 1., 1.]], dtype='f4')

        es = edge_size / 2
        verts = np.array([[-es, es, es, -es], [-es, -es, es, es], [0, 0, 0, 0.]], dtype='f4')

        self._order = 0
        verts = rotate_points_to_normal(verts, init_ori)
        verts += np.array(init_pos)[:, None]

        vert_inds = [0, 1, 2, 2, 3, 0]

        self.xres, self.yres = xres, yres
        self.rate = rate

        self.indices = np.indices((self.xres, self.yres))
        h, v = np.meshgrid(np.linspace(-np.pi / 4, np.pi / 4, self.xres), np.linspace(-np.pi / 4, np.pi / 4, self.yres))
        self.center_dists = np.sqrt(h ** 2 + v ** 2)
        self.atans = np.arctan2(h, v)

        self.gratings = []
        self.add_grating(sf, tf, c, o, phi_i, sd, maxframes)
        self.frame_ind = 0

        super().__init__(
            window=window,
            gl_type=GL_TRIANGLES,
            verts=verts,
            vert_inds=vert_inds,
            # ani = self.ani,
            tex_coords=tex_coords,
            add=add)

        if add: self.add()

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
        phi_ss = 2 * np.pi * sf * np.cos(self.atans + (np.radians(o) - np.pi / 2)) * self.center_dists

        # temporal
        if hasattr(tf, '__iter__'):
            tf_array = np.array(tf)  # is it changing?
        else:
            tf_array = np.repeat(tf, min(abs(self.rate / tf), maxframes))  # or constant?

        nframes = len(tf_array)
        self.num_frames = nframes
        phi_ts = np.cumsum(-2 * np.pi * tf_array / float(self.rate))

        for f in np.arange(nframes):
            lum = 127 * (1 + c * np.sin(phi_ss + phi_ts[f] + phi_i))
            data[:, :, :3] = lum[:, :, None]
            data[:, :, 3] = mask_ss * 255
            # make each frame and append the image to the frames list
            frames.append(pyglet.image.ImageData(self.yres, self.xres, 'RGBA', data.tostring()))

        # make the animation with all the frames and append it to the list of playable, moving gratings
        self.ani = pyglet.image.Animation.from_image_sequence(frames, 1. / self.rate)
        self.tbin = pyglet.image.atlas.TextureBin()

        self.ani.add_to_texture_bin(self.tbin)
        self.texture_id = self.tbin.atlases[0].texture.id
        self.gratings.append(pyglet.image.Animation.from_image_sequence(frames, 1. / self.rate))


class STL(Movable_Lighted):
    """Get all the vertices from an stl file

    """

    def __init__(self, window, fn, color=1.0, scale=1,
                 light_pos=(1.,1.,1.), ambient_strength=0.2, add=False):
        stl_mesh = stl.mesh.Mesh.from_file(fn)

        xs = stl_mesh.vectors[:, :, 0].ravel()
        ys = stl_mesh.vectors[:, :, 1].ravel()
        zs = stl_mesh.vectors[:, :, 2].ravel()

        verts = np.array([xs * scale, ys * scale, zs * scale])

        xns = np.repeat(stl_mesh.normals[:, 0].ravel(), 3)
        yns = np.repeat(stl_mesh.normals[:, 1].ravel(), 3)
        zns = np.repeat(stl_mesh.normals[:, 2].ravel(), 3)
        normals = np.array([xns, yns, zns], dtype='f4')

        self.light_pos = light_pos
        self.ambient_strength = ambient_strength

        super().__init__(
            window=window,
            gl_type=GL_TRIANGLES,
            verts=verts,
            vert_inds=None,
            normals = normals,
            colors=color,
            add=add)


# class Horizon(Shape):
#     """A horizon rendered out to some large distance off.

#     """
#     def __init__(self, window, depth, dist, color=1., add=False):
#         c = np.array([[-dist, -dist, dist, dist],[depth, depth, depth, depth], [-dist, dist, dist, -dist]])
#         super(Horizon, self).__init__(window, coords=c, color=color)


class Spherical_Segment(Movable_Color):
    """a spherical segment of any color that can be raised or tilted.
    Making this with phi degrees for the top and bottom polar angles,
    which is 0 at the north pole, and 90 at the south pole.

    """

    def __init__(self, window, polang_top=0, polang_bot=90, radius=1.,
                 color=0., elres=60, azres=60, add=False):

        v, vi = self.init_coords(polang_top, polang_bot, radius, elres, azres)

        super().__init__(
            window=window,
            gl_type=GL_TRIANGLES,
            verts=v,
            vert_inds=vi,
            colors=color,
            add=add)

    def init_coords(self, polang_top=0, polang_bot=180, radius=1.,
                    elres=60, azres=60):
        xlist, ylist, zlist = [], [], []
        vert_inds = []
        vert_list = np.array([0, 1, 2, 2, 3, 0])
        ind = 0
        # change to radians
        phi1 = polang_top * np.pi / 180
        phi2 = polang_bot * np.pi / 180
        # goes a fraction of the way around, so choose num segs based on elres
        els = np.linspace(phi1, phi2, int(np.ceil(elres * (phi2 - phi1) / (2 * np.pi))) + 1)
        # this always goes all the way around
        azs = np.linspace(0, 2 * np.pi, int(azres) + 1)
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

                vert_inds.extend(vert_list + ind)
                ind += 4

        return np.array([xlist, ylist, zlist]), np.array(vert_inds)

class Dot_Cohere_Sph(Movable_Color):
    """Class of randomly distributed points moving along great circle
    trajectories on a spherical surface. Coherent regions move points
    non randomly to some destination azimuth and elevation.

    """

    def __init__(self, window, num=1000, colors=1.0, r=1, pt_size=3,
                 speed=.01, duration=10, new_az=False,
                 add=False):  ###
        """Initialize the sphere points.

        Args:
        window: pyglet window for this stimulus

        num: integer number of points

        colors: how bright

        r: radius for the sphere, typically 1

        pt_size: size

        speed: initial angular speed

        duration: how many frames does each last

        new_az: ?

        add: add this to the window?

        """

        self.num = num
        self.pt_size = pt_size
        self.speed = speed
        self.r = r

        self.duration = duration
        self.regions = []
        self.region_ind = 0

        # for compatibility. new_az puts azimuths where set_ry puts
        # objects
        self.new_az = new_az

        self.init_verts()

        super().__init__(
            window=window,
            gl_type=GL_POINTS,
            verts=self.verts,
            vert_inds=None,
            colors=colors,
            add=add)

        self.time = 0.0
        pyglet.clock.schedule(self.update_time)

        if add: self.add()

    def add_region(self, center=(0,0), radius=5,
                   flow = (0,0), speed=None,
                   coherence=1, regenerate=False, active=False):
        """Add a region where points flow non-randomly.
        
        This adds a dictionary to the regions list that describes how
        points should flow in that space

        Args:
            center : (azimuth, elevation) in degrees, center of the flowing
              region, 0,0 is straight ahead

            radius : float in degrees for the great circle distance from the
              center of the region that points flow

            flow : (azimuth, elevation) floats in degrees for the point on the sphere that
              points in the region flow towards

            speed : float in degrees for the angular distance that points flow
              per frame. None defaults to the wide field dot speed

            coherence : float in degrees, 0--1 for dot coherence in the region

            regenerate : boolean to indicate whether points flowing off the side
              of a region reappear on the other side

        """
        azimuth, elevation = center
        flow_azimuth, flow_elevation = flow

        if speed is None:
            speed = self.speed

        if self.new_az: azimuth*=-1
        cent = azel_to_cart(azimuth, elevation)

        # theta = (elevation + 90) * np.pi / 180
        # if self.new_az:
        #     phi = (-azimuth - 90) * np.pi / 180
        # else:
        #     phi = (azimuth - 90) * np.pi / 180
        #
        # vec = self.sph_to_cart(theta, phi)

        if self.new_az: flow_azimuth*=-1
        flow_vec = azel_to_cart(flow_azimuth, flow_elevation)

        # theta = (flow_elevation + 90) * np.pi / 180
        # if self.new_az:
        #     phi = (-flow_azimuth - 90) * np.pi / 180
        # else:
        #     phi = (flow_azimuth - 90) * np.pi / 180
        # flow_vec = self.sph_to_cart(theta, phi)
        # print(f'{self.gc_distances(vec)}')

        region = {'azimuth': azimuth, 'elevation': elevation,
                  'center': cent, 'radius': radius,
                  'flow_azimuth': flow_azimuth,
                  'flow_elevation': flow_elevation, 'flow_vec': flow_vec,
                  'speed': speed, 'coherence': coherence,
                  'regenerate': regenerate,
                  'old_inds': np.array([], dtype=np.int64),
                  'curr_inds': np.where(self.gc_distances(cent) < np.radians(radius))[0],
                  'active': active}
        # print(f'add_region  {cent=}, {radius=} {region["curr_inds"]=}')

        self.regions.append(region)


    def remove_region(self, ind=-1):
        """Remove one of the regions.

        """
        self.regions.pop(ind)


    def activate_region(self, ind=None):
        """Activate one of the regions."""
        if ind is None:
            ind = self.region_ind
        # ensure it is in range
        ind = ind % len(self.regions)

        region = self.regions[ind]

        region['active'] = True
        center, radius = region['center'], region['radius']
        region['curr_inds'] = np.where(self.gc_distances(center) < np.radians(radius))[0]

        # print(f'activate {ind} {center=} {radius=} {region["curr_inds"]=}')


    def deactivate_region(self, ind=None):
        """Deactivate one of the regions."""
        if ind is None:
            ind = self.region_ind
        # ensure it is in range
        ind = ind % len(self.regions)

        self.regions[ind]['active'] = False
        inds = self.regions[ind]['curr_inds']
        self.assign_uvecs(inds)
        self.speeds[inds] = np.random.uniform(self.speed, self.speed, len(inds))

    def deactivate_regions(self):
        """Deactivate all regions."""
        for region in self.regions:
            region['active'] = False

    def print_active_region_azel(self):
        active_region = [region for region in self.regions if region['active']][0]
        print(f'azimuth = {active_region['azimuth']}, elevation = {active_region['elevation']}')

    def increment_region(self):
        """Add one to the region index"""
        self.region_ind = (self.region_ind + 1) % len(self.regions)


    def get_distributed_centers(self, density):
        """Calculates nearly equal-spaced centers to distribute regions over a
        sphere.

        Uses a Fibonacci sphere spacing to determine the equal spacing of centers
        over the face of the sphere for any density.

        density : the density of centers in regions/steradian

        return: array of coords
        """

        # density = samples/area (sr)
        # area = 4 pi r**2 (sr)
        # samples = density * area (about 12sr/sphere) + 1 (to ensure on odd num)
        samples = density * 12 + 1
        points = []
        phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians

        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append((x, z, y))

        return points

    def set_distributed_regions(self, density, center, radius,
                                flow_center=None, rel_flow_center=None, speed=None):
        """
        Set up many, adjacent regions in a field of view.

        density: the regions/steradian in the group

        center: the center of the fov in (azimuth, elevation) (degrees)

        radius: the maximum distance in degrees from the center to add regions

        """
        # spherical center of receptive tests
        s_x, s_y, s_z = azel_to_cart(*center)
        # prospective regional centers all over the sphere
        r_cents = self.get_distributed_centers(density)
        num_tot = len(r_cents)

        if speed is None:
            speed = self.speed

        for r_x, r_y, r_z in r_cents:
            if np.arccos(np.dot((r_x, r_y, r_z), (s_x, s_y, s_z))) < np.radians(radius):
                r_az, r_el = cart_to_azel(r_x, r_y, r_z)
                # reg_radius = 2*np.arccos(1-2/num_tot)
                reg_radius = 10

                if flow_center is not None:
                    f_az, f_el = flow_center
                elif rel_flow_center is not None:
                    rf_az, rf_el = rel_flow_center
                    f_az = r_az + rf_az
                    f_el = r_el + rf_el
                else:
                    f_az, f_el = 180,0

                self.add_region((r_az, r_el), reg_radius,
                                flow = (f_az, f_el), speed=speed,
                                coherence=1, regenerate=True)


    def view_centers(self):
        """This is for debugging, set the point coords to the positions of the
        region centers. Just to visualize
        """
        for pt_ind in np.arange(self.num):
            if pt_ind >= len(self.regions):
                reg_ind = -1
            else:
                reg_ind = pt_ind
            az, el = self.regions[reg_ind]['azimuth'], self.regions[reg_ind]['elevation']
            x,y,z = azel_to_cart(az, el)
            # print(f'{x,y,z=}')
            self.verts[:,pt_ind] = x,y,z


    def update_coherence(self, reg_ind, coherence):
        """Change coherence in a region.

        """
        self.regions[reg_ind]['coherence'] = coherence

    def update_speed(self, reg_ind, speed):
        """Change dot speed in a region.

        """
        self.regions[reg_ind]['speed'] = speed

    def update_radius(self, reg_ind, radius):
        """Change radius of a region.

        """
        self.regions[reg_ind]['radius'] = radius

    def update_region(self, reg_ind, azimuth=None, elevation=None):
        """Change location of a region.

        """
        if azimuth is not None:
            self.regions[reg_ind]['azimuth'] = azimuth
        if elevation is not None:
            self.regions[reg_ind]['elevation'] = elevation

        theta = (self.regions[reg_ind]['elevation'] + 90) * np.pi / 180
        if self.new_az:
            phi = (-self.regions[reg_ind]['azimuth'] - 90) * np.pi / 180
        else:
            phi = (self.regions[reg_ind]['azimuth'] - 90) * np.pi / 180

        self.regions[reg_ind]['center'] = self.sph_to_cart(theta, phi)

    def update_flow(self, reg_ind, flow_azimuth=None, flow_elevation=None):
        """Change flow direction of coherent dots in a region.

        """
        if flow_azimuth is not None:
            self.regions[reg_ind]['flow_azimuth'] = flow_azimuth
        if flow_elevation is not None:
            self.regions[reg_ind]['flow_elevation'] = flow_elevation

        theta = (self.regions[reg_ind]['flow_elevation'] + 90) * np.pi / 180
        if self.new_az:
            phi = (-self.regions[reg_ind]['flow_azimuth'] - 90) * np.pi / 180
        else:
            phi = (self.regions[reg_ind]['flow_azimuth'] - 90) * np.pi / 180

        self.regions[reg_ind]['flow_vec'] = self.sph_to_cart(theta, phi)

    def gc_distances(self, target_pt, inds=slice(None)):
        """Return the great-circle distance between a unit vector and the
        coordinate points, in radians.

        """
        return np.arccos(np.dot(target_pt, self.verts[:,inds]))


    def rand_pts(self, num):
        """Make random points on the sphere, uniformly distributed.

        """
        phi = np.random.uniform(-np.pi, np.pi, num)
        # arccos needed to make distribution uniform:
        theta = np.arccos(np.random.uniform(-1, 1, num))
        x = self.r * np.sin(theta) * np.cos(phi)
        y = self.r * np.sin(theta) * np.sin(phi)
        # use 0*phi just to make the expression broadcast to the proper length
        z = self.r * np.cos(theta) + 0 * phi
        # in opengl, y is z, and z goes in the negative direction
        return np.array((x, -z, y))

    def assign_uvecs(self, inds=None, target_pts=None):
        """Assign unit rotation vectors to each coordinate point. They must be
        perpendicular to the coordinate point and some other point we
        are targeting on the sphere.

        """
        if inds is None:
            inds = np.arange(self.num)
        if target_pts is None:
            target_pts = self.rand_pts(len(inds))
        vs = np.cross(self.verts[:, inds], target_pts, axis=0)
        vs = vs / np.linalg.norm(vs, axis=0)
        self.uvecs[:, inds] = vs

    def init_verts(self, num=None):
        """Initial sphere of dot positions, their durations, speeds, and
        rotational motion vectors.

        """
        # initial positions
        # self.verts = np.zeros((3, num))
        if num is None:
            num = self.num
        # self.verts = self.rand_pts(num)
        self.set_verts(*self.rand_pts(num))

        # initial directions
        self.uvecs = np.zeros((3, num))
        self.assign_uvecs(np.arange(num))

        # durations, how many frames left for each, temporary dot
        if self.duration == np.inf:
            self.durations = np.repeat(np.inf, num)
        else:
            self.durations = np.random.randint(0, self.duration, num)

        # speeds---since the direction vectors can point anywhere, we don't need negative speeds
        self.speeds = np.random.uniform(self.speed, self.speed, num)

    def rotate(self, inds=slice(None)):
        """Rodrigues' rotation formula, takes a position vector v, a unit
        rotation vector, and an angle, and returns the new position.

        """
        # spd = self.time * self.speeds
        spd = self.speeds[inds]
        cosangs, sinangs = np.cos(spd), np.sin(spd)
        uvecs = self.uvecs[:,inds]
        verts = self.verts[:,inds]

        # the first term scales the vector down
        t1 = verts * cosangs

        # the second skews it (via vector addition) toward the new
        # rotational position.
        t2 = np.cross(uvecs, verts, axis=0) * sinangs

        # The third term re-adds the height (relative to k) that was
        # lost by the first term.
        t3 = uvecs * np.sum(uvecs * verts, axis=0) * (1 - cosangs)

        self.verts[inds] = t1 + t2 + t3



    def wrap_in_circle(self, center, rad, uvec):
        """
        Compute the two intersection points between a spherical circle on the unit sphere
        and a plane through the origin using only NumPy and exact algebraic solving.

        Args:
            center (np.ndarray): (3,) unit vector, center of the spherical circle on the sphere
            rad (float): angular radius in radians
            uvec (np.ndarray): (3,) unit vector, normal vector of the intersecting plane

        Returns:
            np.ndarray: shape (2, 3) array containing the two intersection points
        """
        center = center / np.linalg.norm(center)
        uvec = uvec / np.linalg.norm(uvec)

        # Step 1: Find two orthonormal vectors in the plane
        temp = np.cross(center, uvec)
        if np.linalg.norm(temp) < 1e-8:
            raise ValueError("Circle center is parallel to plane normal.")

        t1 = np.cross(uvec, temp)
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(uvec, t1)

        # Step 2: Solve for a and b such that:
        # a^2 + b^2 = 1 (point lies on sphere in the plane)
        # a*c1 + b*c2 = cos(rad) (angle from center)
        c1 = np.dot(center, t1)
        c2 = np.dot(center, t2)
        cos_theta = np.cos(rad)

        a_solutions = []
        b_solutions = []

        if abs(c2) > 1e-8:
            A = 1 + (c1 / c2) ** 2
            B = -2 * cos_theta * c1 / (c2 ** 2)
            C = (cos_theta ** 2 / c2 ** 2) - 1
            D = B ** 2 - 4 * A * C

            if D < 0:
                print(f'{center=}; {rad=}; {uvec=} # D<0')
                raise ValueError("No real intersection.")

            sqrtD = np.sqrt(D)
            a1 = (-B + sqrtD) / (2 * A)
            b1 = (cos_theta - a1 * c1) / c2
            a2 = (-B - sqrtD) / (2 * A)
            b2 = (cos_theta - a2 * c1) / c2

            a_solutions = [a1, a2]
            b_solutions = [b1, b2]
        else:
            a = cos_theta / c1
            b_squared = 1 - a ** 2
            if b_squared < 0:
                print(f'{center=}; {rad=}; {uvec=} # b_squared<0')
                raise ValueError("No real intersection.")
            b = np.sqrt(b_squared)
            a_solutions = [a, a]
            b_solutions = [b, -b]

        # Step 3: Construct the 3D points from a, b
        points = []
        for a, b in zip(a_solutions, b_solutions):
            p = a * t1 + b * t2
            p /= np.linalg.norm(p)
            points.append(p)
        return p
        # return np.array(points)

    def move(self):
        """Move every point on the sphere by rotating along its uvec

        """
        # decrement the durations
        self.durations -= 1

        # make new positions where old pts have expired
        # start with random directions
        inds = np.where(self.durations <= 0)[0]
        self.verts[:, inds] = self.rand_pts(len(inds))
        self.assign_uvecs(inds)
        self.durations[inds] = self.duration
        self.speeds[inds] = np.random.uniform(self.speed, self.speed, len(inds))

        # check each region, and update uvec direction if needed
        for region in [r for r in self.regions if r['active']]:
            rad = region['radius']
            center = region['center']
            regen = region['regenerate']
            cohere = region['coherence']
            spd = region['speed']
            target = region['flow_vec']

            # which points are in the target area?
            inds = np.where(self.gc_distances(center) < np.radians(rad))[0]

            region['old_inds'] = region['curr_inds']
            region['curr_inds'] = inds.copy()

            # have points entered or exited the region?
            entered = np.setdiff1d(region['curr_inds'], region['old_inds'])
            exited = np.setdiff1d(region['old_inds'], region['curr_inds'])
            # if len(entered)>0:
            #     print(f'{entered=}')
            
            if regen:
                # if points are recycled in the region,
                # first, boot out the ones that entered
                self.durations[entered] = 0
                # then rotate the exited ones 180 deg around the center,
                # so they re-enter the region with the next turn
                if len(exited)>0:
                    # v = self.verts[:,exited]
                    # new_v = 2 * center[:, None] @ (center @ v)[None, :] - v
                    # # new_v = (2 * center[:, None] @ (center @ v)[None:] - v)
                    # self.verts[:,exited] = new_v
                    for ind in exited:
                        vec = self.verts[:,ind]
                        uvec = self.uvecs[:,ind]
                        # p = self.reenter_small_circle(self.verts[:,ind], self.uvecs[:,ind], center, rad)
                        # print (f'wrap {ind} {vec=}; {uvec=}; {center=}; {rad=}#pre')
                        p = self.wrap_in_circle(center, np.radians(rad), uvec)
                        # ap = np.array(p)
                        self.verts[:,ind] = p

                    region['curr_inds'] = region['old_inds']

            else:
                # if points are free to enter or leave the region,
                # just set the exited ones to a random direction vector
                self.assign_uvecs(exited, self.rand_pts(len(exited)))

            # and only some fraction of those flow coherently
            inds = inds[np.where(np.random.rand(len(inds)) < cohere)]

            # change the uvecs for those inds
            if len(inds) > 0:
                self.assign_uvecs(inds, np.tile(target, (len(inds), 1)).T)
                self.speeds[inds] = np.random.uniform(spd, spd, len(inds))


        # rotate all points along their uvecs, according to speed
        self.rotate()

        if self.visible:
            self.vl.vertices = self.verts.T.flatten()

    def set_state(self):
        # self.move()
        # self.time = 0
        super().set_state()
        glPointSize(self.pt_size)

    def unset_state(self):
        super().unset_state()
        glPointSize(1)




class Deadleaf():
    """Randomly distributed disks, overlapping.

    """
    def __init__(self, window, num=500, color=[.7,.8], add=False):
        self.leaves = []
        self.mots = []
        for i in range(num):
            col = np.random.uniform(color[0],color[1])
            size = np.random.uniform(7,20)
            s = Spherical_Segment(window, color=col,
                                  polang_bot=size,
                                  radius=1+i/(3*num))
            s.set_rx(np.random.uniform(0,360))
            s.set_rz(np.random.uniform(0,360))

            self.leaves.append(s)
            self.mots.append(np.random.randn(3))


    def add(self, arg):
        for leaf in self.leaves:
            leaf.add(arg)

    def on(self, arg):
        for leaf in self.leaves:
            leaf.on(arg)

    def move(self):
        for mot,leaf in zip(self.mots, self.leaves):
            leaf.inc_rx(mot[0])
            leaf.inc_ry(mot[1])
            leaf.inc_rz(mot[2])

