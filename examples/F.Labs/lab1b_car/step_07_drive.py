"""
Lab 1 -- Car, step 7: the car drives back and forth

The car of step 5, exactly. Slide 2: what if the car could also move forward? The car node moves, the tires follow
it (they are its children) AND turn on their own axles. The angle of the tires is tied to the distance the car has
covered. Driving backwards (x decreasing) turns the tires backwards, for free.

    python step_07_drive.py
"""
import numpy as np
import time

import Elements.pyECSS.math_utilities as util
from Elements.pyECSS.Entity import Entity
from Elements.pyECSS.Component import BasicTransform, RenderMesh
from Elements.pyECSS.System import TransformSystem
from Elements.pyGLV.GL.Scene import Scene
from Elements.pyGLV.GUI.ImguiDecorator import ImGUIecssDecorator2
from Elements.pyGLV.GL.Shader import InitGLShaderSystem, Shader, ShaderGLDecorator, RenderGLShaderSystem
from Elements.pyGLV.GL.VertexArray import VertexArray
from Elements.utils.terrain import generateTerrain
from Elements.definitions import SHADER_DIR

from OpenGL.GL import GL_LINES

winWidth = 1200
winHeight = 800

scene = Scene()

# Scenegraph: everything hangs below the root
rootEntity = scene.world.createEntity(Entity(name="RooT"))

# Systems
transUpdate = scene.world.createSystem(TransformSystem("transUpdate", "TransformSystem", "001"))
initUpdate = scene.world.createSystem(InitGLShaderSystem())
renderUpdate = scene.world.createSystem(RenderGLShaderSystem())


## TERRAIN: a grid on the xz plane. One square of the grid is one unit ##
vertexTerrain, indexTerrain, colorTerrain = generateTerrain(size=5)
terrain = scene.world.createEntity(Entity(name="terrain"))
scene.world.addEntityChild(rootEntity, terrain)
terrain_trans = scene.world.addComponent(terrain, BasicTransform(name="terrain_trans", trs=util.identity()))
terrain_mesh = scene.world.addComponent(terrain, RenderMesh(name="terrain_mesh"))
terrain_mesh.vertex_attributes.append(vertexTerrain)
terrain_mesh.vertex_attributes.append(colorTerrain)
terrain_mesh.vertex_index.append(indexTerrain)
terrain_vArray = scene.world.addComponent(terrain, VertexArray(primitive=GL_LINES))
terrain_shader = scene.world.addComponent(terrain, ShaderGLDecorator(
    Shader(vertex_import_file=SHADER_DIR / "ColorMVP.vert", fragment_import_file=SHADER_DIR / "Color.frag")))

## AXES: x is red, y is green, z is blue ##
vertexAxes = np.array([
    [0.0, 0.0, 0.0, 1.0], [3.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 1.0], [0.0, 3.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 3.0, 1.0]
], dtype=np.float32)
colorAxes = np.array([
    [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]
], dtype=np.float32)
indexAxes = np.array((0, 1, 2, 3, 4, 5), np.uint32)

axes = scene.world.createEntity(Entity(name="axes"))
scene.world.addEntityChild(rootEntity, axes)
axes_trans = scene.world.addComponent(axes, BasicTransform(name="axes_trans", trs=util.translate(0.0, 0.002, 0.0)))   # a hair above the grid
axes_mesh = scene.world.addComponent(axes, RenderMesh(name="axes_mesh"))
axes_mesh.vertex_attributes.append(vertexAxes)
axes_mesh.vertex_attributes.append(colorAxes)
axes_mesh.vertex_index.append(indexAxes)
axes_vArray = scene.world.addComponent(axes, VertexArray(primitive=GL_LINES))
axes_shader = scene.world.addComponent(axes, ShaderGLDecorator(
    Shader(vertex_import_file=SHADER_DIR / "ColorMVP.vert", fragment_import_file=SHADER_DIR / "Color.frag")))

# ==========================================================================================
# MESHES. A mesh is a few arrays, one row per vertex: positions (x, y, z, 1), colors (r, g, b, a)
# and an index list that says which three vertices make a triangle.
# ==========================================================================================

# Primitive B of the exercise: the trapezoid, pulled out 1 unit along z (z from -0.5 to 0.5).
# Its origin is the bottom right corner: x from -2 to 0, y from 0 to 1.
#
#       3 ------- 2
#       |           \
#       |             \
#       0 ------------- 1          vertices 0-3 are on the front face (z = +0.5),
#                                  vertices 4-7 are the same points on the back face (z = -0.5)
vertexTrapezoid = np.array([
    [-2.0, 0.0,  0.5, 1.0],     # 0
    [ 0.0, 0.0,  0.5, 1.0],     # 1
    [-1.0, 1.0,  0.5, 1.0],     # 2
    [-2.0, 1.0,  0.5, 1.0],     # 3
    [-2.0, 0.0, -0.5, 1.0],     # 4
    [ 0.0, 0.0, -0.5, 1.0],     # 5
    [-1.0, 1.0, -0.5, 1.0],     # 6
    [-2.0, 1.0, -0.5, 1.0]      # 7
], dtype=np.float32)

indexTrapezoid = np.array((0, 1, 2,  0, 2, 3,       # front
                           4, 6, 5,  4, 7, 6,       # back
                           0, 4, 5,  0, 5, 1,       # bottom
                           1, 5, 6,  1, 6, 2,       # slanted side
                           2, 6, 7,  2, 7, 3,       # top
                           3, 7, 4,  3, 4, 0), np.uint32)   # left

# Primitive A of the exercise: the circle of radius 1, pulled out 1 unit along y (y from -0.5 to 0.5):
# a can standing up, along the green axis.
SEGMENTS = 20                    # the more segments, the rounder the circle

vertexWheel = []
for i in range(SEGMENTS):
    angle = 2 * np.pi * i / SEGMENTS
    x = np.cos(angle)
    z = -np.sin(angle)                           # minus: so that the triangles below keep facing outwards
    vertexWheel.append([x,  0.5, z, 1.0])        # vertex 2*i      : on the circle, top face
    vertexWheel.append([x, -0.5, z, 1.0])        # vertex 2*i + 1  : on the circle, bottom face
vertexWheel.append([0.0,  0.5, 0.0, 1.0])        # vertex 2*SEGMENTS      : centre of the top face
vertexWheel.append([0.0, -0.5, 0.0, 1.0])        # vertex 2*SEGMENTS + 1  : centre of the bottom face
vertexWheel = np.array(vertexWheel, dtype=np.float32)

indexWheel = []
for i in range(SEGMENTS):
    j = (i + 1) % SEGMENTS                       # the next vertex around the circle
    indexWheel += [2*i, 2*i + 1, 2*j + 1,   2*i, 2*j + 1, 2*j]    # the side: two triangles
    indexWheel += [2 * SEGMENTS, 2*i, 2*j]                        # top face: one triangle (like a pizza slice)
    indexWheel += [2 * SEGMENTS + 1, 2*j + 1, 2*i + 1]            # bottom face
indexWheel = np.array(indexWheel, np.uint32)

# ==========================================================================================
# THE CAR: one Entity with one transform. All the parts below are its children.
# ==========================================================================================
car = scene.world.createEntity(Entity(name="car"))
scene.world.addEntityChild(rootEntity, car)
car_trans = scene.world.addComponent(car, BasicTransform(name="car_trans", trs=util.identity()))

parts = []                     # (shader, transform) of every mesh we draw

def draw_trapezoid(name, color, trs):
    """Put a trapezoid in the scene. `color` is [r, g, b, a]; `trs` says how big, how turned and where."""
    entity = scene.world.createEntity(Entity(name=name))
    scene.world.addEntityChild(car, entity)
    trans = scene.world.addComponent(entity, BasicTransform(name=name + "_trans", trs=trs))
    mesh = scene.world.addComponent(entity, RenderMesh(name=name + "_mesh"))
    mesh.vertex_attributes.append(vertexTrapezoid)
    mesh.vertex_attributes.append(np.array([color] * 8, dtype=np.float32))     # one color per vertex: 8 vertices
    mesh.vertex_index.append(indexTrapezoid)
    scene.world.addComponent(entity, VertexArray())
    shader = scene.world.addComponent(entity, ShaderGLDecorator(
        Shader(vertex_import_file=SHADER_DIR / "ColorMVP.vert", fragment_import_file=SHADER_DIR / "Color.frag")))
    parts.append((shader, trans))


def draw_cylinder(name, color, trs):
    """Put a cylinder in the scene. `color` is [r, g, b, a]; `trs` says how big, how turned and where."""
    entity = scene.world.createEntity(Entity(name=name))
    scene.world.addEntityChild(car, entity)
    trans = scene.world.addComponent(entity, BasicTransform(name=name + "_trans", trs=trs))
    mesh = scene.world.addComponent(entity, RenderMesh(name=name + "_mesh"))
    colors = [color] * len(vertexWheel)
    colors[0] = colors[1] = [0.95, 0.95, 0.95, 1.0]        # one white mark, so we can SEE the cylinder turn
    mesh.vertex_attributes.append(vertexWheel)
    mesh.vertex_attributes.append(np.array(colors, dtype=np.float32))
    mesh.vertex_index.append(indexWheel)
    scene.world.addComponent(entity, VertexArray())
    shader = scene.world.addComponent(entity, ShaderGLDecorator(
        Shader(vertex_import_file=SHADER_DIR / "ColorMVP.vert", fragment_import_file=SHADER_DIR / "Color.frag")))
    parts.append((shader, trans))
    return trans                                           # NEW: we want to turn the tires later

red = [0.85, 0.10, 0.10, 1.0]
orange = [0.95, 0.55, 0.05, 1.0]
green = [0.15, 0.60, 0.20, 1.0]
blue = [0.20, 0.35, 0.85, 1.0]
lightblue = [0.65, 0.85, 0.92, 1.0]
grey = [0.60, 0.60, 0.60, 1.0]
tire_grey = [0.42, 0.42, 0.46, 1.0]

# the body, two trapezoids that fit together into a 3 x 1 x 2 box
draw_trapezoid("body_1", red, util.translate(3.0, 0.5, 0.0) @ util.scale(1.0, 1.0, 2.0))
draw_trapezoid("body_2", orange, util.translate(2.0, 1.5, 0.0) @ util.rotate((0, 0, 1), 180) @ util.scale(1.0, 1.0, 2.0))
# the glass and the back
draw_trapezoid("glass", blue, util.translate(4.0, 1.5, 0.0) @ util.scale(1.5, 1.0, 2.0))
draw_trapezoid("back", green, util.translate(1.0, 2.5, 0.0) @ util.rotate((0, 0, 1), 90) @ util.scale(1.0, 1.0, 2.0))

def tire_matrix(x, z, angle):
    """The tire is scaled to its size, spun about ITS OWN axis (the y of the cylinder), laid down, and only then moved to its place."""
    return util.translate(x, 0.5, z) @ util.rotate((1, 0, 0), 90) @ util.rotate((0, 1, 0), angle) @ util.scale(0.5, 0.4, 0.5)

# The four tires, as in step 5, but we keep their transforms (and where they are): we will turn them.
tires = []                     # (transform, x, z)
for x in (1.0, 3.0):
    for z in (-1.0, 1.0):
        trans = draw_cylinder("tire_%d_%d" % (x, z), tire_grey, tire_matrix(x, z, 0.0))
        tires.append((trans, x, z))

# Moving the car = changing ONE matrix. Every part follows: the world matrix of a part is  car.trs @ part.trs ----
# Read it from right to left: scale the whole car down to 0.8, turn it 30 degrees about y (so we see its side), move it toward the camera (away from the axes).
car_start = util.translate(-2.0, 0.0, 3.8) @ util.rotate((0, 1, 0), 30) @ util.scale(0.8)
car_trans.trs = car_start

# ==========================================================================================
# WINDOW, EVENTS, CAMERA
# ==========================================================================================
scene.init(imgui=True, windowWidth=winWidth, windowHeight=winHeight, windowTitle="Elements: Car lab - step 7", customImGUIdecorator=ImGUIecssDecorator2, openGLversion=4)

# Everything created so far has to be sent to the GPU once (needs the window, i.e. an OpenGL context)
scene.world.traverse_visit(initUpdate, scene.world.root)

gWindow = scene.renderWindow          # the SDL2 window: pixels, mouse, keyboard, and the camera matrix
gGUI = scene.gContext               # the ImGui layer around it, which also owns the camera

# Camera: where we stand (eye), what we look at (target), which way is up
eye = util.vec(7.5, 7.0, 11.0)
target = util.vec(0.5, 0.5, 0.0)
up = util.vec(0.0, 1.0, 0.0)
gGUI.createViewMatrix(eye, target, up)                  # sets the camera (the mouse and the GUI move it from here)
projMat = util.perspective(50.0, winWidth / winHeight, 0.1, 100.0)

start_time = time.time()

# ==========================================================================================
# MAIN LOOP
# ==========================================================================================
running = True
while running:
    running = scene.render()                                          # window, events, GUI

    scene.world.traverse_visit(transUpdate, scene.world.root)         # compute the world matrix (l2world) of every transform
    view = gWindow._myCamera                                           # the camera: moves with the mouse / the GUI
    terrain_shader.setUniformVariable(key='modelViewProj', value=projMat @ view @ terrain_trans.l2world, mat4=True)
    axes_shader.setUniformVariable(key='modelViewProj', value=projMat @ view @ axes_trans.l2world, mat4=True)
    for shader, trans in parts:
        shader.setUniformVariable(key='modelViewProj', value=projMat @ view @ trans.l2world, mat4=True)


    # ---- NEW: the car drives, the tires roll ----
    t = time.time() - start_time                                       # seconds since the program started
    x = 3.0 * np.sin(t)                                                # the car swings between x = -3 and x = 3 ...
    car_trans.trs = car_start @ util.translate(x, 0.0, 0.0)            # ... the translation of the WHOLE car comes last
    tire_angle = -np.degrees(x / 0.5)                                  # the tires ROLL: angle = distance / radius (radius = 0.5)
    for tire_trans, tx, tz in tires:
        tire_trans.trs = tire_matrix(tx, tz, tire_angle)
    scene.world.traverse_visit(renderUpdate, scene.world.root)        # draw everything
    scene.render_post()

scene.shutdown()
