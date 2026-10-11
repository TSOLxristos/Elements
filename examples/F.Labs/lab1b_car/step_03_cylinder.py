"""
Lab 1 -- Car, step 3: the second building block, the cylinder, and the four tires

draw_cylinder() is draw_trapezoid() with another mesh (primitive A of the exercise: a circle of radius 1,
pulled out 1 unit along y). The light grey one is the cylinder as it is built, standing along the green axis.
The four mid-grey ones are the same mesh as tires: scaled small (the scale uses the cylinder's own axes), laid down
(rotated 90 degrees about x) so that they roll along x, and moved to the four corners of the car.
The white line is there so that we can SEE a tire turn, later.

    python step_03_cylinder.py
"""
import numpy as np

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

parts = []                     # (shader, transform) of every mesh we draw

def draw_cylinder(name, color, trs):
    """Put a cylinder in the scene. `color` is [r, g, b, a]; `trs` says how big, how turned and where."""
    entity = scene.world.createEntity(Entity(name=name))
    scene.world.addEntityChild(rootEntity, entity)
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

red = [0.85, 0.10, 0.10, 1.0]
orange = [0.95, 0.55, 0.05, 1.0]
green = [0.15, 0.60, 0.20, 1.0]
blue = [0.20, 0.35, 0.85, 1.0]
lightblue = [0.65, 0.85, 0.92, 1.0]
grey = [0.60, 0.60, 0.60, 1.0]
tire_grey = [0.42, 0.42, 0.46, 1.0]

# As it is built: radius 1, 1 high, standing along y. (Only moved aside, to x = -3, to keep it clear of the tires.)
draw_cylinder("cylinder", grey, util.translate(-3.0, 0.0, 0.0))

# The four tires: the cylinder, scaled to a small tire, laid down, and moved to its place.
# Read each matrix from right to left:
#   scale   in the cylinder's own axes: x and z are the circle (radius 0.5), y is its thickness (0.4)
#   rotate  90 degrees about x: the axis of the cylinder, y, becomes z (the tire now rolls along x)
#   translate  to x = 1 or 3 (along the car), y = 0.5 (radius: it touches the ground), z = -1 or 1 (the two sides of the car)
draw_cylinder("tire_1", tire_grey, util.translate(1.0, 0.5, -1.0) @ util.rotate((1, 0, 0), 90) @ util.scale(0.5, 0.4, 0.5))
draw_cylinder("tire_2", tire_grey, util.translate(1.0, 0.5,  1.0) @ util.rotate((1, 0, 0), 90) @ util.scale(0.5, 0.4, 0.5))
draw_cylinder("tire_3", tire_grey, util.translate(3.0, 0.5, -1.0) @ util.rotate((1, 0, 0), 90) @ util.scale(0.5, 0.4, 0.5))
draw_cylinder("tire_4", tire_grey, util.translate(3.0, 0.5,  1.0) @ util.rotate((1, 0, 0), 90) @ util.scale(0.5, 0.4, 0.5))

# ==========================================================================================
# WINDOW, EVENTS, CAMERA
# ==========================================================================================
scene.init(imgui=True, windowWidth=winWidth, windowHeight=winHeight, windowTitle="Elements: Car lab - step 3", customImGUIdecorator=ImGUIecssDecorator2, openGLversion=4)

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

    scene.world.traverse_visit(renderUpdate, scene.world.root)        # draw everything
    scene.render_post()

scene.shutdown()
