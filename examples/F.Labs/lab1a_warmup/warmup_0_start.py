"""
Lab 1 -- Warm-up, starting point: ONE cube, a terrain and the axes (example B5, trimmed)

This is where the lab starts. A cube is an Entity with 4 components:
    BasicTransform (where), RenderMesh (the vertices), VertexArray (sent to the GPU), Shader (how to draw).

    python warmup_0_start.py
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

# A cube with side 1, centered on the origin (the cube of examples 3, 4 and 5)
vertexCube = np.array([
    [-0.5, -0.5,  0.5, 1.0],
    [-0.5,  0.5,  0.5, 1.0],
    [ 0.5,  0.5,  0.5, 1.0],
    [ 0.5, -0.5,  0.5, 1.0],
    [-0.5, -0.5, -0.5, 1.0],
    [-0.5,  0.5, -0.5, 1.0],
    [ 0.5,  0.5, -0.5, 1.0],
    [ 0.5, -0.5, -0.5, 1.0]
], dtype=np.float32)

indexCube = np.array((1, 0, 3,  1, 3, 2,
                      2, 3, 7,  2, 7, 6,
                      3, 0, 4,  3, 4, 7,
                      6, 5, 1,  6, 1, 2,
                      4, 5, 6,  4, 6, 7,
                      5, 4, 0,  5, 0, 1), np.uint32)

# One color per vertex (8 vertices -> 8 colors). The GPU blends the colors across each triangle.
colorCubeRainbow = [
    [0.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0, 1.0]
]


# ==========================================================================================
# THE CUBE: one Entity with a transform, a mesh, a vertex array and a shader
# ==========================================================================================
cube = scene.world.createEntity(Entity(name="cube"))
scene.world.addEntityChild(rootEntity, cube)
cube_trans = scene.world.addComponent(cube, BasicTransform(name="cube_trans", trs=util.translate(0.0, 0.5, 0.0)))   # lift it onto the grid
cube_mesh = scene.world.addComponent(cube, RenderMesh(name="cube_mesh"))
cube_mesh.vertex_attributes.append(vertexCube)                                           # attribute 0: positions
cube_mesh.vertex_attributes.append(np.array(colorCubeRainbow, dtype=np.float32))         # attribute 1: colors
cube_mesh.vertex_index.append(indexCube)                                                 # which vertices make a triangle
cube_vArray = scene.world.addComponent(cube, VertexArray())
cube_shader = scene.world.addComponent(cube, ShaderGLDecorator(
    Shader(vertex_import_file=SHADER_DIR / "ColorMVP.vert", fragment_import_file=SHADER_DIR / "Color.frag")))

# ==========================================================================================
# WINDOW, EVENTS, CAMERA
# ==========================================================================================
scene.init(imgui=True, windowWidth=winWidth, windowHeight=winHeight, windowTitle="Elements: Lab 1 - one cube", customImGUIdecorator=ImGUIecssDecorator2, openGLversion=4)

# Everything created so far has to be sent to the GPU once (needs the window, i.e. an OpenGL context)
scene.world.traverse_visit(initUpdate, scene.world.root)

gWindow = scene.renderWindow          # the SDL2 window: pixels, mouse, keyboard, and the camera matrix
gGUI = scene.gContext               # the ImGui layer around it, which also owns the camera

# Camera: where we stand (eye), what we look at (target), which way is up
eye = util.vec(5.0, 4.0, 7.0)
target = util.vec(0.0, 0.3, 0.0)
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
    cube_shader.setUniformVariable(key='modelViewProj', value=projMat @ view @ cube_trans.l2world, mat4=True)

    scene.world.traverse_visit(renderUpdate, scene.world.root)        # draw everything
    scene.render_post()

scene.shutdown()
