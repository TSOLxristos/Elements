"""
Lab 1 -- Warm-up (c): the same cubes as warm-up (b), now with normals and a lit (Blinn-Phong) shader

Exactly warmup_b_dynamic.py, with three changes (search for NEW):
  1. one normal per vertex, from Elements.utils.normals          (add_cube)
  2. the shader that uses the normals (Blinn-Phong)              (add_cube)
  3. the light's uniform variables, set for every cube every frame (main loop; given in lab_lighting.py)
Nothing else changes. A cube with ONE color now has visible faces.

    python warmup_c_normals.py
"""
import numpy as np
import imgui
import random

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
import Elements.utils.normals as norm

from OpenGL.GL import GL_LINES

from lab_lighting import apply_lighting      # given to you: the light (lights are the next lecture)

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
# add_cube(): everything we typed for the cube above, in a function we can call again and again
# ==========================================================================================
cubes = []                     # (entity, shader, transform) of every cube we have added

def add_cube(name, position, size, colors):
    """Add a cube centered at `position` = (x, y, z), with edge length `size`.
    `colors` is a list with one color per vertex (8 colors)."""
    entity = scene.world.createEntity(Entity(name=name))
    scene.world.addEntityChild(rootEntity, entity)
    # The cube we have has side 1. First SCALE it to the size we want, then MOVE it to its place:
    trs = util.translate(position[0], position[1], position[2]) @ util.scale(size)
    trans = scene.world.addComponent(entity, BasicTransform(name=name + "_trans", trs=trs))
    mesh = scene.world.addComponent(entity, RenderMesh(name=name + "_mesh"))
    # NEW: one normal per vertex. Flat shading: each triangle gets its own vertices and its own normal.
    vertices, indices, colors, normals = norm.generateFlatNormalsMesh(vertexCube, indexCube, np.array(colors, dtype=np.float32))
    mesh.vertex_attributes.append(vertices)
    mesh.vertex_attributes.append(colors)
    mesh.vertex_attributes.append(normals)                     # NEW: attribute 2 = normals
    mesh.vertex_index.append(indices)
    scene.world.addComponent(entity, VertexArray())
    shader = scene.world.addComponent(entity, ShaderGLDecorator(
        Shader(vertex_import_file=SHADER_DIR / "Phong.vert", fragment_import_file=SHADER_DIR / "BlinnPhong.frag")))   # NEW: a shader that uses the normals (Blinn-Phong)
    cubes.append((entity, shader, trans))

# ==========================================================================================
# WARM-UP (a): as many small cubes as we like, placed from the code. No GUI needed.
# ==========================================================================================

# A row of 10 small cubes along the x axis. Their color goes from red to blue.
for i in range(10):
    t = i / 9.0                                       # goes from 0.0 to 1.0
    color = [1.0 - t, 0.2, t, 1.0]                    # r, g, b, a
    add_cube("row_%d" % i, (-2.25 + 0.5 * i, 0.15, 3.0), 0.3, [color] * 8)

# A 5 x 5 "chess board" of small cubes: every other one is orange, the rest are white.
for i in range(5):
    for j in range(5):
        if (i + j) % 2 == 0:
            color = [1.0, 0.5, 0.0, 1.0]
        else:
            color = [1.0, 1.0, 1.0, 1.0]
        add_cube("board_%d_%d" % (i, j), (-1.0 + 0.5 * i, 0.15, -2.0 + 0.5 * j), 0.3, [color] * 8)

# One cube with a different color on each corner (8 vertices -> 8 colors)
add_cube("rainbow", (3.0, 0.5, 0.0), 1.0, colorCubeRainbow)

# ==========================================================================================
# WARM-UP (b): our own ImGui panel, to add cubes WHILE the program is running
# ==========================================================================================

# The values the sliders edit. A dictionary is an easy way to keep them together.
ui = {"x": 0.0, "y": 0.15, "z": 0.0, "size": 0.3, "color": [1.0, 0.5, 0.0], "rainbow": False}

def spawn_cube():
    """What the 'Add cube' button does."""
    if ui["rainbow"]:
        colors = colorCubeRainbow
    else:
        colors = [list(ui["color"]) + [1.0]] * 8                  # the same color on all 8 vertices
    add_cube("cube_%d" % len(cubes), (ui["x"], ui["y"], ui["z"]), ui["size"], colors)
    # The new cube has never been sent to the GPU, so do it now -- just for this entity.
    # (Skip this and the program crashes when it tries to draw the cube.)
    scene.world.traverse_visit(initUpdate, cubes[-1][0])

# ==========================================================================================
# WINDOW, EVENTS, CAMERA
# ==========================================================================================
scene.init(imgui=True, windowWidth=winWidth, windowHeight=winHeight, windowTitle="Elements: Lab 1 - warm-up (b), cubes at runtime", customImGUIdecorator=ImGUIecssDecorator2, openGLversion=4)

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

    # ---- our panel ---------------------------------------------------------------------
    imgui.set_next_window_position(winWidth - 320, 20, imgui.FIRST_USE_EVER)
    imgui.set_next_window_size(300, 0, imgui.FIRST_USE_EVER)
    imgui.begin("Add cubes")
    _, ui["x"] = imgui.slider_float("x", ui["x"], -5.0, 5.0)
    _, ui["y"] = imgui.slider_float("y", ui["y"], 0.0, 5.0)
    _, ui["z"] = imgui.slider_float("z", ui["z"], -5.0, 5.0)
    _, ui["size"] = imgui.slider_float("size", ui["size"], 0.1, 1.0)
    _, ui["color"] = imgui.color_edit3("color", *ui["color"])
    _, ui["rainbow"] = imgui.checkbox("a different color on every corner", ui["rainbow"])
    if imgui.button("Add cube"):
        spawn_cube()
    imgui.same_line()
    if imgui.button("Random color"):
        ui["color"] = [random.random(), random.random(), random.random()]
    if imgui.button("Remove last") and len(cubes) > 0:
        entity, shader, trans = cubes.pop()
        rootEntity.remove(entity)                          # take it out of the scenegraph
    imgui.text("cubes in the scene: %d" % len(cubes))
    imgui.end()
    # -------------------------------------------------------------------------------------

    scene.world.traverse_visit(transUpdate, scene.world.root)         # compute the world matrix (l2world) of every transform
    view = gWindow._myCamera                                           # the camera: moves with the mouse / the GUI
    terrain_shader.setUniformVariable(key='modelViewProj', value=projMat @ view @ terrain_trans.l2world, mat4=True)
    axes_shader.setUniformVariable(key='modelViewProj', value=projMat @ view @ axes_trans.l2world, mat4=True)
    for entity, shader, trans in cubes:
        shader.setUniformVariable(key='modelViewProj', value=projMat @ view @ trans.l2world, mat4=True)
        apply_lighting(shader, trans.l2world, gWindow._cameraEye)      # NEW: the light (given); _cameraEye is where the camera is

    scene.world.traverse_visit(renderUpdate, scene.world.root)        # draw everything
    scene.render_post()

scene.shutdown()
