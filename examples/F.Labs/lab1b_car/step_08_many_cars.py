"""
Lab 1 -- Car, step 8: many cars

Everything we typed for ONE car goes into a function, make_car(). Call it four times from the code, and then from a button
of our own ImGui panel while the program runs (the same trick as warm-up b).

    python step_08_many_cars.py
"""
import numpy as np
import imgui
import random
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
# NEW: make_car() -- everything of step 7, for ONE car, in a function.
# It returns what the main loop needs: the car node, its transform, its parts, its tires.
# ==========================================================================================
red = [0.85, 0.10, 0.10, 1.0]
orange = [0.95, 0.55, 0.05, 1.0]
green = [0.15, 0.60, 0.20, 1.0]
purple = [0.50, 0.25, 0.70, 1.0]
lightblue = [0.65, 0.85, 0.92, 1.0]
tire_grey = [0.42, 0.42, 0.46, 1.0]

def tire_matrix(x, z, angle):
    """The tire is scaled to its size, spun about ITS OWN axis (the y of the cylinder), laid down, and only then moved to its place."""
    return util.translate(x, 0.5, z) @ util.rotate((1, 0, 0), 90) @ util.rotate((0, 1, 0), angle) @ util.scale(0.5, 0.4, 0.5)

def make_car(name, trs, body_color):
    """Build a car (a node with seven children) below the root. `trs` places the whole car, `body_color` is [r, g, b, a]."""
    car = scene.world.createEntity(Entity(name=name))
    scene.world.addEntityChild(rootEntity, car)
    car_trans = scene.world.addComponent(car, BasicTransform(name=name + "_trans", trs=trs))
    parts = []                     # (shader, transform) of every mesh of this car
    tires = []                     # (transform, x, z)

    def draw_trapezoid(part_name, color, part_trs):
        entity = scene.world.createEntity(Entity(name=name + "_" + part_name))
        scene.world.addEntityChild(car, entity)
        trans = scene.world.addComponent(entity, BasicTransform(name=name + "_" + part_name + "_trans", trs=part_trs))
        mesh = scene.world.addComponent(entity, RenderMesh(name=name + "_" + part_name + "_mesh"))
        mesh.vertex_attributes.append(vertexTrapezoid)
        mesh.vertex_attributes.append(np.array([color] * 8, dtype=np.float32))
        mesh.vertex_index.append(indexTrapezoid)
        scene.world.addComponent(entity, VertexArray())
        shader = scene.world.addComponent(entity, ShaderGLDecorator(
            Shader(vertex_import_file=SHADER_DIR / "ColorMVP.vert", fragment_import_file=SHADER_DIR / "Color.frag")))
        parts.append((shader, trans))

    def draw_cylinder(part_name, color, part_trs):
        entity = scene.world.createEntity(Entity(name=name + "_" + part_name))
        scene.world.addEntityChild(car, entity)
        trans = scene.world.addComponent(entity, BasicTransform(name=name + "_" + part_name + "_trans", trs=part_trs))
        mesh = scene.world.addComponent(entity, RenderMesh(name=name + "_" + part_name + "_mesh"))
        colors = [color] * len(vertexWheel)
        colors[0] = colors[1] = [0.95, 0.95, 0.95, 1.0]        # one white mark, so we can SEE the tire turn
        mesh.vertex_attributes.append(vertexWheel)
        mesh.vertex_attributes.append(np.array(colors, dtype=np.float32))
        mesh.vertex_index.append(indexWheel)
        scene.world.addComponent(entity, VertexArray())
        shader = scene.world.addComponent(entity, ShaderGLDecorator(
            Shader(vertex_import_file=SHADER_DIR / "ColorMVP.vert", fragment_import_file=SHADER_DIR / "Color.frag")))
        parts.append((shader, trans))
        return trans

    # the body of step 2: the body and the back take the color of this car, the glass stays light blue
    draw_trapezoid("body_1", body_color, util.translate(3.0, 0.5, 0.0) @ util.scale(1.0, 1.0, 2.0))
    draw_trapezoid("body_2", body_color, util.translate(2.0, 1.5, 0.0) @ util.rotate((0, 0, 1), 180) @ util.scale(1.0, 1.0, 2.0))
    draw_trapezoid("glass", lightblue, util.translate(4.0, 1.5, 0.0) @ util.scale(1.5, 1.0, 2.0))
    draw_trapezoid("back", body_color, util.translate(1.0, 2.5, 0.0) @ util.rotate((0, 0, 1), 90) @ util.scale(1.0, 1.0, 2.0))
    # the four tires of step 6
    for x in (1.0, 3.0):
        for z in (-1.0, 1.0):
            tires.append((draw_cylinder("tire_%d_%d" % (x, z), tire_grey, tire_matrix(x, z, 0.0)), x, z))
    return {"entity": car, "trans": car_trans, "base": trs, "parts": parts, "tires": tires}

def car_matrix(x, z, heading, size):
    """Scale about the car's middle (x = 2): move the middle to the origin LAST in the product (= first applied), then scale, turn, place."""
    return util.translate(x, 0.0, z) @ util.rotate((0, 1, 0), heading) @ util.scale(size) @ util.translate(-2.0, 0.0, 0.0)

# Four cars from the code: four places, four colors. The list `cars` keeps track of them.
cars = []
cars.append(make_car("car_0", car_matrix(0.0, -4.2, 0.0, 0.8), red))
cars.append(make_car("car_1", car_matrix(0.0, -1.4, 0.0, 0.8), orange))
cars.append(make_car("car_2", car_matrix(0.0,  1.4, 0.0, 0.8), green))
cars.append(make_car("car_3", car_matrix(0.0,  4.2, 0.0, 0.8), purple))

# ==========================================================================================
# NEW: our own ImGui panel, to add cars WHILE the program is running (as warm-up b did with cubes)
# ==========================================================================================
ui = {"x": 0.0, "z": 0.0, "heading": 0.0, "size": 0.8, "color": [0.85, 0.10, 0.10], "drive": True, "speed": 1.0}
drive_time = 0.0                       # our own clock: it only runs while the box "drive" is ticked

def spawn_car():
    """What the 'Add car' button does."""
    cars.append(make_car("car_%d" % len(cars), car_matrix(ui["x"], ui["z"], ui["heading"], ui["size"]), list(ui["color"]) + [1.0]))
    # The new car has never been sent to the GPU: do it now, for this car only.
    scene.world.traverse_visit(initUpdate, cars[-1]["entity"])

# ==========================================================================================
# WINDOW, EVENTS, CAMERA
# ==========================================================================================
scene.init(imgui=True, windowWidth=winWidth, windowHeight=winHeight, windowTitle="Elements: Car lab - step 8", customImGUIdecorator=ImGUIecssDecorator2, openGLversion=4)

# Everything created so far has to be sent to the GPU once (needs the window, i.e. an OpenGL context)
scene.world.traverse_visit(initUpdate, scene.world.root)

gWindow = scene.renderWindow          # the SDL2 window: pixels, mouse, keyboard, and the camera matrix
gGUI = scene.gContext               # the ImGui layer around it, which also owns the camera

# Camera: where we stand (eye), what we look at (target), which way is up
eye = util.vec(5.0, 13.0, 10.0)
target = util.vec(1.0, 0.0, 0.0)
up = util.vec(0.0, 1.0, 0.0)
gGUI.createViewMatrix(eye, target, up)                  # sets the camera (the mouse and the GUI move it from here)
projMat = util.perspective(50.0, winWidth / winHeight, 0.1, 100.0)

last_time = time.time()

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
    for car in cars:
        for shader, trans in car["parts"]:
            shader.setUniformVariable(key='modelViewProj', value=projMat @ view @ trans.l2world, mat4=True)

    # ---- the cars drive back and forth, all with the same x ----
    drive_time += ui["speed"] * (time.time() - last_time) if ui["drive"] else 0.0
    last_time = time.time()
    x = 3.0 * np.sin(drive_time)
    tire_angle = -np.degrees(x / 0.5)
    for car in cars:
        car["trans"].trs = car["base"] @ util.translate(x, 0.0, 0.0)        # the translation of the WHOLE car comes last
        for tire_trans, tx, tz in car["tires"]:
            tire_trans.trs = tire_matrix(tx, tz, tire_angle)

    scene.world.traverse_visit(renderUpdate, scene.world.root)        # draw everything

    # ---- our panel (after the scene is drawn, so that it is on top) ----
    imgui.set_next_window_position(winWidth - 320, 20, imgui.FIRST_USE_EVER)
    imgui.set_next_window_size(300, 0, imgui.FIRST_USE_EVER)
    imgui.begin("Cars")
    _, ui["x"] = imgui.slider_float("x", ui["x"], -5.0, 5.0)
    _, ui["z"] = imgui.slider_float("z", ui["z"], -5.0, 5.0)
    _, ui["heading"] = imgui.slider_float("heading", ui["heading"], 0.0, 360.0)
    _, ui["size"] = imgui.slider_float("size", ui["size"], 0.3, 1.5)
    _, ui["color"] = imgui.color_edit3("color", *ui["color"])
    if imgui.button("Add car"):
        spawn_car()
    imgui.same_line()
    if imgui.button("Random color"):
        ui["color"] = [random.random(), random.random(), random.random()]
    if imgui.button("Remove last") and len(cars) > 0:
        rootEntity.remove(cars.pop()["entity"])
    _, ui["drive"] = imgui.checkbox("drive back and forth", ui["drive"])
    _, ui["speed"] = imgui.slider_float("speed", ui["speed"], 0.0, 3.0)
    imgui.text("cars in the scene: %d" % len(cars))
    imgui.end()

    scene.render_post()

scene.shutdown()
