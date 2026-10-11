import numpy as np
import imgui

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

from lab_lighting import apply_lighting 

from dataclasses import dataclass

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
terrainSize = 10
vertexTerrain, indexTerrain, colorTerrain = generateTerrain(size=terrainSize)
terrain = scene.world.createEntity(Entity(name="terrain"))
scene.world.addEntityChild(rootEntity, terrain)
terrain_trans = scene.world.addComponent(terrain, BasicTransform(name="terrain_trans", trs=util.translate(terrainSize, 0.0, terrainSize)))
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

# basic colours
white   = [1.0, 1.0, 1.0, 1.0]
black   = [0.0, 0.0, 0.0, 1.0]
red     = [1.0, 0.0, 0.0, 1.0]
green   = [0.0, 1.0, 0.0, 1.0]
blue    = [0.0, 0.0, 1.0, 1.0]
yellow  = [1.0, 1.0, 0.0, 1.0]
cyan    = [0.0, 1.0, 1.0, 1.0]
magenta = [1.0, 0.0, 1.0, 1.0]
orange  = [1.0, 0.65,0.0, 1.0]

# lego
@dataclass
class CellDimensions:
    width:  float = 1.0
    height: float = 1.2
    length: float = 1.0

cellDimensions = CellDimensions()

legos = []
studs = []

def draw_lego(name, n, m, color, i, j, layer):
    halfWidth = cellDimensions.width / 2
    halfHeight = cellDimensions.height / 2
    halfLength = cellDimensions.length / 2

    mainEntity = scene.world.createEntity(Entity(name=name))
    scene.world.addEntityChild(rootEntity, mainEntity)
    scene.world.addComponent(mainEntity, BasicTransform(name="legoTrans", trs=util.translate(halfWidth * (i * 2 + 1) + halfWidth * (n - 1), halfHeight + cellDimensions.height * layer, halfLength * (j * 2 + 1) + halfLength * (m - 1)))) 

    # body
    vertices = np.array([
        [-halfWidth * n, -halfHeight,  halfLength * m, 1.0],
        [-halfWidth * n,  halfHeight,  halfLength * m, 1.0],
        [ halfWidth * n,  halfHeight,  halfLength * m, 1.0],
        [ halfWidth * n, -halfHeight,  halfLength * m, 1.0],
        [-halfWidth * n, -halfHeight, -halfLength * m, 1.0],
        [-halfWidth * n,  halfHeight, -halfLength * m, 1.0],
        [ halfWidth * n,  halfHeight, -halfLength * m, 1.0],
        [ halfWidth * n, -halfHeight, -halfLength * m, 1.0]
    ], dtype=np.float32)

    indices = np.array((
        1, 0, 3,  1, 3, 2,
        2, 3, 7,  2, 7, 6,
        3, 0, 4,  3, 4, 7,
        6, 5, 1,  6, 1, 2,
        4, 5, 6,  4, 6, 7,
        5, 4, 0,  5, 0, 1
    ), np.uint32)

    colors = [color] * len(vertices)

    entity = scene.world.createEntity(Entity(name=name + "Body"))
    scene.world.addEntityChild(mainEntity, entity)
    trans = scene.world.addComponent(entity, BasicTransform(name="trans", trs=util.identity()))
    mesh = scene.world.addComponent(entity, RenderMesh(name="mesh"))

    vertices, indices, colors, normals = norm.generateFlatNormalsMesh(vertices, indices, np.array(colors, dtype=np.float32))
    mesh.vertex_attributes.append(vertices)
    mesh.vertex_attributes.append(colors)
    mesh.vertex_attributes.append(normals)                    
    mesh.vertex_index.append(indices)
    
    vArray = scene.world.addComponent(entity, VertexArray())
    shader = scene.world.addComponent(entity, ShaderGLDecorator(Shader(vertex_import_file=SHADER_DIR / "Phong.vert", fragment_import_file=SHADER_DIR / "BlinnPhong.frag")))
    
    body = {
        "entity": entity,
        "trans":  trans,
        "vArray": vArray,
        "shader": shader
    }

    # studs
    studs = []

    numberOfSegments = 16
    radius = 0.3
    studHeight = 0.225
    halfStudHeight = studHeight / 2

    vertices = []
    for _i in range(numberOfSegments):
        angle = 2 * np.pi * _i / numberOfSegments
        x =  radius * np.cos(angle)
        z = -radius * np.sin(angle)
        vertices.append([x,  halfStudHeight, z, 1.0])
        vertices.append([x, -halfStudHeight, z, 1.0])
    vertices.append([0.0,  halfStudHeight, 0.0, 1.0])
    vertices.append([0.0, -halfStudHeight, 0.0, 1.0])
    vertices = np.array(vertices, dtype=np.float32)

    indices = []
    for _i in range(numberOfSegments):
        _j = (_i + 1) % numberOfSegments
        indices += [2*_i, 2*_i + 1, 2*_j + 1, 2*_i, 2*_j + 1, 2*_j]
        indices += [2*numberOfSegments, 2*_i, 2*_j]
        indices += [2*numberOfSegments + 1, 2*_j + 1, 2*_i + 1]
    indices = np.array(indices, dtype=np.uint32)

    colors = [color] * len(vertices)

    for l in range(m):
        for k in range(n):
            entity = scene.world.createEntity(Entity(name=name + "Stud%d" % (l*n+k)))
            scene.world.addEntityChild(mainEntity, entity)
            trans = scene.world.addComponent(entity, BasicTransform(name="trans", trs=util.translate(-halfWidth * (n - 1) + k * cellDimensions.width, halfHeight + halfStudHeight, -halfLength * (m - 1) + l * cellDimensions.length)))
            mesh = scene.world.addComponent(entity, RenderMesh(name="mesh"))

            vertices, indices, colors, normals = norm.generateFlatNormalsMesh(vertices, indices, np.array(colors, dtype=np.float32))
            mesh.vertex_attributes.append(vertices)
            mesh.vertex_attributes.append(colors)
            mesh.vertex_attributes.append(normals)                    
            mesh.vertex_index.append(indices)

            vArray = scene.world.addComponent(entity, VertexArray())
            shader = scene.world.addComponent(entity, ShaderGLDecorator(Shader(vertex_import_file=SHADER_DIR / "Phong.vert", fragment_import_file=SHADER_DIR / "BlinnPhong.frag")))

            stud = {
                "entity": entity,
                "trans":  trans,
                "vArray": vArray,
                "shader": shader
            }

            studs.append(stud)

    lego = {
        "entity": mainEntity,
        "body":   body,
        "studs":  studs
    }

    legos.append(lego)

    return lego

def spawn_lego(name, n, m, color, i, j, layer):
    lego = draw_lego(name, n, m, list(color) + [1.0], i, j, layer)

    body  = lego["body"]
    scene.world.traverse_visit(initUpdate, body["entity"])

    studs = lego["studs"]

    for stud in studs:
        scene.world.traverse_visit(initUpdate, stud["entity"])

ui = {"n": 1, "m": 1, "color": [1.0, 1.0, 1.0], "i": 0, "j": 0, "layer": 0}

draw_lego("lego0", 1, 1, red,    0, 0, 0)
draw_lego("lego0", 4, 2, blue,   2, 0, 0)
draw_lego("lego0", 2, 2, yellow, 2, 0, 1)

# ==========================================================================================
# WINDOW, EVENTS, CAMERA
# ==========================================================================================
scene.init(imgui=True, windowWidth=winWidth, windowHeight=winHeight, windowTitle="Exercise-1a", customImGUIdecorator=ImGUIecssDecorator2, openGLversion=4)

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

    # add_lego panel
    imgui.set_next_window_position(winWidth - 320, 20, imgui.FIRST_USE_EVER)
    imgui.set_next_window_size(300, 0, imgui.FIRST_USE_EVER)
    imgui.begin("Add legos", closable=True)
    imgui.text_colored("Size", 1.0, 1.0, 1.0)
    _, ui["n"] = imgui.slider_int("n (x)", ui["n"],  1, 8)
    _, ui["m"] = imgui.slider_int("m (z)", ui["m"],  1, 8)
    _, ui["color"] = imgui.color_edit3("color", *ui["color"])
    imgui.separator()
    imgui.text_colored("Position", 1.0, 1.0, 1.0)
    _, ui["i"] = imgui.slider_int("i (x)", ui["i"], 0, 15)
    _, ui["j"] = imgui.slider_int("j (z)", ui["j"], 0, 15)
    _, ui["layer"] = imgui.slider_int("layer", ui["layer"],  0, 5)
    imgui.separator()
    if imgui.button("Add"):
        spawn_lego("lego%d" % len(legos), ui["n"], ui["m"], ui["color"], ui["i"], ui["j"], ui["layer"])
    imgui.same_line()
    if imgui.button("Delete last") and len(legos) > 0:
        lego = legos.pop()
        entity = lego["entity"]

        body = lego["body"]
        entity.remove(body["entity"])

        studs = lego["studs"]
        for stud in studs:
            entity.remove(stud["entity"])

        rootEntity.remove(entity)

    imgui.text("legos in the scene: %d" % len(legos))
    imgui.end()
    

    scene.world.traverse_visit(transUpdate, scene.world.root)         # compute the world matrix (l2world) of every transform
    view = gWindow._myCamera                                           # the camera: moves with the mouse / the GUI
    terrain_shader.setUniformVariable(key='modelViewProj', value=projMat @ view @ terrain_trans.l2world, mat4=True)
    axes_shader.setUniformVariable(key='modelViewProj', value=projMat @ view @ axes_trans.l2world, mat4=True)

    for lego in legos:
        body = lego["body"]
        body["shader"].setUniformVariable(key='modelViewProj', value=projMat @ view @ body["trans"].l2world, mat4=True)
        apply_lighting(body["shader"], body["trans"].l2world, gWindow._cameraEye)

        studs = lego["studs"]
        for stud in studs:
            stud["shader"].setUniformVariable(key='modelViewProj', value=projMat @ view @ stud["trans"].l2world, mat4=True)
            apply_lighting(stud["shader"], stud["trans"].l2world, gWindow._cameraEye)



    scene.world.traverse_visit(renderUpdate, scene.world.root)        # draw everything
    scene.render_post()

scene.shutdown()
