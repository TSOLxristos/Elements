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
vertexTerrain, indexTerrain, colorTerrain = generateTerrain(size = terrainSize)
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

# colours
stone      = [0.80, 0.77, 0.70, 1.0]
marble     = [0.93, 0.92, 0.88, 1.0]
terracotta = [0.72, 0.32, 0.20, 1.0]
bronze     = [0.55, 0.36, 0.18, 1.0]

cubeWidth  = 1.0
cubeHeight = 1.0
cubeLength = 1.0
halfCubeWidth  = cubeWidth  / 2
halfCubeHeight = cubeHeight / 2
halfCubeLength = cubeLength / 2
cubeVertices = np.array([
    [-halfCubeWidth, -halfCubeHeight,  halfCubeLength, 1.0],
    [-halfCubeWidth,  halfCubeHeight,  halfCubeLength, 1.0],
    [ halfCubeWidth,  halfCubeHeight,  halfCubeLength, 1.0],
    [ halfCubeWidth, -halfCubeHeight,  halfCubeLength, 1.0],
    [-halfCubeWidth, -halfCubeHeight, -halfCubeLength, 1.0],
    [-halfCubeWidth,  halfCubeHeight, -halfCubeLength, 1.0],
    [ halfCubeWidth,  halfCubeHeight, -halfCubeLength, 1.0],
    [ halfCubeWidth, -halfCubeHeight, -halfCubeLength, 1.0]
], dtype=np.float32)

cubeIndices = np.array((
    1, 0, 3,  1, 3, 2,
    2, 3, 7,  2, 7, 6,
    3, 0, 4,  3, 4, 7,
    6, 5, 1,  6, 1, 2,
    4, 5, 6,  4, 6, 7,
    5, 4, 0,  5, 0, 1
), np.uint32)

cylinderRadius = 1.0
cylinderHeight = 1.0
halfCylinderHeight = cylinderHeight / 2
numberOfSegments = 16
cylinderVertices = []
for i in range(numberOfSegments):
    angle = 2 * np.pi * i / numberOfSegments
    x =  cylinderRadius * np.cos(angle)
    z = -cylinderRadius * np.sin(angle)
    cylinderVertices.append([x,  halfCylinderHeight, z, 1.0])
    cylinderVertices.append([x, -halfCylinderHeight, z, 1.0])
cylinderVertices.append([0.0,  halfCylinderHeight, 0.0, 1.0])
cylinderVertices.append([0.0, -halfCylinderHeight, 0.0, 1.0])
cylinderVertices = np.array(cylinderVertices, dtype=np.float32)

cylinderIndices = []
for i in range(numberOfSegments):
    j = (i + 1) % numberOfSegments
    cylinderIndices += [2*i, 2*i + 1, 2*j + 1, 2*i, 2*j + 1, 2*j]
    cylinderIndices += [2*numberOfSegments, 2*i, 2*j]
    cylinderIndices += [2*numberOfSegments + 1, 2*j + 1, 2*i + 1]
cylinderIndices = np.array(cylinderIndices, dtype=np.uint32)

triangularPrismWidth  = 1.0
triangularPrismHeight = 1.0
triangularPrismLength = 1.0
halfTriangularPrismWidth  = triangularPrismWidth  / 2
halfTriangularPrismHeight = triangularPrismHeight / 2
halfTriangularPrismLength = triangularPrismLength / 2

triangularPrismVertices = np.array([
    [-halfCubeWidth, -halfCubeHeight,  halfCubeLength, 1.0],
    [-halfCubeWidth,  halfCubeHeight,  halfCubeLength, 1.0],
    [ halfCubeWidth, -halfCubeHeight,  halfCubeLength, 1.0],
    [-halfCubeWidth, -halfCubeHeight, -halfCubeLength, 1.0],
    [-halfCubeWidth,  halfCubeHeight, -halfCubeLength, 1.0],
    [ halfCubeWidth, -halfCubeHeight, -halfCubeLength, 1.0]
], dtype=np.float32)

triangularPrismIndices = np.array((
    0, 2, 1, 3, 4, 5,
    0, 3, 5, 0, 5, 2,
    1, 2, 5, 1, 5, 4,
    0, 1, 4, 0, 4, 3
), np.uint32)

cubeEntity = scene.world.createEntity(Entity(name="cube"))
scene.world.addEntityChild(rootEntity, cubeEntity)
cubeTrans = scene.world.addComponent(cubeEntity, BasicTransform(name="trans", trs = util.translate(7.0, 0.5, 16)))
cubeMesh  = scene.world.addComponent(cubeEntity, RenderMesh(name="mesh"))

cubeVertices, cubeIndices, cubeColors, cubeNormals = norm.generateFlatNormalsMesh(cubeVertices, cubeIndices, np.array([stone] * len(cubeVertices), dtype=np.float32))
cubeMesh.vertex_attributes.append(cubeVertices)
cubeMesh.vertex_attributes.append(cubeColors)
cubeMesh.vertex_attributes.append(cubeNormals)                    
cubeMesh.vertex_index.append(cubeIndices)

cubeVertexArray = scene.world.addComponent(cubeEntity, VertexArray())
cubeShader      = scene.world.addComponent(cubeEntity, ShaderGLDecorator(Shader(vertex_import_file=SHADER_DIR / "Phong.vert", fragment_import_file=SHADER_DIR / "BlinnPhong.frag")))

cylinderEntity = scene.world.createEntity(Entity(name="cylinder"))
scene.world.addEntityChild(rootEntity, cylinderEntity)
cylinderTrans = scene.world.addComponent(cylinderEntity, BasicTransform(name="trans", trs = util.translate(10, 0.5, 16)))
cylinderMesh  = scene.world.addComponent(cylinderEntity, RenderMesh(name="mesh"))

cylinderVertices, cylinderIndices, cylinderColors, cylinderNormals = norm.generateFlatNormalsMesh(cylinderVertices, cylinderIndices, np.array([marble] * len(cylinderVertices), dtype=np.float32))
cylinderMesh.vertex_attributes.append(cylinderVertices)
cylinderMesh.vertex_attributes.append(cylinderColors)
cylinderMesh.vertex_attributes.append(cylinderNormals)                    
cylinderMesh.vertex_index.append(cylinderIndices)

cylinderVertexArray = scene.world.addComponent(cylinderEntity, VertexArray())
cylinderShader      = scene.world.addComponent(cylinderEntity, ShaderGLDecorator(Shader(vertex_import_file=SHADER_DIR / "Phong.vert", fragment_import_file=SHADER_DIR / "BlinnPhong.frag")))

triangularPrismEntity = scene.world.createEntity(Entity(name="triangularPrism"))
scene.world.addEntityChild(rootEntity, triangularPrismEntity)
triangularPrismTrans = scene.world.addComponent(triangularPrismEntity, BasicTransform(name="trans", trs = util.translate(13, 0.5, 16)))
triangularPrismMesh  = scene.world.addComponent(triangularPrismEntity, RenderMesh(name="mesh"))

triangularPrismVertices, triangularPrismIndices, triangularPrismColors, triangularPrismNormals = norm.generateFlatNormalsMesh(triangularPrismVertices, triangularPrismIndices, np.array([terracotta] * len(triangularPrismVertices), dtype=np.float32))
triangularPrismMesh.vertex_attributes.append(triangularPrismVertices)
triangularPrismMesh.vertex_attributes.append(triangularPrismColors)
triangularPrismMesh.vertex_attributes.append(triangularPrismNormals)                    
triangularPrismMesh.vertex_index.append(triangularPrismIndices)

triangularPrismVertexArray = scene.world.addComponent(triangularPrismEntity, VertexArray())
triangularPrismShader      = scene.world.addComponent(triangularPrismEntity, ShaderGLDecorator(Shader(vertex_import_file=SHADER_DIR / "Phong.vert", fragment_import_file=SHADER_DIR / "BlinnPhong.frag")))

frontParthenonEntity = scene.world.createEntity(Entity(name="Front Of A Parthenon"))
scene.world.addEntityChild(rootEntity, frontParthenonEntity)
scene.world.addComponent(frontParthenonEntity, BasicTransform(name="trans", trs=util.translate(terrainSize, 0.0, terrainSize))) 

baseEntity = scene.world.createEntity(Entity(name="base"))
scene.world.addEntityChild(frontParthenonEntity, baseEntity)
baseTrans = scene.world.addComponent(baseEntity, BasicTransform(name="trans", trs = util.translate(0.0, 0.25, 0.0) @ util.scale(10.0, 0.5, 3.0)))
baseMesh = scene.world.addComponent(baseEntity, RenderMesh(name="mesh"))

baseVertices, baseIndices, baseColors, baseNormals = norm.generateFlatNormalsMesh(cubeVertices, cubeIndices, np.array([stone] * len(cubeVertices), dtype=np.float32))
baseMesh.vertex_attributes.append(baseVertices)
baseMesh.vertex_attributes.append(baseColors)
baseMesh.vertex_attributes.append(baseNormals)                    
baseMesh.vertex_index.append(baseIndices)

baseVertexArray = scene.world.addComponent(baseEntity, VertexArray())
baseShader = scene.world.addComponent(baseEntity, ShaderGLDecorator(Shader(vertex_import_file=SHADER_DIR / "Phong.vert", fragment_import_file=SHADER_DIR / "BlinnPhong.frag")))

leftDoorVertices = cubeVertices.copy()
leftDoorVertices[0::1, 0] -= halfCubeWidth

leftDoorEntity = scene.world.createEntity(Entity(name="leftDoor"))
scene.world.addEntityChild(frontParthenonEntity, leftDoorEntity)
leftDoorTranslate = util.translate(2.95, 3.0, 0.0)
leftDoorScale     = util.scale(2.9, 5.0, 0.2)
leftDoorTrans = scene.world.addComponent(leftDoorEntity, BasicTransform(name="trans", trs = leftDoorTranslate @ leftDoorScale))
leftDoorMesh  = scene.world.addComponent(leftDoorEntity, RenderMesh(name="mesh"))

leftDoorVertices, leftDoorIndices, leftDoorColors, leftDoorNormals = norm.generateFlatNormalsMesh(cubeVertices @ util.translate(-halfCubeWidth, 0.0, 0.0).T, cubeIndices, np.array([bronze] * len(cubeVertices), dtype=np.float32))
leftDoorMesh.vertex_attributes.append(leftDoorVertices)
leftDoorMesh.vertex_attributes.append(leftDoorColors)
leftDoorMesh.vertex_attributes.append(leftDoorNormals)                    
leftDoorMesh.vertex_index.append(leftDoorIndices)

leftDoorVertexArray = scene.world.addComponent(leftDoorEntity, VertexArray())
leftDoorShader = scene.world.addComponent(leftDoorEntity, ShaderGLDecorator(Shader(vertex_import_file=SHADER_DIR / "Phong.vert", fragment_import_file=SHADER_DIR / "BlinnPhong.frag")))

rightDoorEntity = scene.world.createEntity(Entity(name="rightDoor"))
scene.world.addEntityChild(frontParthenonEntity, rightDoorEntity)
rightDoorTranslate = util.translate(-2.95, 3.0, 0.0)
rightDoorScale     = util.scale(2.9, 5.0, 0.2)
rightDoorTrans = scene.world.addComponent(rightDoorEntity, BasicTransform(name="trans", trs = util.translate(-2.95, 3.0, 0.0) @ util.scale(2.9, 5.0, 0.2)))
rightDoorMesh  = scene.world.addComponent(rightDoorEntity, RenderMesh(name="mesh"))

rightDoorVertices, rightDoorIndices, rightDoorColors, rightDoorNormals = norm.generateFlatNormalsMesh(cubeVertices @ util.translate(halfCubeWidth, 0.0, 0.0).T, cubeIndices, np.array([bronze] * len(cubeVertices), dtype=np.float32))
rightDoorMesh.vertex_attributes.append(rightDoorVertices)
rightDoorMesh.vertex_attributes.append(rightDoorColors)
rightDoorMesh.vertex_attributes.append(rightDoorNormals)                    
rightDoorMesh.vertex_index.append(rightDoorIndices)

rightDoorVertexArray = scene.world.addComponent(rightDoorEntity, VertexArray())
rightDoorShader = scene.world.addComponent(rightDoorEntity, ShaderGLDecorator(Shader(vertex_import_file=SHADER_DIR / "Phong.vert", fragment_import_file=SHADER_DIR / "BlinnPhong.frag")))

leftColumnEntity = scene.world.createEntity(Entity(name="leftColumn"))
scene.world.addEntityChild(frontParthenonEntity, leftColumnEntity)
leftColumnTrans = scene.world.addComponent(leftColumnEntity, BasicTransform(name="trans", trs = util.translate(3.5, 3.0, 0.0) @ util.scale(0.5, 5, 0.5)))
leftColumnMesh = scene.world.addComponent(leftColumnEntity, RenderMesh(name="mesh"))

leftColumnVertices, leftColumnIndices, leftColumnColors, leftColumnNormals = norm.generateFlatNormalsMesh(cylinderVertices, cylinderIndices, np.array([marble] * len(cylinderVertices), dtype=np.float32))
leftColumnMesh.vertex_attributes.append(leftColumnVertices)
leftColumnMesh.vertex_attributes.append(leftColumnColors)
leftColumnMesh.vertex_attributes.append(leftColumnNormals)                    
leftColumnMesh.vertex_index.append(leftColumnIndices)

leftColumnVertexArray = scene.world.addComponent(leftColumnEntity, VertexArray())
leftColumnShader = scene.world.addComponent(leftColumnEntity, ShaderGLDecorator(Shader(vertex_import_file=SHADER_DIR / "Phong.vert", fragment_import_file=SHADER_DIR / "BlinnPhong.frag")))

rightColumnEntity = scene.world.createEntity(Entity(name="rightColumn"))
scene.world.addEntityChild(frontParthenonEntity, rightColumnEntity)
rightColumnTrans = scene.world.addComponent(rightColumnEntity, BasicTransform(name="trans", trs = util.translate(-3.5, 3.0, 0.0) @ util.scale(0.5, 5, 0.5)))
rightColumnMesh = scene.world.addComponent(rightColumnEntity, RenderMesh(name="mesh"))

rightColumnVertices, rightColumnIndices, rightColumnColors, rightColumnNormals = norm.generateFlatNormalsMesh(cylinderVertices, cylinderIndices, np.array([marble] * len(cylinderVertices), dtype=np.float32))
rightColumnMesh.vertex_attributes.append(rightColumnVertices)
rightColumnMesh.vertex_attributes.append(rightColumnColors)
rightColumnMesh.vertex_attributes.append(rightColumnNormals)                    
rightColumnMesh.vertex_index.append(rightColumnIndices)

rightColumnVertexArray = scene.world.addComponent(rightColumnEntity, VertexArray())
rightColumnShader = scene.world.addComponent(rightColumnEntity, ShaderGLDecorator(Shader(vertex_import_file=SHADER_DIR / "Phong.vert", fragment_import_file=SHADER_DIR / "BlinnPhong.frag")))

leftRoofEntity = scene.world.createEntity(Entity(name="leftRoof"))
scene.world.addEntityChild(frontParthenonEntity, leftRoofEntity)
leftRoofTrans = scene.world.addComponent(leftRoofEntity, BasicTransform(name="trans", trs = util.translate(2.75, 6.5, 0.0) @ util.scale(5.5, 2, 3.4)))
leftRoofMesh = scene.world.addComponent(leftRoofEntity, RenderMesh(name="mesh"))

leftRoofVertices, leftRoofIndices, leftRoofColors, leftRoofNormals = norm.generateFlatNormalsMesh(triangularPrismVertices, triangularPrismIndices, np.array([terracotta] * len(triangularPrismVertices), dtype=np.float32))
leftRoofMesh.vertex_attributes.append(leftRoofVertices)
leftRoofMesh.vertex_attributes.append(leftRoofColors)
leftRoofMesh.vertex_attributes.append(leftRoofNormals)
leftRoofMesh.vertex_index.append(leftRoofIndices)

leftRoofVertexArray = scene.world.addComponent(leftRoofEntity, VertexArray())
leftRoofShader = scene.world.addComponent(leftRoofEntity, ShaderGLDecorator(Shader(vertex_import_file=SHADER_DIR / "Phong.vert", fragment_import_file=SHADER_DIR / "BlinnPhong.frag")))

rightRoofEntity = scene.world.createEntity(Entity(name="rightRoof"))
scene.world.addEntityChild(frontParthenonEntity, rightRoofEntity)
rightRoofTrans = scene.world.addComponent(rightRoofEntity, BasicTransform(name="trans", trs = util.translate(-2.75, 6.5, 0.0) @ util.rotate((0, 1, 0), 180) @ util.scale(5.5, 2, 3.4)))
rightRoofMesh = scene.world.addComponent(rightRoofEntity, RenderMesh(name="mesh"))

rightRoofVertices, rightRoofIndices, rightRoofColors, rightRoofNormals = norm.generateFlatNormalsMesh(triangularPrismVertices, triangularPrismIndices, np.array([terracotta] * len(triangularPrismVertices), dtype=np.float32))
rightRoofMesh.vertex_attributes.append(rightRoofVertices)
rightRoofMesh.vertex_attributes.append(rightRoofColors)
rightRoofMesh.vertex_attributes.append(rightRoofNormals)
rightRoofMesh.vertex_index.append(rightRoofIndices)

rightRoofVertexArray = scene.world.addComponent(rightRoofEntity, VertexArray())
rightRoofShader = scene.world.addComponent(rightRoofEntity, ShaderGLDecorator(Shader(vertex_import_file=SHADER_DIR / "Phong.vert", fragment_import_file=SHADER_DIR / "BlinnPhong.frag")))

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
start_time = time.time()

running = True
while running:
    running = scene.render()                                          # window, events, GUI

    scene.world.traverse_visit(transUpdate, scene.world.root)         # compute the world matrix (l2world) of every transform
    view = gWindow._myCamera                                           # the camera: moves with the mouse / the GUI
    terrain_shader.setUniformVariable(key='modelViewProj', value=projMat @ view @ terrain_trans.l2world, mat4=True)
    axes_shader.setUniformVariable(key='modelViewProj', value=projMat @ view @ axes_trans.l2world, mat4=True)

    cubeShader.setUniformVariable(key='modelViewProj', value=projMat @ view @ cubeTrans.l2world, mat4=True)
    apply_lighting(cubeShader, cubeTrans.l2world, gWindow._cameraEye)
    cylinderShader.setUniformVariable(key='modelViewProj', value=projMat @ view @ cylinderTrans.l2world, mat4=True)
    apply_lighting(cylinderShader, cylinderTrans.l2world, gWindow._cameraEye)
    triangularPrismShader.setUniformVariable(key='modelViewProj', value=projMat @ view @ triangularPrismTrans.l2world, mat4=True)
    apply_lighting(triangularPrismShader, triangularPrismTrans.l2world, gWindow._cameraEye)

    baseShader.setUniformVariable(key='modelViewProj', value=projMat @ view @ baseTrans.l2world, mat4=True)
    apply_lighting(baseShader, baseTrans.l2world, gWindow._cameraEye)

    t = time.time() - start_time
    angle = 45 + np.sin(t*2) * 45
    leftDoorTrans.trs  = leftDoorTranslate  @ util.rotate((0, 1, 0),  angle) @ leftDoorScale
    rightDoorTrans.trs = rightDoorTranslate @ util.rotate((0, 1, 0), -angle) @ rightDoorScale

    leftDoorShader.setUniformVariable(key='modelViewProj', value=projMat @ view @ leftDoorTrans.l2world, mat4=True)
    apply_lighting(leftDoorShader, leftDoorTrans.l2world, gWindow._cameraEye)
    rightDoorShader.setUniformVariable(key='modelViewProj', value=projMat @ view @ rightDoorTrans.l2world, mat4=True)
    apply_lighting(rightDoorShader, rightDoorTrans.l2world, gWindow._cameraEye)

    leftColumnShader.setUniformVariable(key='modelViewProj', value=projMat @ view @ leftColumnTrans.l2world, mat4=True)
    apply_lighting(leftColumnShader, leftColumnTrans.l2world, gWindow._cameraEye)
    rightColumnShader.setUniformVariable(key='modelViewProj', value=projMat @ view @ rightColumnTrans.l2world, mat4=True)
    apply_lighting(rightColumnShader, rightColumnTrans.l2world, gWindow._cameraEye)
    leftRoofShader.setUniformVariable(key='modelViewProj', value=projMat @ view @ leftRoofTrans.l2world, mat4=True)
    apply_lighting(leftRoofShader, leftRoofTrans.l2world, gWindow._cameraEye)
    rightRoofShader.setUniformVariable(key='modelViewProj', value=projMat @ view @ rightRoofTrans.l2world, mat4=True)
    apply_lighting(rightRoofShader, rightRoofTrans.l2world, gWindow._cameraEye)

    scene.world.traverse_visit(renderUpdate, scene.world.root)        # draw everything
    scene.render_post()

# scene.shutdown()# ---- NEW: turn the tires, every frame ----
# t = time.time() - start_time                                       # seconds since the program started
# for tire_trans, x, z in tires:
#     tire_trans.trs = tire_matrix(x, z, 120.0 * t)                  # spin about its own axis, THEN lay it down and move it
# scene.world.traverse_visit(renderUpdate, scene.world.root)        # draw everything
# scene.render_post()

# def tire_matrix(x, z, angle):
#    return util.translate(x, 0.5, z) @ util.rotate((1, 0, 0), 90) @ util.rotate((0, 1, 0), angle) @ util.scale(0.5, 0.4, 0.5)