"""
GIVEN to the students -- the light.

Lights are the topic of the next lecture, so for now this file is a black box. All you need to know:

    apply_lighting(shader, model_matrix, camera_position)

sets the uniform variables of the Blinn-Phong shader (the light, the material) for ONE object. Call it once per
object, every frame, next to the line that sets 'modelViewProj'.
"""
import Elements.pyECSS.math_utilities as util

light_position = util.vec(2.0, 8.0, 5.0)
light_color = util.vec(1.0, 1.0, 1.0)
light_intensity = 0.9
ambient_color = util.vec(1.0, 1.0, 1.0)
ambient_strength = 0.35
shininess = 0.4
specular_exponent = 48.0


def apply_lighting(shader, model_matrix, camera_position):
    shader.setUniformVariable(key='model', value=model_matrix, mat4=True)
    shader.setUniformVariable(key='ambientColor', value=ambient_color, float3=True)
    shader.setUniformVariable(key='ambientStr', value=ambient_strength, float1=True)
    shader.setUniformVariable(key='viewPos', value=util.vec(camera_position), float3=True)
    shader.setUniformVariable(key='lightPos', value=light_position, float3=True)
    shader.setUniformVariable(key='lightColor', value=light_color, float3=True)
    shader.setUniformVariable(key='lightIntensity', value=light_intensity, float1=True)
    shader.setUniformVariable(key='shininess', value=shininess, float1=True)
    shader.setUniformVariable(key='specularExponent', value=specular_exponent, float1=True)
