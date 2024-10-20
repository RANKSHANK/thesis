#version 330

uniform mat4 MVP;

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_texcoord;

out vec3 v_vert;
out vec3 v_norm;
out vec2 v_tex;

void main() {
    gl_Position = MVP * vec4(in_position, 1.0);
    v_vert = in_position;
    v_norm = in_normal;
    v_tex = in_texcoord;
}
