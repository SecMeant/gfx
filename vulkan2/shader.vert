#version 450

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 fragColor;


//vec2 positions[6] = vec2[](
//    vec2(-0.35, -0.25),
//    vec2(-0.1, 0.25),
//    vec2(-0.6, 0.25),
//    vec2(0.35, -0.25),
//    vec2(0.6, 0.25),
//    vec2(0.1, 0.25)
//);

//vec3 colors[6] = vec3[](
//    vec3(1.0, 0.0, 0.0),
//    vec3(0.0, 1.0, 0.0),
//    vec3(0.0, 0.0, 1.0),
//    vec3(0.0, 0.0, 1.0),
//    vec3(0.0, 1.0, 0.0),
//    vec3(1.0, 0.0, 0.0)
//);

void main() {
    gl_Position = vec4(inPosition, 0.0, 1.0);
    fragColor = inColor;
}
