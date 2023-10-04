#version 330 core

in vec3 vpoint;
uniform mat4 view;
uniform mat4 projection;

out float red;
out float green;
out float blue;

void main()
{
    // gl_Position = vec4(vpoint - position, 1.0);
    gl_Position = projection*view*vec4(vpoint, 1.0);
    red = blue = 0.0;
    green = 1.0;

    //int id = gl_VertexID % 3;

    //if (id == 0) {
    //    red = 1.0;
    //}
    //else if (id == 1) {
    //    green = 1.0;
    //}
    //else if (id == 2) {
    //    blue = 1.0;
    //}
    //else {
    //    red = green = blue = 1.0;
    //}
}
