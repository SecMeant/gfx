#include <stdio.h>

#include <array>
#include <expected>
#include <memory>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/gl.h>
#include <GL/glext.h>

#include <mipc/file.h>

static std::expected<GLuint, int>
load_shader(const char *path, GLuint kind)
{
    mipc::finbuf f(path);

    if (!f) {
        fprintf(stderr, "load_shader: failed to open %s\n", path);
        return std::unexpected(1);
    }

    const GLchar *shader_sources[1] = { f.begin() };
    const GLint lengths[1] = { static_cast<GLint>(f.size()) }; // TODO: overflow when casting shader source length

    static_assert(std::size(shader_sources) == std::size(lengths));

    GLuint shader = glCreateShader(kind);
    glShaderSource(shader, std::size(shader_sources), shader_sources, lengths);
    glCompileShader(shader);

    return shader;
}

static std::expected<GLuint, int>
load_shaders(const char *vertex_shader_path, const char *fragment_shader_path)
{
    /*
     * Dont clean the shaders on the failure path - let the kernel do the
     * cleaning. In case of error we exit soon anyway.
     */
    GLuint vshader_id = load_shader(vertex_shader_path, GL_VERTEX_SHADER).value();
    GLuint fshader_id = load_shader(fragment_shader_path, GL_FRAGMENT_SHADER).value();
    GLint success = GL_FALSE;

    glGetShaderiv(vshader_id, GL_COMPILE_STATUS, &success);
    if (!success) {
        std::array<char, 255> log;
        glGetShaderInfoLog(vshader_id, log.size(), NULL, log.data());
        fprintf(stderr, "GLSL, vertex: %s\n", log.data());

        return std::unexpected(1);
    }

    glGetShaderiv(fshader_id, GL_COMPILE_STATUS, &success);
    if (!success) {
        std::array<char, 255> log;
        glGetShaderInfoLog(fshader_id, log.size(), NULL, log.data());
        fprintf(stderr, "GLSL, fragment: %s\n", log.data());

        return std::unexpected(1);
    }

    GLuint program_id = glCreateProgram();
    glAttachShader(program_id, vshader_id);
    glAttachShader(program_id, fshader_id);
    glLinkProgram(program_id);

    glGetProgramiv(program_id, GL_LINK_STATUS, &success);
    if (!success) {
        std::array<char, 255> log;
        glGetShaderInfoLog(program_id, log.size(), NULL, log.data());
        fprintf(stderr, "GLSL, program: %s\n", log.data());

        return std::unexpected(1);
    }

    return program_id;
}

const GLfloat vpoint[] = {
    -1.0f, -1.0f, 0.0f,
     1.0f, -1.0f, 0.0f,
     0.0f,  1.0f, 0.0f,
};

static void
init()
{
    glClearColor(0.0, 0.0, 0.0, 1.0);

    GLuint shader_program = load_shaders("vshader.glsl", "fshader.glsl").value();

    glUseProgram(shader_program);

    GLuint vao_id;
    glGenVertexArrays(1, &vao_id);
    glBindVertexArray(vao_id);

    GLuint vbo_id;
    glGenBuffers(1, &vbo_id);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_id);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vpoint), vpoint, GL_STATIC_DRAW);

    GLuint vpoint_id = glGetAttribLocation(shader_program, "vpoint");
    glEnableVertexAttribArray(vpoint_id);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_id);
    glVertexAttribPointer(vpoint_id, 3, GL_FLOAT, GL_FALSE, 0, NULL);
}

static void
render()
{
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

int
main()
{
    if (!glfwInit()) {
        fprintf(stderr, "glfw\n");
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow *window = glfwCreateWindow(512, 512, "triangle", NULL, NULL);

    if (!window) {
        fprintf(stderr, "window\n");
        return 1;
    }

    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;
    if (const int glew_init_result = glewInit(); glew_init_result) {
        fprintf(stderr, "glew\n");
        return 1;
    }

    init();

    while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && !glfwWindowShouldClose(window)) {
        render();
        glfwSwapBuffers(window);
        glfwSwapInterval(1);
        glfwPollEvents();
    }

    return 0;
}