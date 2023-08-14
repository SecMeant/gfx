#include <stdio.h>

#include <array>
#include <expected>
#include <memory>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/gl.h>
#include <GL/glext.h>

#include <glm/glm.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <mipc/file.h>

const int win_width = 512;
const int win_height = 512;

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

static glm::vec3 position = { 0.0f, 0.0f, 0.0f };
static glm::mat4x4 projection_mat;

struct
{
    glm::vec3 eye;
    glm::vec3 center;
    glm::vec3 up;
} static camera;

const GLfloat vpoint[] = {
    -0.5f,  0.5f, 0.0f,
     0.5f,  0.5f, 0.0f,
     0.5f, -0.5f, 0.0f,

    -0.5f,  0.5f, 0.0f,
     0.5f, -0.5f, 0.0f,
    -0.5f, -0.5f, 0.0f,
};

static GLuint
init()
{
    glClearColor(0.0, 0.0, 0.0, 1.0);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    glViewport(0, 0, win_width, win_height);
    projection_mat = glm::perspective(3.141592653589793f/2.0f, (float)win_width / (float)win_height, 0.1f, 10.0f);

    camera.eye = { 0.0f, 0.0f, -1.0f };
    camera.center = { 0.0f, 0.0f, 0.0f };
    camera.up = { 0.0f, 1.0f, 0.0f };

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

    return shader_program;
}

static void
render(GLuint shader_program, glm::mat4x4 model, glm::mat4x4 view, glm::mat4x4 projection)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUniformMatrix4fv(glGetUniformLocation(shader_program, "model"), 1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix4fv(glGetUniformLocation(shader_program, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(shader_program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

#if defined(DRAW_OUTLINE)
    glDrawArrays(GL_LINE_LOOP, 0, 6);
#else
    glDrawArrays(GL_TRIANGLES, 0, 6);
#endif
}

// Returns if should exit the render loop
static int
handle_key(GLFWwindow *window)
{
    if (glfwWindowShouldClose(window))
        return 1;

    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        return 1;

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        camera.eye.z += 0.125f;
        camera.center.z += 0.125f;
    }

    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        camera.eye.z -= 0.125f;
        camera.center.z -= 0.125f;
    }

    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        camera.eye.x += 0.125f;
        camera.center.x += 0.125f;
    }

    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        camera.eye.x -= 0.125f;
        camera.center.x -= 0.125f;
    }

    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
        camera.eye.y += 0.125f;
        camera.center.y += 0.125f;
    }

    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) {
        camera.eye.y -= 0.125f;
        camera.center.y -= 0.125f;
    }

    return 0;
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

    GLFWwindow *window = glfwCreateWindow(win_width, win_height, "triangle", NULL, NULL);

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

    GLuint shader_program = init();

    glm::mat4 object_mat_rot = glm::mat4(1.0);

    while (1) {
        if (handle_key(window))
            break;

        render(shader_program, object_mat_rot, glm::lookAt(camera.eye, camera.center, camera.up), projection_mat);
        glfwSwapBuffers(window);
        glfwSwapInterval(1);
        glfwPollEvents();
    }

    return 0;
}