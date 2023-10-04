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
#include <glm/gtx/rotate_vector.hpp>

#include <mipc/file.h>

const int win_width = 1920/2;
const int win_height = 1080;

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
    glm::vec3 up;

    float yaw;
    float pitch;
} static camera;

const glm::vec3 vpoint[] = {
    /* Front face */
    { -0.5f,  0.5f, 0.0f, },
    {  0.5f,  0.5f, 0.0f, },
    {  0.5f, -0.5f, 0.0f, },

    { -0.5f,  0.5f, 0.0f, },
    {  0.5f, -0.5f, 0.0f, },
    { -0.5f, -0.5f, 0.0f, },


    /* Left face */
    {  0.5f, -0.5f, 0.0f, },
    {  0.5f, -0.5f, 1.0f, },
    {  0.5f,  0.5f, 0.0f, },

    {  0.5f,  0.5f, 0.0f, },
    {  0.5f,  0.5f, 1.0f, },
    {  0.5f, -0.5f, 1.0f, },


    /* Right face */
    { -0.5f, -0.5f, 0.0f, },
    { -0.5f, -0.5f, 1.0f, },
    { -0.5f,  0.5f, 0.0f, },

    { -0.5f,  0.5f, 0.0f, },
    { -0.5f,  0.5f, 1.0f, },
    { -0.5f, -0.5f, 1.0f, },


    /* Back face */
    { -0.5f,  0.5f, 1.0f, },
    {  0.5f,  0.5f, 1.0f, },
    {  0.5f, -0.5f, 1.0f, },

    { -0.5f,  0.5f, 1.0f, },
    {  0.5f, -0.5f, 1.0f, },
    { -0.5f, -0.5f, 1.0f, },


    /* Up face */
    { -0.5f,  0.5f, 0.0f, },
    {  0.5f,  0.5f, 0.0f, },
    {  0.5f,  0.5f, 1.0f, },

    {  0.5f,  0.5f, 1.0f, },
    { -0.5f,  0.5f, 1.0f, },
    { -0.5f,  0.5f, 0.0f, },


    /* Down face */
    { -0.5f, -0.5f, 0.0f, },
    {  0.5f, -0.5f, 0.0f, },
    {  0.5f, -0.5f, 1.0f, },

    {  0.5f, -0.5f, 1.0f, },
    { -0.5f, -0.5f, 1.0f, },
    { -0.5f, -0.5f, 0.0f, },
};

static GLuint
init()
{
    glClearColor(0.0, 0.0, 0.0, 1.0);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    glViewport(0, 0, win_width, win_height);
    projection_mat = glm::perspective(glm::radians(90.0f), (GLfloat)win_width / (GLfloat)win_height, 0.125f, 150.0f);

    camera.eye = { -1.0f, 1.0f, -2.0f };
    camera.up = { 0.0f, 1.0f, 0.0f };
    camera.yaw = 90.0f;
    camera.pitch = 0.0f;

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
render(GLuint shader_program, glm::mat4x4 view, glm::mat4x4 projection)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUniformMatrix4fv(glGetUniformLocation(shader_program, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(shader_program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

    glDrawArrays(GL_TRIANGLES, 0, sizeof(vpoint) / sizeof(GLfloat));
}

static bool m_visible;

static void cursor_position_callback(GLFWwindow *window, const double xpos, const double ypos)
{
    static bool first_mouse = true;
    static double m_lastx = 0.0;
    static double m_lasty = 0.0;

    if (first_mouse) {
        m_lastx = xpos;
        m_lasty = ypos;
        first_mouse = false;
    }

    printf("Mouse at %lf %lf\n", xpos, ypos);

    float xoffset = static_cast<float>(xpos - m_lastx);
    float yoffset = static_cast<float>(m_lasty - ypos);

    m_lastx = xpos;
    m_lasty = ypos;

    const float sensitivity = 0.1f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    camera.yaw   += xoffset;
    camera.pitch += yoffset;

    if (camera.pitch > 89.0f)
        camera.pitch = 89.0f;
    if (camera.pitch < -89.0f)
        camera.pitch = -89.0f;

    printf("Pitch: %f Yaw: %f\n", camera.pitch, camera.yaw);
}

static glm::vec3
make_look_vec()
{
    glm::vec3 vec_look; 

    vec_look.x = cos(glm::radians(camera.yaw)) * cos(glm::radians(camera.pitch));
    vec_look.y = sin(glm::radians(camera.pitch));
    vec_look.z = sin(glm::radians(camera.yaw)) * cos(glm::radians(camera.pitch));

    return vec_look;
}

static glm::vec3
make_camera_center()
{
    return camera.eye + make_look_vec();
}

// Returns if should exit the render loop
static int
handle_key(GLFWwindow *window)
{
    constexpr GLfloat moving_speed = 1.0f / 32.0f;

    if (glfwWindowShouldClose(window) || (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS))
        return 1;

    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        return 1;

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        camera.eye += glm::normalize(make_look_vec()) * moving_speed;
    }

    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        camera.eye -= glm::normalize(make_look_vec()) * moving_speed;
    }

    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        const auto vec_look = make_look_vec();
        const auto vec_self_right = glm::normalize(glm::cross(camera.up, vec_look));
        /* We add the right vector, because OpenGL has X axis "flipped". */
        camera.eye += vec_self_right * moving_speed;
    }

    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        const auto vec_look = make_look_vec();
        const auto vec_self_right = glm::normalize(glm::cross(camera.up, vec_look));
        /* We subtract the right vector, because OpenGL has X axis "flipped". */
        camera.eye -= vec_self_right * moving_speed;
    }

    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
        camera.eye.y += moving_speed;
    }

    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) {
        camera.eye.y -= moving_speed;
    }

    if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS) {
        if (m_visible)
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        else
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

        m_visible = !m_visible;
    }

    if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS) {
        printf("Camera at: %f %f %f\n", camera.eye.x, camera.eye.y, camera.eye.z);
        const auto lookat = make_camera_center();
        printf("Look at: %f %f %f\n", lookat.x, lookat.y, lookat.z);
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

    GLFWwindow *window = glfwCreateWindow(win_width, win_height, "cube", NULL, NULL);

    if (!window) {
        fprintf(stderr, "window\n");
        return 1;
    }

    glfwSetCursorPosCallback(window, &cursor_position_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    glfwMakeContextCurrent(window);

#if defined(DRAW_OUTLINE)
    glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
#endif

    glewExperimental = GL_TRUE;
    if (const int glew_init_result = glewInit(); glew_init_result) {
        fprintf(stderr, "glew\n");
        return 1;
    }

    GLuint shader_program = init();

    while (1) {
        if (handle_key(window))
            break;

        render(shader_program, glm::lookAt(camera.eye, make_camera_center(), camera.up), projection_mat);
        glfwSwapBuffers(window);
        glfwSwapInterval(1);
        glfwPollEvents();
    }

    return 0;
}