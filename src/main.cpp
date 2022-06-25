/**
 * @file main.cpp
 * @author Timbre Freeman (tztz8)
 * @brief Start of the program
 * @version 0.1
 * @date 2022-06-20
 * 
 * @copyright Copyright (c) 2022
 * 
 */

//          --- Libraries ---

// Normal Lib
#include <map>
#include <string>
#include <filesystem>

// Include GLEW
#include <GL/glew.h>

// Include GLFW
#include <GLFW/glfw3.h>

// Image lib
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

// Math Lib
#define GLM_FORCE_RADIANS
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>

// Logger Lib
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#include <spdlog/spdlog.h>
#include <spdlog/cfg/argv.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "spdlog/sinks/rotating_file_sink.h"

// OpenGL Hellper Methons
#include "OpenGLHelperMethods.h"
#include "main.h"

#include "Sphere.h"

//          --- Filled's ---
// glfw window id
GLFWwindow* window;

// Bool to know when to exit
bool exitWindowFlag = false;

/**
 * if the program is set to fullscreen
 */
bool isFullScreen = false; // do not change // look at main to make it full screen

// initial screen size
int screenWidth = 512, screenHeight = 512;

// Current screen size
GLint glScreenWidth, glScreenHeight;

// flag to know when screen size changes
bool freeGLUTSizeUpdate;

// title info
#define TITLE_LENGTH 100

/**
 * description on what the keyboard key used for <br>
 *  - Map Key is char for the keyboard key being used <br>
 *  - Map Value is std::string for the description
 * @note when key is uppercase it use for normally Special cases like using shift or up arrow
 */
std::map<char, std::string> keyDescription;

//      --- predef methons ---
void setupLogger(int argc, char** argv);
void windowSizeChangeCallback([[maybe_unused]] GLFWwindow* thisWindow, int width, int height);
void updateAngle(GLfloat deltaTime);
void keyboard(bool setDiscrption);
void Display();
void Initialize();

Sphere* sphere;


/**
 * Main - Start of the program
 * @brief Main
 * @param argc number of argumnets
 * @param argv pointer to the array of arguments
 * @note the first argument shude (TODO: fix spelling) alwasy be the program name
 * @return int the sussecuse (TODO: fix spelling) of the program
 */
int main(int argc, char* argv[]) {
    setupLogger(argc, argv);
    SPDLOG_INFO("#####################");
    SPDLOG_INFO("#   Start of main   #");
    SPDLOG_INFO("#####################");
    // Initialise GLFW
    SPDLOG_INFO("Initialise GLFW");
    if (!glfwInit()) {
        SPDLOG_ERROR("initializing GLFW failed");
        return EXIT_FAILURE;
    }

    SPDLOG_INFO("Setting window hint's");
    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_TRUE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make macOS happy; should not be needed
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
#if OPENGL_DEBUG_FOR_GLFW
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, true);
#endif

    // Open a window and create its OpenGL context
    SPDLOG_INFO("Open a window and create its OpenGL context");
    char orginal_title[TITLE_LENGTH];
    strcpy(orginal_title, "GLFW - OpengGL-All - Bonus");
    window = glfwCreateWindow(screenWidth, screenHeight, orginal_title, nullptr, nullptr);
    if (window == nullptr) {
        SPDLOG_ERROR("Failed to open GLFW window");
        glfwTerminate();
        return EXIT_FAILURE;
    }
    glfwMakeContextCurrent(window);

    // resize
    SPDLOG_INFO("Setup resize (size change callback)");
    glfwSetWindowSizeCallback(window, windowSizeChangeCallback);
    glfwGetWindowSize(window, &glScreenWidth, &glScreenHeight);
    // use so it knows the screen size the system wants
    // (for example when using high-res (4k) screen the system will likely want
    // double the size (Hi-DPI) to make it possible to see for the user)
    screenWidth = glScreenWidth;
    screenHeight = glScreenHeight;
    windowSizeChangeCallback(window, glScreenWidth, glScreenHeight);

    // fullscreen
    bool makeFullScreen = false;
    if (makeFullScreen) {
        SPDLOG_INFO("Setting FullScreen");
        setFullScreen(true);
    }

    // icon
    SPDLOG_INFO("Setup icon for the window");
    GLFWimage icons[1];
    icons[0].pixels = stbi_load(
            "res/icon/Timbre-Logo_O.png",
            &icons[0].width,
            &icons[0].height,
            nullptr, 4);
    if (icons[0].pixels == nullptr) {
        SPDLOG_ERROR("Unable to load icon");
    } else {
        glfwSetWindowIcon(window, 1, icons);
        stbi_image_free(icons[0].pixels);
    }

    // Initialize GLEW
    SPDLOG_INFO("Initialize GLEW");
    if (glewInit() != GLEW_OK) {
        SPDLOG_ERROR("Failed to initialize GLEW");
        glfwTerminate();
        return EXIT_FAILURE;
    }

#if OPENGL_DEBUG_FOR_GLFW
    SPDLOG_INFO("Initialize GL Debug Output");
    int flags;
    glGetIntegerv(GL_CONTEXT_FLAGS, &flags);
    if (flags & GL_CONTEXT_FLAG_DEBUG_BIT)
    {
        glEnable(GL_DEBUG_OUTPUT);
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
        glDebugMessageCallback(glDebugOutput, nullptr);
        glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);
    } else {
        SPDLOG_ERROR("OpenGL Debug Fall to initialize");
    }
#endif

    SPDLOG_INFO("setting up some variables for Initialize");
    Sphere mainSphere( 64);
    sphere = &mainSphere;

    SPDLOG_INFO("Running Initialize method");
    Initialize();

    // GL info
    SPDLOG_INFO(spdlog::fmt_lib::format("GL Vendor : {}", glGetString(GL_VENDOR)));
    SPDLOG_INFO(spdlog::fmt_lib::format("GL Renderer : {}", glGetString(GL_RENDERER)));
    SPDLOG_INFO(spdlog::fmt_lib::format("GL Version (shading language) : {}", glGetString(GL_SHADING_LANGUAGE_VERSION)));
    SPDLOG_INFO(spdlog::fmt_lib::format("GL Version : {}", glGetString(GL_VERSION)));

    // Ensure we can capture the escape key being pressed below and any other keys
    SPDLOG_INFO("Setup user input mode");
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

    // List Keys being used
    // called to set the keys in keyboard map
    keyboard(true);
    // Go throw the map and print each key being used
    for (std::pair<const char, std::string> node: keyDescription) {
        if (isupper(node.first)) { // Use uppercase for normally Special cases like using shift or up arrow
            SPDLOG_INFO(spdlog::fmt_lib::format("Current Set Special Key: {} : Description: {}", node.first, node.second));
        } else {
            SPDLOG_INFO(spdlog::fmt_lib::format("Current Set Normal Key: {} : Description: {}", node.first, node.second));
        }
    }

    SPDLOG_INFO("setting up variables for the loop");

    // DeltaTime variables
    GLfloat lastFrame = 0.0f;

    // FPS variables
    GLfloat lastTimeFPS = 0.0f;
    GLint numberOfFrames = 0;
    double fps;
    double avgFPS = 0.0;
    int qtyFPS = 0;

    SPDLOG_INFO(spdlog::fmt_lib::format("Start window loop with exit:{} and glfwWindowShouldClose(window):{}",
            exitWindowFlag ? "true" : "false",
            glfwWindowShouldClose(window) ? "true" : "false"));
    while (!exitWindowFlag && !glfwWindowShouldClose(window)) {

        // Calculate delta time
        GLfloat currentFrame;
        currentFrame = static_cast<GLfloat>(glfwGetTime());
        GLfloat deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // FPS
        {
            GLfloat deltaTimeFPS = currentFrame - lastTimeFPS;
            numberOfFrames++;
            if (deltaTimeFPS >= 1.0f) {
                fps = static_cast<double>(numberOfFrames) / deltaTimeFPS;
                qtyFPS++;
                avgFPS += (fps - avgFPS) / qtyFPS;

                std::string title(orginal_title);
                title.append(" - [FPS: ");
//                title.append(fmt::format("{:0f}, Avg:{:0f}", fps, avgFPS));
                title.append(fmt::format("{:0f}", fps));
                title.append("]");

//                snprintf(title, TITLE_LENGTH - 1,
//                         "%s - [FPS: %3.2f]", orginal_title,
//                         fps);
                glfwSetWindowTitle(window, title.c_str());
                //fprintf(stdout, "Info: FPS: %f\n", fps);

                numberOfFrames = 0;
                lastTimeFPS = currentFrame;
            }
        }


        // Render
        Display();

        // Swap buffers
        glfwSwapBuffers(window);

        // Get evens (ex user input)
        glfwPollEvents();

        // check for user input to exit
        exitWindowFlag = glfwGetKey(window, GLFW_KEY_ESCAPE ) == GLFW_PRESS || exitWindowFlag;

        // check for user input
        keyboard(false);

        // update data (often angles of things)
        updateAngle(deltaTime);

    }
    SPDLOG_INFO(spdlog::fmt_lib::format("Avg FPS: {:0f}", avgFPS));

    // Close OpenGL window and terminate GLFW
    SPDLOG_INFO("Close OpenGL window and terminate GLFW");
    glfwTerminate();

    SPDLOG_INFO("Shutdown spdlog");
    spdlog::shutdown();

    return EXIT_SUCCESS;
}

void setupLogger(int argc, char* argv[]) {
    spdlog::cfg::load_argv_levels(argc, argv);
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(spdlog::level::trace);
    console_sink->set_pattern("[%Y-%m-%d %r %z UTC](%F) [pid:%P, tid:%t] [%^%l%$] [%s:%#] [%!]  %v");

    auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>("logs/OpenGL-ALL.log", 1048576 * 5, 3, false);
    file_sink->set_level(spdlog::level::trace);
    file_sink->set_pattern(spdlog::fmt_lib::format("[%Y-%m-%d %r %z UTC](%F) [pid:%P, tid:%t] [{}] [%l] [%@] [%!]  %v", argv[0]));

    auto file_sink_simple = std::make_shared<spdlog::sinks::rotating_file_sink_mt>("logs/OpenGL-ALL_Simple.log", 1048576 * 5, 3, false);
    file_sink_simple->set_level(spdlog::level::trace);
//    file_sink_simple->set_pattern("[%Y-%m-%d %r] [%l] [%s:%#] [%!]  %v");
    file_sink_simple->set_pattern("%Y-%m-%d %H:%M:%S,%e %l [%s:%#] [%!]  %v");

    auto loggerPtr = std::make_shared<spdlog::logger>(spdlog::logger("multi_sink", {console_sink, file_sink, file_sink_simple}));
    spdlog::set_default_logger(loggerPtr);
//    spdlog::set_pattern("[%Y-%m-%d %r %z UTC](%F) [pid:%P, tid:%t] [%^%l%$] [%@] [%!]  %v");
    spdlog::set_level(spdlog::level::trace);
}


// Window GL variables
/**
 * Aspect ratio <br>
 * Proportion between the width and the height of the window
 */
GLfloat aspect = float(screenWidth) / float(screenHeight);

// Booleans for current state
/**
 * Flag if to stop the rotate of the camera around the object
 */
bool stop_rotate = false;
/**
 * Flag to show the lines (not fill the trinalges)
 */
bool show_line = false;
/**
 * Flag to show the lines with GL_CULL_FACE (not fill the trinalges)
 */
bool show_line_new = false;
/**
 * Move the camera to look from above and change rotate to rotate the up vector
 */
bool top_view_flag = false;

// GL loc
/**
 * The location of the model matrix in the shader
 */
GLint matrix_loc;
/**
 * The location of the projection matrix in the shader
 */
GLint projection_matrix_loc;
/**
 * The location of the view (Camera) matrix in the shader
 */
GLint view_matrix_loc;

// shader program
/**
 * The handle of the shader program object.
 */
GLuint program;
GLuint programOne;

// Matrix's
/**
 * Camera matrix <br>
 * Use glm::lookAt to make
 */
glm::mat4 view_matrix(1.0f);
/**
 * 3d to 2d Matrix <br>
 * Normally using glm::perspective to make
 */
glm::mat4 projection_matrix(1.0f);
/**
 * matrix to apply to things being dawn <br>
 * Often use at less one of these <br>
 *     - glm::scale <br>
 *     - glm::translate <br>
 *     - glm::rotate <br>
 */
glm::mat4 model_matrix(1.0f);

// Add light components
/**
 * Vector of where the light position in 3d world
 */
glm::vec4 light_position(10.0, 6.0, 8.0, 1.0);
glm::vec4 light_intensity(1.0f, 1.0f, 1.0f, 1.0f);
glm::vec4 material_ambient(0.9, 0.5, 0.3, 1.0f);
glm::vec4 material_diffuse(0.9, 0.5, 0.3, 1.0f);
glm::vec4 material_specular(0.8, 0.8, 0.8, 1.0f);

float material_shininess = 50.0;
glm::vec4 ambient_product = light_intensity * material_ambient;
glm::vec4 diffuse_product = light_intensity * material_diffuse;
glm::vec4 specular_product = light_intensity * material_specular;

/**
 * Vector of where the light position in 3d canvas from using view (camera) matrix and 3d world position
 */
glm::vec4 light_position_camera;

// uniform indices of light
/**
 * The location of the light position in the shader
 */
GLint light_position_loc;
GLint ambient_product_loc;
GLint diffuse_product_loc;
GLint specular_product_loc;
GLint material_shininess_loc;

// Angle
/**
 * Angle used for rotating the view (camera)
 */
GLfloat rotateAngle = 0.0f;

//          --- Methods ---

/**
 * Set all the gl uniform for currentProgram
 * @param shaderProgram to set as the current shader program being used
 */
void setUniformLocations(GLuint shaderProgram) {
    glUseProgram(shaderProgram);
    view_matrix_loc = glGetUniformLocation(shaderProgram, "view_matrix");
    glUniformMatrix4fv(view_matrix_loc, 1, GL_FALSE, (GLfloat*)&view_matrix[0]);

    matrix_loc = glGetUniformLocation(shaderProgram, "model_matrix");
    glUniformMatrix4fv(matrix_loc, 1, GL_FALSE, (GLfloat*)&model_matrix[0]);

    projection_matrix_loc = glGetUniformLocation(shaderProgram, "projection_matrix");
    glUniformMatrix4fv(projection_matrix_loc, 1, GL_FALSE, (GLfloat*)&projection_matrix[0]);

    light_position_loc = glGetUniformLocation(shaderProgram, "LightPosition");
    glUniform4fv(light_position_loc, 1, &light_position_camera[0]);

    ambient_product_loc = glGetUniformLocation(shaderProgram, "AmbientProduct");
    glUniform4fv(ambient_product_loc, 1, (GLfloat*)&ambient_product[0]);

    diffuse_product_loc = glGetUniformLocation(shaderProgram, "DiffuseProduct");
    glUniform4fv(diffuse_product_loc, 1, (GLfloat*)&diffuse_product[0]);

    specular_product_loc = glGetUniformLocation(shaderProgram, "SpecularProduct");
    glUniform4fv(specular_product_loc, 1, (GLfloat*)&specular_product[0]);

    material_shininess_loc = glGetUniformLocation(shaderProgram, "Shininess");
    glUniform1f(material_shininess_loc, material_shininess);
}

/**
 * Called set setup open gl things (for example making the models)
 */
void Initialize(){

    // Create the program for rendering the model
    program = initShaders({"res/shaders/shader.frag", "res/shaders/shader.vert"});

    // Check if making the shader work or not // This is not in FreeGLUT as does need an exit flag
    if (exitWindowFlag) {
        return;
    }

    // attribute indices
    model_matrix = glm::mat4(1.0f);

    // Use the shader program
    setUniformLocations(program);

    // Set Clear Color (background color)
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

    // TODO: see why this is here
    // glEnable(GL_PROGRAM_POINT_SIZE);
    sphere->create();
}

/**
 * Called for every frame to draw on the screen
 */
void Display() {
    // Clear
    if (freeGLUTSizeUpdate) {
        glViewport(0, 0, glScreenWidth, glScreenHeight);
        freeGLUTSizeUpdate = false;
    }
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    // Show Lines
    // Tell GL to use GL_CULL_FACE
    if (show_line_new) {
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
    } else {
        glDisable(GL_CULL_FACE);
    }
    // Tell to fill or use Lines (not to fill) for the triangles
    if (show_line || show_line_new) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    } else {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    // Set Point Size
   glPointSize(10);

    // Set view matrix
    float rotateAngleRadians = glm::radians(rotateAngle);
    if (top_view_flag) { // Top View
        view_matrix = glm::lookAt(
                glm::vec3(0.0, 10.0f, 0.0), // camera is at the top
                glm::vec3(0, 0, 0), // look at the center
                glm::vec3(
                        sinf(rotateAngleRadians),
                        0.0f,
                        cosf(rotateAngleRadians)
                ) // rotating the camera
        );
    } else { // Normal View
        view_matrix = glm::lookAt(
                glm::vec3(
                        10.0f * sinf(rotateAngleRadians),
                        0.0f,
                        10.0f * cosf(rotateAngleRadians)
                ), // Moving around the center in a Center
                glm::vec3(0, 0, 0), // look at the center
                glm::vec3(0, 1, 0) // keeping the camera up
        );
    }
    // Let opengl know about the change
   glUniformMatrix4fv(view_matrix_loc, 1, GL_FALSE, (GLfloat*)&view_matrix[0]);

    // update light_position_camera base off on both light position and view matrix
    light_position_camera = view_matrix * light_position;
//    light_position_camera = light_position;
   glUniform4fv(light_position_loc, 1, &light_position_camera[0]);

    // update projection matrix (useful when the window resize)
    projection_matrix = glm::perspective(glm::radians(45.0f), aspect, 0.3f, 100.0f);
    glUniformMatrix4fv(projection_matrix_loc, 1, GL_FALSE, (GLfloat*)&projection_matrix[0]);

    // ---- Draw things ----

    model_matrix = glm::scale(glm::mat4(1.0f), glm::vec3(2.0f, 2.0f, 2.0f));
    glUniformMatrix4fv(matrix_loc, 1, GL_FALSE, (GLfloat*)&model_matrix[0]);
    sphere->draw();

    // ---- End of Draw things ----
    glFlush();
}

// ------------------ This is where the code between GLFW and FreeGLUT are Different ---------------------------

/**
 * keyboard key was pressed on last frame <br>
 *  - Map Key is char for the keyboard key being used <br>
 *  - Value is bool was pressed on last frame
 * @note when key is uppercase it use for normally Special cases like using shift or up arrow
 */
std::map<char, bool> keyPressed;
/**
 * keyboard key is pressed this frame <br>
 *  - Map Key is char for the keyboard key being used <br>
 *  - Map Value is bool is pressed this frame
 * @note when key is uppercase it use for normally Special cases like using shift or up arrow
 */
std::map<char, bool> keyCurrentlyPressed;

/**
 * On each frame it check for user input to toggle a flag
 */
void keyboard(bool setDiscrption) {
    if (setDiscrption) keyDescription['q'] = "Quit program";
    if (checkKey('q', GLFW_KEY_Q)) {
        tellWindowToClose();
    }

    bool sKey = checkKey('s', GLFW_KEY_S);
    bool shiftKeys = checkKey('S', GLFW_KEY_LEFT_SHIFT) || checkKey('S', GLFW_KEY_RIGHT_SHIFT);
    if (setDiscrption) keyDescription['s'] = "Show line view";
    if (sKey && !shiftKeys) {
        show_line = !show_line;
    }
    if (setDiscrption) keyDescription['S'] = "(SHIFT S) Show Line view but let the gpu hide hidden lines";
    if (sKey && shiftKeys) {
        show_line_new = !show_line_new;
    }

    if (setDiscrption) keyDescription['u'] = "Top view";
    if (setDiscrption) keyDescription['t'] = "Top view";
    if (checkKey('t', GLFW_KEY_T) || checkKey('u', GLFW_KEY_U)) {
        top_view_flag = !top_view_flag;
    }

    if (setDiscrption) keyDescription['r'] = "Rotate of camera";
    if (checkKey('r', GLFW_KEY_R)) {
        stop_rotate = !stop_rotate;
    }

    if (setDiscrption) keyDescription['F'] = "(F11) Full Screen";
    if (checkKey('F', GLFW_KEY_F11)) {
        setFullScreen(!isFullScreen);
    }
}

/**
 * check if key was press <br>
 * will only return true ones intel key not press on next check <br>
 * usefully for toggle things like show lines flag
 * @param key char often the key to store the state
 * @param GLFW_key the key that glfw recognise
 * @return false if still being press or not being press
 */
bool checkKey(char key, int GLFW_key) {
    bool returnValue;
    keyCurrentlyPressed[key] = glfwGetKey(window, GLFW_key ) == GLFW_PRESS;
    returnValue = (!keyPressed[key] && keyCurrentlyPressed[key]);
    keyPressed[key] = keyCurrentlyPressed[key];
    return returnValue;
}

int windowPosAtFullScreenX = 0;
int windowPosAtFullScreenY = 0;

/**
 * check window to be fullscreen
 * @param isFullScreenIn the state want
 */
void setFullScreen(bool isFullScreenIn) {
    if (isFullScreen == isFullScreenIn) {
        return;
    }
    if (!isFullScreenIn) {
        // auto unhide cursor when leaving full screen
        //glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        glfwSetWindowMonitor( window, nullptr,  windowPosAtFullScreenX, windowPosAtFullScreenY, screenWidth, screenHeight, 0 );
        isFullScreen = false;
        SPDLOG_INFO("Done setting to window (turn fullscreen off)");
    } else {
        if (glfwGetWindowMonitor(window) == nullptr) {
            // auto hide cursor when full screen
            //glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
            GLFWmonitor* _monitor = glfwGetPrimaryMonitor();
            // get resolution of monitor
            const GLFWvidmode * mode = glfwGetVideoMode(_monitor);
            // get the x and y of the window
            glfwGetWindowPos(window, &windowPosAtFullScreenX, &windowPosAtFullScreenY);

            // switch to full screen
            glfwSetWindowMonitor( window, _monitor, 0, 0, mode->width, mode->height, 0 );
            isFullScreen = true;
            SPDLOG_INFO("Done setting to fullscreen (turn fullscreen on)");
        } else {
            SPDLOG_ERROR("All ready fullscreen");
        }
    }
}

/**
 * Auto update GL Viewport when window size changes <br>
 * This is a callback method for GLFW
 * @param thisWindow the window that updated
 * @param width the new width
 * @param height the new height
 */
void windowSizeChangeCallback([[maybe_unused]] GLFWwindow* thisWindow, int width, int height) {
    glScreenHeight = height;
    glScreenWidth = width;
    freeGLUTSizeUpdate = true;
    aspect = float(glScreenWidth) / float(glScreenHeight);
}

/**
 * Update Angle on each frame
 * @param deltaTime the time between frames
 */
void updateAngle(GLfloat deltaTime) {

    if (!stop_rotate) {
        rotateAngle -= 2.75f * 10 * deltaTime;
    }

}

/**
 * Set the window flag to exit window
 */
void tellWindowToClose() {
    exitWindowFlag = true;
}