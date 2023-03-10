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
#pragma comment(lib, "opengl32.lib")

// Include GLFW
#include <GLFW/glfw3.h>

// Image lib
// #define STB_IMAGE_IMPLEMENTATION
// #include <stb_image.h>

// ImGUI lib
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

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

// OpenGL Helper Methods
#include "OpenGLHelperMethods.h"
#include "ModelLoader.h"
#include "main.h"

// Cuda Methods
#include "cudaInfo.cuh"

#include "Sphere.h"
#include "Cube.h"

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
std::string original_title("GLFW - OpenGL-All");

/**
 * description on what the keyboard key used for <br>
 *  - Map Key is char for the keyboard key being used <br>
 *  - Map Value is std::string for the description
 * @note when key is uppercase it use for normally Special cases like using shift or up arrow
 */
std::map<char, std::string> keyDescription;

//      --- pre-def methods ---
void setupLogger(int argc, char** argv);
static void glfw_error_callback(int error, const char* description);
void setupImGUI();
void windowSizeChangeCallback([[maybe_unused]] GLFWwindow* thisWindow, int width, int height);
void updateAngle(GLfloat deltaTime);
void keyboard(bool setDescription, GLfloat deltaTime);
void Display();
void ImGUIDisplay();
void Initialize();

enum class Models {
    sphere,
    cube,
    wireframeOBJ
};
//const Models allModels[] = {Models::sphere, Models::cube};
std::array<Models, 3> allModels = {Models::sphere, Models::cube, Models::wireframeOBJ};
std::array<std::string, 3> allModelsNames = {"Sphere", "Cube", "Wireframe OBJ"};
Models select_model = Models::sphere;
Sphere* sphere;
Cube* cube;
SimpleModel* wireframeOBJModel;

/**
 * Main - Start of the program
 * @brief Main
 * @param argc number of arguments
 * @param argv pointer to the array of arguments
 * @note the first argument should always be the program name
 * @return int the success of the program
 */
int main(int argc, char* argv[]) {
    setupLogger(argc, argv);
    SPDLOG_INFO("#####################");
    SPDLOG_INFO("#   Start of main   #");
    SPDLOG_INFO("#####################");
    // Initialise GLFW
    SPDLOG_INFO("Initialise GLFW");
    glfwSetErrorCallback(glfw_error_callback);
    if (glfwInit() == GLFW_FALSE) {
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
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
#endif

    // Open a window and create its OpenGL context
    SPDLOG_INFO("Open a window and create its OpenGL context");
    window = glfwCreateWindow(screenWidth, screenHeight, original_title.c_str(), nullptr, nullptr);
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
    loadGLFWIcon(window, "res/icon/Timbre-Logo_O.png");

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

    SPDLOG_INFO("Setting up IMGUI");
    setupImGUI();

    SPDLOG_INFO("setting up some variables for Initialize");
    Sphere mainSphere(32);
#pragma clang diagnostic push
#pragma ide diagnostic ignored "LocalValueEscapesScope"
    sphere = &mainSphere;
    Cube mainCube{};
    cube = &mainCube;
#pragma clang diagnostic pop

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
    keyboard(true, 0.0F);
    // Go throw the map and print each key being used
    for (std::pair<const char, std::string>& node: keyDescription) {
        if (isupper(node.first)) { // Use uppercase for normally Special cases like using shift or up arrow
            SPDLOG_INFO(spdlog::fmt_lib::format("Current Set Special Key: {} : Description: {}", node.first, node.second));
        } else {
            SPDLOG_INFO(spdlog::fmt_lib::format("Current Set Normal Key: {} : Description: {}", node.first, node.second));
        }
    }

    // Check if we have cuda
    if (!checkCuda()) {
        SPDLOG_ERROR("Cuda Not Avable");
    }

    SPDLOG_INFO("setting up variables for the loop");

    // DeltaTime variables
    GLfloat lastFrame = 0.0F;

    // FPS variables
    GLfloat lastTimeFPS = 0.0F;
    GLint numberOfFrames = 0;
    double fps;
    double avgFPS = 0.0;
    int qtyFPS = 0;

    SPDLOG_INFO(spdlog::fmt_lib::format("Start window loop with exit:{} and glfwWindowShouldClose(window):{}",
            exitWindowFlag ? "true" : "false",
            glfwWindowShouldClose(window) ? "true" : "false"));
    while (!exitWindowFlag && glfwWindowShouldClose(window) == GLFW_FALSE) {

        // Calculate delta time
        GLfloat currentFrame;
        currentFrame = static_cast<GLfloat>(glfwGetTime());
        GLfloat deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // FPS
        {
            GLfloat deltaTimeFPS = currentFrame - lastTimeFPS;
            numberOfFrames++;
            if (deltaTimeFPS >= 1.0F) {
                fps = static_cast<double>(numberOfFrames) / deltaTimeFPS;
                qtyFPS++;
                avgFPS += (fps - avgFPS) / static_cast<double>(qtyFPS);

                std::string title(original_title);
                title.append(" - [FPS: ");
//                title.append(fmt::format("{:0f}, Avg:{:0f}", fps, avgFPS));
                title.append(fmt::format("{:0f}", fps));
                title.append("]");

//                snprintf(title, TITLE_LENGTH - 1,
//                         "%s - [FPS: %3.2f]", original_title,
//                         fps);
                glfwSetWindowTitle(window, title.c_str());
                //fprintf(stdout, "Info: FPS: %f\n", fps);

                numberOfFrames = 0;
                lastTimeFPS = currentFrame;
            }
        }

        // Get evens (ex user input)
        glfwPollEvents();

        // check for user input to exit
        exitWindowFlag = glfwGetKey(window, GLFW_KEY_ESCAPE ) == GLFW_PRESS || exitWindowFlag;

        // check for user input
        keyboard(false, deltaTime);

        // ImGui
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGUIDisplay();

        // Render
        ImGui::Render();
        Display();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // Swap buffers
        glfwSwapBuffers(window);

        // update data (often angles of things)
        updateAngle(deltaTime);

    }
    SPDLOG_INFO(spdlog::fmt_lib::format("Exit Window Loop, Avg FPS: {:0f}", avgFPS));

    // ImGUI Cleanup
    SPDLOG_INFO("Cleanup ImGUI");
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    // Close OpenGL window and terminate GLFW
    SPDLOG_INFO("Close GLFW window and terminate GLFW");
    glfwDestroyWindow(window);
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

void setupImGUI() {
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui styleis:issue is:open
//    ImGui::StyleColorsDark();
    ImGui::StyleColorsClassic();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    // Set High DPI scale factor;
    GLFWmonitor *monitor = glfwGetPrimaryMonitor();
    float xScale, yScale;
    glfwGetMonitorContentScale(monitor, &xScale, &yScale);
    io.DisplayFramebufferScale = ImVec2(xScale, yScale);
    ImGui::GetStyle().ScaleAllSizes(xScale);
    ImFontConfig cfg;
    cfg.SizePixels = 13.0F * xScale;

    // Load Fonts
    // - If no fonts are loaded, dear imgui will use the default font. You can also load multiple fonts and use ImGui::PushFont()/PopFont() to select them.
    // - AddFontFromFileTTF() will return the ImFont* so you can store it if you need to select the font among multiple.
    // - If the file cannot be loaded, the function will return NULL. Please handle those errors in your application (e.g. use an assertion, or display an error and quit).
    // - The fonts will be rasterized at a given size (w/ oversampling) and stored into a texture when calling ImFontAtlas::Build()/GetTexDataAsXXXX(), which ImGui_ImplXXXX_NewFrame below will call.
    // - Read 'docs/FONTS.md' for more instructions and details.
    // - Remember that in C/C++ if you want to include a backslash \ in a string literal you need to write a double backslash \\ !
    io.Fonts->AddFontDefault(&cfg);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/Roboto-Medium.ttf", 16.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/Cousine-Regular.ttf", 15.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/DroidSans.ttf", 16.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/ProggyTiny.ttf", 10.0f);
    //ImFont* font = io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\ArialUni.ttf", 18.0f, NULL, io.Fonts->GetGlyphRangesJapanese());
    //IM_ASSERT(font != NULL);
}

static void glfw_error_callback(int error, const char* description) {
    SPDLOG_ERROR(spdlog::fmt_lib::format("Glfw Error {}: {}", error, description));
}


std::vector<std::filesystem::path> shaderPaths = {"res/shaders/shader.frag", "res/shaders/shader.vert"};

// Window GL variables
/**
 * Aspect ratio <br>
 * Proportion between the width and the height of the window
 */
GLfloat aspect = static_cast<float>(screenWidth) / static_cast<float>(screenHeight);

// Booleans for current state
/**
 * Flag if to stop the rotate of the camera around the object
 */
bool stop_rotate = true;
/**
 * Flag to show the lines (not fill the triangles)
 */
bool show_line = false;
/**
 * Flag to show the lines with GL_CULL_FACE (not fill the triangles)
 */
bool cull_face = false;
bool cull_face_back = false;
/**
 * Move the camera to look from above and change rotate to rotate the up vector
 */
bool top_view_flag = false;

ImVec4 clear_color = ImVec4(0.0F, (170.0F/255.0F), 1.0F, 1.0F);

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
glm::mat4 view_matrix(1.0F);
float view_eye_radias = 10.0F;
float view_eye_y = 3.5F;
glm::vec3 view_center(0.0F, 0.0F, 0.0F);
/**
 * 3d to 2d Matrix <br>
 * Normally using glm::perspective to make
 */
glm::mat4 projection_matrix(1.0F);
float fov = 45.0F;
/**
 * matrix to apply to things being dawn <br>
 * Often use at less one of these <br>
 *     - glm::scale <br>
 *     - glm::translate <br>
 *     - glm::rotate <br>
 */
glm::mat4 model_matrix(1.0F);

float movingScale = 1.5F;

// Add light components
/**
 * Vector of where the light position in 3d world
 */
glm::vec4 light_position(10.0F, 6.0F, 8.0F, 1.0F);
glm::vec4 light_intensity(1.0F, 1.0F, 1.0F, 1.0F);
glm::vec4 material_ambient(0.9F, 0.9F, 0.9F, 1.0F);
glm::vec4 material_diffuse(0.9F, 0.9F, 0.9F, 1.0F);
glm::vec4 material_specular(0.9F, 0.9F, 0.9F, 1.0F);

float material_shininess = 50.0F;
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
GLfloat rotateAngle = 180.0F;

glm::vec3 model_Scale(2.0F, 2.0F, 2.0F);
GLfloat model_rotate_angle = 180.0F;
glm::vec3 model_rotate_vector(1.0F, 0.0F, 0.0F);
//std::vector

// Texture ID's
GLuint earthTexID;
GLuint randomMadeTexID;
GLuint cubeTexID;
GLuint modelTexID;

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

    glUniform1i(glGetUniformLocation(program, "Tex1"), 0);
}

/**
 * Called set setup open gl things (for example making the models)
 */
void Initialize(){

    // Create the program for rendering the model
    program = initShaders(shaderPaths);

    // Check if making the shader work or not // This is not in FreeGLUT as does need an exit flag
    if (exitWindowFlag) {
        return;
    }

    // attribute indices
    model_matrix = glm::mat4(1.0F);

    // Use the shader program
    setUniformLocations(program);

    // Set Clear Color (background color)
    glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);

    earthTexID = loadTexture("res/textures/Earth.jpg");
    randomMadeTexID = loadTexture("res/textures/randomMade.png");
    cubeTexID = loadTexture("res/textures/wests_textures/stone wall 9.png");
    modelTexID = loadTexture("res/textures/failsafe.png");

    // load model
    wireframeOBJModel = readOBJ("res/models/box.obj");
    if (wireframeOBJModel == nullptr) {
        // TODO: add
    } else {
        // TODO: add
    }

    // TODO: see why this is here
    // glEnable(GL_PROGRAM_POINT_SIZE);
    sphere->create();
    cube->create();
}

void ImGUIDisplay() {
    static bool show_demo_window = false;
    static bool p_open = true;
    static int counter = 0;
    if (show_demo_window) {
        ImGui::ShowDemoWindow(&show_demo_window);
        if (!show_demo_window) {
            counter = 0;
        }
    }

    if(p_open) {
        static bool no_titlebar = false;
        static bool no_scrollbar = false;
        static bool no_menu = false;
        static bool no_move = false;
        static bool no_resize = false;
        static bool no_collapse = false;
        static bool no_nav = false;
        static bool no_background = false;
        static bool no_bring_to_front = false;
        static bool unsaved_document = false;

        ImGuiWindowFlags window_flags = 0;
        if (no_titlebar)        window_flags |= ImGuiWindowFlags_NoTitleBar;
        if (no_scrollbar)       window_flags |= ImGuiWindowFlags_NoScrollbar;
        if (!no_menu)           window_flags |= ImGuiWindowFlags_MenuBar;
        if (no_move)            window_flags |= ImGuiWindowFlags_NoMove;
        if (no_resize)          window_flags |= ImGuiWindowFlags_NoResize;
        if (no_collapse)        window_flags |= ImGuiWindowFlags_NoCollapse;
        if (no_nav)             window_flags |= ImGuiWindowFlags_NoNav;
        if (no_background)      window_flags |= ImGuiWindowFlags_NoBackground;
        if (no_bring_to_front)  window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus;
        if (unsaved_document)   window_flags |= ImGuiWindowFlags_UnsavedDocument;

        if(!ImGui::Begin(original_title.c_str(), &p_open, window_flags)) {
            // Early out if the window is collapsed, as an optimization.
            ImGui::End();
            return;
        }

        // Menu Bar
        if (ImGui::BeginMenuBar())
        {
            if (ImGui::BeginMenu("Menu"))
            {
                ImGui::MenuItem("(demo menu)", nullptr, false, false);
                if (ImGui::MenuItem("Quit", "Alt+F4")) {
                    SPDLOG_INFO("ImGui, user tell window to close");
                    tellWindowToClose();
                }
                ImGui::EndMenu();
            }
            ImGui::EndMenuBar();
        }

        if (ImGui::CollapsingHeader("Camera")) {
            ImGui::Checkbox("Stop Rotate camera", &stop_rotate);      // Edit bools storing our window open/close state
            ImGui::Checkbox("Top view camera", &top_view_flag);
            ImGui::SliderFloat("camera rotate angle", &rotateAngle, 0.0F, 360.0F);
            ImGui::SliderFloat("Field of view", &fov, 0.0F, 180.0F);
            ImGui::SliderFloat("Eye Hight", &view_eye_y, -75.0F, 75.0F);
            ImGui::SliderFloat("Eye Radias", &view_eye_radias, 0.0F, 75.0F);
            ImGui::DragFloat3("Eye Center", (float*)&view_center, 0.1F);
        }

        if (ImGui::CollapsingHeader("Light Settings")) {
            bool updateShader = false;
            ImGui::DragFloat3("position", (float*)&light_position, 0.1F);
            updateShader = ImGui::ColorEdit3("intensity", (float*)&light_intensity) || updateShader;
            updateShader = ImGui::ColorEdit3("ambient", (float*)&material_ambient) || updateShader;
            updateShader = ImGui::ColorEdit3("diffuse", (float*)&material_diffuse) || updateShader;
            updateShader = ImGui::ColorEdit3("specular", (float*)&material_specular) || updateShader;
            updateShader = ImGui::SliderFloat("shininess", &material_shininess, 0.0F, 70.0F) || updateShader;
            if (updateShader) {
                ambient_product = light_intensity * material_ambient;
                diffuse_product = light_intensity * material_diffuse;
                specular_product = light_intensity * material_specular;
                setUniformLocations(program);
            }
        }

        if (ImGui::CollapsingHeader("Window Settings")) {

            int windowPosX, windowPosY;
            glfwGetWindowPos(window, &windowPosX, &windowPosY);
            ImGui::Value("Window x pos", windowPosX);
            ImGui::Value("Window y pos", windowPosY);
            ImGui::Value("Window width", glScreenWidth);
            ImGui::Value("Window height", glScreenHeight);
            int numOfMonitors;
            GLFWmonitor** monitors = glfwGetMonitors(&numOfMonitors);
            for (int i = 0; i < numOfMonitors; ++i) {
                if (ImGui::CollapsingHeader(glfwGetMonitorName(monitors[i]))) {
                    ImGui::Value("Monitor i", i);
                    const GLFWvidmode* mode = glfwGetVideoMode(monitors[i]);
                    ImGui::Value("Monitor refreshRate", mode->refreshRate);
                    ImGui::Value("Monitor width", mode->width);
                    ImGui::Value("Monitor height", mode->height);
                    int monitorPosX, monitorPosY;
                    glfwGetMonitorPos(monitors[i], &monitorPosX, &monitorPosY);
                    ImGui::Value("Monitor x pos", monitorPosX);
                    ImGui::Value("Monitor y pos", monitorPosY);
                    int monitorWidth, monitorHeight;
                    glfwGetMonitorWorkarea(monitors[i], &monitorPosX, &monitorPosY, &monitorWidth, &monitorHeight);
                    ImGui::Value("Monitor Work-area width", monitorWidth);
                    ImGui::Value("Monitor Work-area height", monitorHeight);
                    ImGui::Value("Monitor Work-area x pos", monitorPosX);
                    ImGui::Value("Monitor Work-area y pos", monitorPosY);
                    float monitorScaleX, monitorScaleY;
                    glfwGetMonitorContentScale(monitors[i], &monitorScaleX, &monitorScaleY);
                    ImGui::Value("Monitor x scale", monitorScaleX);
                    ImGui::Value("Monitor y scale", monitorScaleY);
                    int monitorPhysicalSizeWidth, monitorPhysicalSizeHeight;
                    glfwGetMonitorPhysicalSize(monitors[i], &monitorPhysicalSizeWidth, &monitorPhysicalSizeHeight);
                    ImGui::Value("Monitor physical size width", monitorPhysicalSizeWidth);
                    ImGui::Value("Monitor physical size height", monitorPhysicalSizeHeight);
                }
            }

            static bool fameLimit = true;
            if (ImGui::Checkbox("vsync (frame limit)", &fameLimit)) {
                glfwSwapInterval(fameLimit);
            }

            bool fullScreenImGui = isFullScreen;
            if (ImGui::Checkbox("Full Screen", &fullScreenImGui)) {
                setFullScreen(fullScreenImGui);
            }

            if (ImGui::ColorEdit4("clear color", (float*)&clear_color)) {
                glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
            }// Edit 3 floats representing a color
        }

        if (ImGui::CollapsingHeader("ImGui Settings")) {
            if (ImGui::TreeNode("Configuration"))
            {
                ImGuiIO& io = ImGui::GetIO();

                if (ImGui::TreeNode("Configuration##2"))
                {
                    ImGui::CheckboxFlags("io.ConfigFlags: NavEnableKeyboard",    &io.ConfigFlags, ImGuiConfigFlags_NavEnableKeyboard);
                    ImGui::SameLine(); HelpMarker("Enable keyboard controls.");
                    ImGui::CheckboxFlags("io.ConfigFlags: NavEnableGamepad",     &io.ConfigFlags, ImGuiConfigFlags_NavEnableGamepad);
                    ImGui::SameLine(); HelpMarker("Enable gamepad controls. Require backend to set io.BackendFlags |= ImGuiBackendFlags_HasGamepad.\n\nRead instructions in imgui.cpp for details.");
                    ImGui::CheckboxFlags("io.ConfigFlags: NavEnableSetMousePos", &io.ConfigFlags, ImGuiConfigFlags_NavEnableSetMousePos);
                    ImGui::SameLine(); HelpMarker("Instruct navigation to move the mouse cursor. See comment for ImGuiConfigFlags_NavEnableSetMousePos.");
                    ImGui::CheckboxFlags("io.ConfigFlags: NoMouse",              &io.ConfigFlags, ImGuiConfigFlags_NoMouse);
                    if (io.ConfigFlags & ImGuiConfigFlags_NoMouse)
                    {
                        // The "NoMouse" option can get us stuck with a disabled mouse! Let's provide an alternative way to fix it:
                        if (fmodf((float)ImGui::GetTime(), 0.40f) < 0.20f)
                        {
                            ImGui::SameLine();
                            ImGui::Text("<<PRESS SPACE TO DISABLE>>");
                        }
                        if (ImGui::IsKeyPressed(ImGuiKey_Space))
                            io.ConfigFlags &= ~ImGuiConfigFlags_NoMouse;
                    }
                    ImGui::CheckboxFlags("io.ConfigFlags: NoMouseCursorChange", &io.ConfigFlags, ImGuiConfigFlags_NoMouseCursorChange);
                    ImGui::SameLine(); HelpMarker("Instruct backend to not alter mouse cursor shape and visibility.");
                    ImGui::Checkbox("io.ConfigInputTrickleEventQueue", &io.ConfigInputTrickleEventQueue);
                    ImGui::SameLine(); HelpMarker("Enable input queue trickling: some types of events submitted during the same frame (e.g. button down + up) will be spread over multiple frames, improving interactions with low framerates.");
                    ImGui::Checkbox("io.ConfigInputTextCursorBlink", &io.ConfigInputTextCursorBlink);
                    ImGui::SameLine(); HelpMarker("Enable blinking cursor (optional as some users consider it to be distracting).");
                    ImGui::Checkbox("io.ConfigDragClickToInputText", &io.ConfigDragClickToInputText);
                    ImGui::SameLine(); HelpMarker("Enable turning DragXXX widgets into text input with a simple mouse click-release (without moving).");
                    ImGui::Checkbox("io.ConfigWindowsResizeFromEdges", &io.ConfigWindowsResizeFromEdges);
                    ImGui::SameLine(); HelpMarker("Enable resizing of windows from their edges and from the lower-left corner.\nThis requires (io.BackendFlags & ImGuiBackendFlags_HasMouseCursors) because it needs mouse cursor feedback.");
                    ImGui::Checkbox("io.ConfigWindowsMoveFromTitleBarOnly", &io.ConfigWindowsMoveFromTitleBarOnly);
                    ImGui::Checkbox("io.MouseDrawCursor", &io.MouseDrawCursor);
                    ImGui::SameLine(); HelpMarker("Instruct Dear ImGui to render a mouse cursor itself. Note that a mouse cursor rendered via your application GPU rendering path will feel more laggy than hardware cursor, but will be more in sync with your other visuals.\n\nSome desktop applications may use both kinds of cursors (e.g. enable software cursor only when resizing/dragging something).");
                    ImGui::Text("Also see Style->Rendering for rendering options.");
                    ImGui::TreePop();
                    ImGui::Separator();
                }

                if (ImGui::TreeNode("Backend Flags"))
                {
                    HelpMarker(
                            "Those flags are set by the backends (imgui_impl_xxx files) to specify their capabilities.\n"
                            "Here we expose them as read-only fields to avoid breaking interactions with your backend.");

                    // Make a local copy to avoid modifying actual backend flags.
                    // FIXME: We don't use BeginDisabled() to keep label bright, maybe we need a BeginReadonly() equivalent..
                    ImGuiBackendFlags backend_flags = io.BackendFlags;
                    ImGui::CheckboxFlags("io.BackendFlags: HasGamepad",           &backend_flags, ImGuiBackendFlags_HasGamepad);
                    ImGui::CheckboxFlags("io.BackendFlags: HasMouseCursors",      &backend_flags, ImGuiBackendFlags_HasMouseCursors);
                    ImGui::CheckboxFlags("io.BackendFlags: HasSetMousePos",       &backend_flags, ImGuiBackendFlags_HasSetMousePos);
                    ImGui::CheckboxFlags("io.BackendFlags: RendererHasVtxOffset", &backend_flags, ImGuiBackendFlags_RendererHasVtxOffset);
                    ImGui::TreePop();
                    ImGui::Separator();
                }

                if (ImGui::TreeNode("Style"))
                {
                    HelpMarker("The same contents can be accessed in 'Tools->Style Editor' or by calling the ShowStyleEditor() function.");
                    ImGui::ShowStyleEditor();
                    ImGui::TreePop();
                    ImGui::Separator();
                }

                if (ImGui::TreeNode("Capture/Logging"))
                {
                    HelpMarker(
                            "The logging API redirects all text output so you can easily capture the content of "
                            "a window or a block. Tree nodes can be automatically expanded.\n"
                            "Try opening any of the contents below in this window and then click one of the \"Log To\" button.");
                    ImGui::LogButtons();

                    HelpMarker("You can also call ImGui::LogText() to output directly to the log without a visual output.");
                    if (ImGui::Button("Copy \"Hello, world!\" to clipboard"))
                    {
                        ImGui::LogToClipboard();
                        ImGui::LogText("Hello, world!");
                        ImGui::LogFinish();
                    }
                    ImGui::TreePop();
                }

                ImGui::TreePop();
                ImGui::Separator();
            }
            if (ImGui::TreeNode("Window options"))
            {
                if (ImGui::BeginTable("split", 3))
                {
                    ImGui::TableNextColumn(); ImGui::Checkbox("No titlebar", &no_titlebar);
                    ImGui::TableNextColumn(); ImGui::Checkbox("No scrollbar", &no_scrollbar);
                    ImGui::TableNextColumn(); ImGui::Checkbox("No menu", &no_menu);
                    ImGui::TableNextColumn(); ImGui::Checkbox("No move", &no_move);
                    ImGui::TableNextColumn(); ImGui::Checkbox("No resize", &no_resize);
                    ImGui::TableNextColumn(); ImGui::Checkbox("No collapse", &no_collapse);
                    ImGui::TableNextColumn(); ImGui::Checkbox("No nav", &no_nav);
                    ImGui::TableNextColumn(); ImGui::Checkbox("No background", &no_background);
                    ImGui::TableNextColumn(); ImGui::Checkbox("No bring to front", &no_bring_to_front);
                    ImGui::TableNextColumn(); ImGui::Checkbox("Unsaved document", &unsaved_document);
                    ImGui::EndTable();
                }
                ImGui::TreePop();
                ImGui::Separator();
            }
        }

        if (ImGui::CollapsingHeader("Graphics")) {
            ImGui::Checkbox("Show lines", &show_line);
            ImGui::Checkbox("GL_CULL_FACE", &cull_face);
            if (cull_face) {
                ImGui::Checkbox("GL_CULL_FACE back", &cull_face_back);
            }

            static bool allSameScaleValue = true;
            if (ImGui::Checkbox("Scale is all the same value", &allSameScaleValue)) {
                if (allSameScaleValue) {
                    float scale = model_Scale.x;
                    model_Scale.x = scale;
                    model_Scale.y = scale;
                    model_Scale.z = scale;
                }
            }
            if (allSameScaleValue) {
                float scale = model_Scale.x;
                if (ImGui::DragFloat("Model Scale", &scale, 0.001F)) {
                    model_Scale.x = scale;
                    model_Scale.y = scale;
                    model_Scale.z = scale;
                }
            } else {
                ImGui::DragFloat3("Model Scale", (float*)&model_Scale, 0.001F);
            }
            ImGui::SliderFloat("Model rotate angle", &model_rotate_angle, 0.0F, 360.0F);
            ImGui::SliderFloat3("Model rotate vector", (float*)&model_rotate_vector, -1.0F, 1.0F);

            const char* models[allModels.size()];
            int item = 0;
            for (int i = 0; i < allModels.size(); ++i) {
                models[i] = allModelsNames[i].c_str();
                if (select_model == allModels[i]) {
                    item = i;
                }
            }
            ImGui::Combo("Model", &item, models, allModels.size());
            select_model = allModels[item];

            int steps = sphere->getStep();
            if (ImGui::SliderInt("Sphere Steps", &steps, 3, 128)) {
                sphere->updateStep(steps);
            }

            if (ImGui::Button("Select new earth texture image file")) {
                glDeleteTextures(1,&earthTexID);
                earthTexID = loadTexture(UserSelectImageFile().string().c_str());
            }

            if (ImGui::Button("Select new random texture image file")) {
                glDeleteTextures(1,&randomMadeTexID);
                randomMadeTexID = loadTexture(UserSelectImageFile().string().c_str());
            }

            if (ImGui::Button("Reload Shaders")) {
                SPDLOG_INFO("Reload Shaders");
                glDeleteProgram(program);
                program = initShaders(shaderPaths);
                setUniformLocations(program);
            }
        }

        ImGui::Spacing();

        if (ImGui::Button("Button")){
            counter++;
        } // Buttons return true when clicked (most widgets return true when edited/activated)
        ImGui::SameLine();
        ImGui::Text("counter = %d", counter);
        if (counter > 0) {
            show_demo_window = true;
        }

        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0F / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

        ImGui::End();
    } else {
        ImGui::Begin("Quiting");
        ImGui::Text("Are you sure?");
        if (ImGui::Button("Quit")) {
            tellWindowToClose();
            SPDLOG_INFO("User used ImGui to quit program");
        }
        ImGui::SameLine();
        if (ImGui::Button("Reopen GUI")) {
            p_open = true;
            SPDLOG_INFO("User reopen ImGui");
        }
        ImGui::End();
    }


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
    if (cull_face) {
        glEnable(GL_CULL_FACE);
        if (cull_face_back) {
            glCullFace(GL_BACK);
        } else {
            glCullFace( GL_FRONT);
        }
    } else {
        glDisable(GL_CULL_FACE);
    }
    // Tell to fill or use Lines (not to fill) for the triangles
    if (show_line) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    } else {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    // Set Point Size
    glPointSize(10.0F);

    // Set view matrix
    float rotateAngleRadians = glm::radians(rotateAngle);
    if (top_view_flag) { // Top View
        view_matrix = glm::lookAt(
                (glm::vec3(0.0F, view_eye_radias, 0.0F) + view_center), // camera is at the top
                view_center, // look at the center
                glm::vec3(
                        sinf(rotateAngleRadians),
                        0.0F,
                        cosf(rotateAngleRadians)
                ) // rotating the camera
        );
    } else { // Normal View
        view_matrix = glm::lookAt(
                (glm::vec3(
                        view_eye_radias * sinf(rotateAngleRadians),
                        view_eye_y,
                        view_eye_radias * cosf(rotateAngleRadians)
                ) + view_center), // Moving around the center in a view Center
                view_center, // look at the center
                glm::vec3(0.0F, 1.0F, 0.0F) // keeping the camera up
        );
    }
    // Let opengl know about the change
   glUniformMatrix4fv(view_matrix_loc, 1, GL_FALSE, (GLfloat*)&view_matrix[0]);

    // update light_position_camera base off on both light position and view matrix
    light_position_camera = view_matrix * light_position;
//    light_position_camera = light_position;
   glUniform4fv(light_position_loc, 1, &light_position_camera[0]);

    // update projection matrix (useful when the window resize)
    projection_matrix = glm::perspective(glm::radians(fov), aspect, 0.3F, 100.0F);
    glUniformMatrix4fv(projection_matrix_loc, 1, GL_FALSE, (GLfloat*)&projection_matrix[0]);

    model_matrix = glm::mat4(1.0F);

    // ---- Draw things ----

    switch (select_model) {
        case Models::sphere:
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, earthTexID);
            break;
        case Models::cube:
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, cubeTexID);
            break;
        case Models::wireframeOBJ:
            // TODO: add 
            break;
    }
    model_matrix = glm::scale(model_matrix, model_Scale);
    model_matrix = glm::rotate(model_matrix, glm::radians(model_rotate_angle), model_rotate_vector);
    glUniformMatrix4fv(matrix_loc, 1, GL_FALSE, (GLfloat*)&model_matrix[0]);
    switch (select_model) {
        case Models::sphere:
            sphere->draw();
            break;
        case Models::cube:
            cube->draw();
            break;
    }

    model_matrix = glm::mat4(1.0F);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, randomMadeTexID);
    model_matrix = glm::translate(model_matrix, glm::vec3(light_position.x, light_position.y, light_position.z));
    if (light_position.x == 0.0F && light_position.y == 0.0F && light_position.z == 0.0F) {
        model_matrix = glm::mat4(1.0F);
    }
    glUniformMatrix4fv(matrix_loc, 1, GL_FALSE, (GLfloat*)&model_matrix[0]);
    cube->draw();

    model_matrix = glm::mat4(1.0F);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, randomMadeTexID);
    model_matrix = glm::translate(model_matrix, view_center);
    glUniformMatrix4fv(matrix_loc, 1, GL_FALSE, (GLfloat*)&model_matrix[0]);
    cube->draw();

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
void keyboard(bool setDescription, GLfloat deltaTime) {
    if (setDescription) keyDescription['q'] = "Quit program";
    if (checkKey('q', GLFW_KEY_Q)) {
        tellWindowToClose();
    }

    if (setDescription) keyDescription['x'] = "Show line view";
    if (checkKey('x', GLFW_KEY_X)) {
        show_line = !show_line;
    }

    if (setDescription) keyDescription['z'] = "GL Cull Face back";
    if (checkKey('z', GLFW_KEY_Z)) {
        cull_face_back = !cull_face_back;
    }

    if (setDescription) keyDescription['c'] = "GL Cull Face";
    if (checkKey('c', GLFW_KEY_C)) {
        cull_face = !cull_face;
    }

    if (setDescription) keyDescription['u'] = "Top view";
    if (setDescription) keyDescription['t'] = "Top view";
    if (checkKey('t', GLFW_KEY_T) || checkKey('u', GLFW_KEY_U)) {
        top_view_flag = !top_view_flag;
    }

    if (setDescription) keyDescription['r'] = "Rotate of camera";
    if (checkKey('r', GLFW_KEY_R)) {
        stop_rotate = !stop_rotate;
    }

    if (setDescription) keyDescription['F'] = "(F11) Full Screen";
    if (checkKey('F', GLFW_KEY_F11)) {
        setFullScreen(!isFullScreen);
    }

    if (glfwGetKey(window, GLFW_KEY_W)) {
        view_center.z += movingScale * deltaTime;
    }
    if (glfwGetKey(window, GLFW_KEY_S)) {
        view_center.z -= movingScale * deltaTime;
    }
    if (glfwGetKey(window, GLFW_KEY_D)) {
        view_center.x -= movingScale * deltaTime;
    }
    if (glfwGetKey(window, GLFW_KEY_A)) {
        view_center.x += movingScale * deltaTime;
    }
    if (glfwGetKey(window, GLFW_KEY_UP)) {
        view_center.y += movingScale * deltaTime;
    }
    if (glfwGetKey(window, GLFW_KEY_DOWN)) {
        view_center.y -= movingScale * deltaTime;
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
int windowWidthAtFullScreen = 0;
int windowHeightAtFullScreen = 0;

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
        glfwSetWindowMonitor( window, nullptr,  windowPosAtFullScreenX, windowPosAtFullScreenY, windowWidthAtFullScreen, windowHeightAtFullScreen, 0 );
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
            // set the screen size
            windowHeightAtFullScreen = glScreenHeight;
            windowWidthAtFullScreen = glScreenWidth;

            // switch to full screen
            glfwSetWindowMonitor( window, _monitor, 0, 0, mode->width, mode->height, mode->refreshRate );
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
    aspect = static_cast<GLfloat>(glScreenWidth) / static_cast<GLfloat>(glScreenHeight);
}

/**
 * Update Angle on each frame
 * @param deltaTime the time between frames
 */
void updateAngle(GLfloat deltaTime) {

    if (!stop_rotate) {
        rotateAngle -= 2.75F * 10.0F * deltaTime;
        if (rotateAngle < 0.0F) {
            rotateAngle = 360.0F;
        }
        if (rotateAngle > 360.0F) {
            rotateAngle = 0.0F;
        }
    }

}

/**
 * Set the window flag to exit window
 */
void tellWindowToClose() {
    exitWindowFlag = true;
}