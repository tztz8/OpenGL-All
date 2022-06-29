cmake_minimum_required(VERSION 3.0.0)
project(OpenGL-ALL VERSION 0.1.0)

# Set the version of CPP we are using
set(CMAKE_CXX_STANDARD 17)

if (MSVC) # microsoft visual code requires this package when working with opengl
    find_package(OpenGL REQUIRED)
endif()

find_package(GLM QUIET)
if (NOT GLM_FOUND)
    add_subdirectory(exernalLibraries/glm)
endif()
message(STATUS "GLM package avable")

find_package(glfw3 QUIET)
if (NOT glfw3_FOUND)
    add_subdirectory(exernalLibraries/glfw)
    link_libraries(glfw ${GLFW_LIBRARIES})
endif()
message(STATUS "glfw3 package avable")

find_package(spdlog QUIET)
if (NOT spdlog_FOUND)
    add_subdirectory(exernalLibraries/spdlog)
endif()
message(STATUS "spdlog package avable")

find_package(imgui QUIET)
if (NOT imgui_FOUND)
    set(IMGUI_INCLUDE_DIR exernalLibraries/imgui)
    file(GLOB IMGUI_SOURCES ${IMGUI_INCLUDE_DIR}/*.cpp)
    file(GLOB IMGUI_HEADERS ${IMGUI_INCLUDE_DIR}/*.h)
    add_library(imgui STATIC ${IMGUI_SOURCES} ${IMGUI_INCLUDE_DIR}/backends/imgui_impl_glfw.cpp ${IMGUI_INCLUDE_DIR}/backends/imgui_impl_opengl3.cpp)
    target_include_directories(imgui PUBLIC ${IMGUI_INCLUDE_DIR} ${IMGUI_INCLUDE_DIR}/backends)
    if (UNIX AND NOT APPLE) # Linux
        target_link_libraries(imgui glfw -lGL -lGLEW)
    elseif (MSVC) # Windows using microsoft visual code
        if (glfw3_FOUND)
            target_link_libraries(imgui glfw glew32 -lOpenGL32)
        else()
            target_link_libraries(imgui glew32 -lOpenGL32)
        endif()
    else() # Windows
        target_link_libraries(imgui glfw glew32 -lOpenGL32) # FreeGLUT is for Modified GLM
    endif()
endif()
message(STATUS "imgui package avable")

# set a name for the executable (Expecting the name of the program to be the same as folder the code is in)
set(executable_name OpenGL-ALL)

include_directories(include exernalLibraries/stb)

add_executable(${executable_name}
        res/icon/programIcon.rc
        src/main.cpp
        src/OpenGLHelperMethods.cpp
        src/Sphere.cpp)

if (UNIX AND NOT APPLE) # Linux
    target_link_libraries(${executable_name} glm glfw -lGL -lGLEW spdlog::spdlog imgui)
elseif (MSVC) # Windows using microsoft visual code
    if (glfw3_FOUND)
        target_link_libraries(${executable_name} glfw glew32 -lOpenGL32 spdlog::spdlog imgui)
    else()
        target_link_libraries(${executable_name} glew32 -lOpenGL32 spdlog::spdlog imgui)
    endif()
else() # Windows
    target_link_libraries(${executable_name} glfw glm glew32 -lOpenGL32 spdlog::spdlog imgui) # FreeGLUT is for Modified GLM
endif()

# NOTE I do not have a mac to tests with
# external include folders
target_include_directories(${executable_name} PRIVATE exernalLibraries/stb ${IMGUI_INCLUDE_DIR} ${IMGUI_INCLUDE_DIR}/backends)
