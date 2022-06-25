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

# set a name for the executable (Expecting the name of the program to be the same as folder the code is in)
set(executable_name OpenGL-ALL)

include_directories(include exernalLibraries/stb)

add_executable(${executable_name}
        src/main.cpp
        src/OpenGLHelperMethods.cpp
        src/Sphere.cpp)

if (UNIX AND NOT APPLE) # Linux
    target_link_libraries(${executable_name} glm glfw -lGL -lGLEW spdlog::spdlog)
elseif (MSVC) # Windows using microsoft visual code
    if (glfw3_FOUND)
        target_link_libraries(${executable_name} glfw glew32 -lOpenGL32 spdlog::spdlog)
    else()
        target_link_libraries(${executable_name} glew32 -lOpenGL32 spdlog::spdlog)
    endif()
else() # Windows
    target_link_libraries(${executable_name} glfw glm glew32 -lOpenGL32 spdlog::spdlog) # FreeGLUT is for Modified GLM
endif()

# NOTE I do not have a mac to tests with
# external include folders
target_include_directories(${executable_name} PRIVATE exernalLibraries/stb)
