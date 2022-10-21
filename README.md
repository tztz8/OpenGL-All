# OpenGL All

A place to put all I lean of OpenGL but using Cpp insead of C


THis is a chage

## Directory structure

- Quick Documentation about this repo [(README.md)](README.md) 
- Source directory [(src)](src)
  - Start of the program [(main.cpp)](src/main.cpp)
  - Methods I need [(OpenGLHelperMethods.cpp)](src/OpenGLHelperMethods.cpp)
  - Sphere Class [(Sphere.cpp)](src/Sphere.cpp)
- Resources folder [(res)](res)
  - icon folder
  - shaders
    - Fragment Shader [(shader.frag)](res/shaders/shader.frag)
    - Vertex Shader [(shader.vert)](res/shaders/shader.vert)
  - textures
    - earth [(Earth.jpg)](res/textures/Earth.jpg)
- Exernal Libraries [(exernalLibraries)](exernalLibraries)
  - glew (opengl)
  - glfw (window manager)
  - glm (graphic math lib)
  - imgui (for later to make a gui)
  - spdlog (logging lib)
  - stb (image lib)
- Header files [(include)](include)
- Test sources [(test)](test)
  - TODO: add tests

```mermaid
classDiagram
    Main <|-- Cube
    Main <|-- Sphere
    Main <|-- OpenGLHelperMethods
    Cube <|-- OpenGLHelperMethods
    Sphere <|-- OpenGLHelperMethods
    class OpenGLHelperMethods {
      +UserSelectImageFile()
      +HelpMarker(char * desc)
      +ReadFile(path & filename)
      +initShaders(path s[], int count)
      +initShaders(vector<path> & shaderPaths)
      +generateAttachmentTexture()
      +loadGLFWIcon(GLFWwindow* thisWindow, path iconFileNamePath)
      +loadTexture(path filename)
      +size int
    }
```