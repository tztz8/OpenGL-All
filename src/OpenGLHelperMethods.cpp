//#include <cstdio>
#include <cstdlib>
#include <malloc.h>

#include "OpenGLHelperMethods.h"

#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>

// Logger Lib
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#include <spdlog/spdlog.h>
#include "spdlog/fmt/bin_to_hex.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "main.h"

#include <portable-file-dialogs.h>

#include "imgui.h"

std::filesystem::path UserSelectImageFile() {
    pfd::open_file fileBox("Select a Image File", ".",
                           {"Image FileS", "*.png *.jpg *.jpeg *.bmp *.tga *.psd *.gif *.hdr *.pic"});
    std::vector<std::string> selection = fileBox.result();
    SPDLOG_INFO(spdlog::fmt_lib::format("Number of selected files {}", selection.size()));
    if (selection.size() == 0) {
        return std::filesystem::path("res/textures/Earth.jpg");
    }
    SPDLOG_INFO(spdlog::fmt_lib::format("User Select Image File \"{}\"", selection[0]));
    return std::filesystem::path(selection[0]);
}

// Helper to display a little (?) mark which shows a tooltip when hovered.
// In your own code you may want to display an actual icon if you are using a merged icon fonts (see docs/FONTS.md)
void HelpMarker(const char* desc)
{
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered())
    {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
        ImGui::TextUnformatted(desc);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

/**
 * Read A file and out put a file a char list
 * @note code from Yasmin and commit and some modification make by Timbre Freeman
 * @param filename path to the file
 * @return pointer to a char list (String)
 */
std::string ReadFile(const std::filesystem::path& filename) {

    std::ifstream infile(filename);

    // is the file available and open
    if (!infile.is_open()) {
        SPDLOG_ERROR(spdlog::fmt_lib::format("ReadFile: Unable to open filepath \"{}\"", filename.string()));
        infile.close();
        return "";
    }

    std::string source;
    std::string line;
    while (infile.good()) { // are we done reading the file
        std::getline(infile, line);
        source.append(line);
        source.append("\n");
    }

    infile.close(); // close the file
    SPDLOG_INFO(spdlog::fmt_lib::format("ReadFile: \"{}\" is ready", filename.filename().string()));
    return source;
}

/**
 * Initialize Shaders
 * @note code from Yasmin and commit and some modification make by Timbre Freeman
 * @param s paths (as char strings) of the shader files, in this order <br>
 *          - Vertex (.vert) <br>
 *          - Fragment (.frag) <br>
 *          - Geometry (.geom) <br>
 * @param count number of shader files
 * @warning if count is above 3 it will exit
 * @warning if a invalid shader is pass in will call `tellWindowToClose()`
 * @return shader program
 */
GLuint initShaders(const std::filesystem::path s[], int count) {

    GLuint p = glCreateProgram();
    size_t i = 0;
    GLuint v;
    std::string shaderType;
    size_t size_t_count = static_cast<size_t>(count);

    while (i < size_t_count) {
        std::string fileExtension = s[i].extension().string();
        if (fileExtension == ".vert" || fileExtension == ".vs") {
            v = glCreateShader(GL_VERTEX_SHADER);
            shaderType = "Vertex";
        } else if (fileExtension == ".frag" || fileExtension == ".fs") {
            v = glCreateShader(GL_FRAGMENT_SHADER);
            shaderType = "Fragment";
        } else if (fileExtension == ".geom" || fileExtension == ".gs") {
            v = glCreateShader(GL_GEOMETRY_SHADER);
            shaderType = "Geometry";
        } else if (fileExtension == ".glsl" && i < 3) {
            if (i == 0) {
                v = glCreateShader(GL_VERTEX_SHADER);
                shaderType = "Vertex";
            } else if (i == 1) {
                v = glCreateShader(GL_FRAGMENT_SHADER);
                shaderType = "Fragment";
            } else { //if (i == 2) {
                v = glCreateShader(GL_GEOMETRY_SHADER);
                shaderType = "Geometry";
            }
        } else {
            SPDLOG_ERROR(spdlog::fmt_lib::format("initShaders: unknown shader pass in, s: {}", fileExtension));
//            tellWindowToClose();
            i++;
            continue;
        }

        std::string shaderString = ReadFile(s[i]);
        if (shaderString.empty()) {
            SPDLOG_ERROR(spdlog::fmt_lib::format("initShaders: {} shader file read is empty: {}", shaderType, s[i].filename().string()));
        } else {
            const GLchar* vs = shaderString.c_str();
            glShaderSource(v, 1, &vs, nullptr);
            glCompileShader(v);
            GLint compiled;

            glGetShaderiv(v, GL_COMPILE_STATUS, &compiled);
            if (!compiled) {
                GLsizei len;
                glGetShaderiv(v, GL_INFO_LOG_LENGTH, &len);

                char* log = (char*)malloc(len + 1);

                if (log == nullptr) {
                    SPDLOG_ERROR("initShaders: Was not able to get memory to get error code for compiled shader");
                    exit(EXIT_FAILURE);
                }

                glGetShaderInfoLog(v, len, &len, log);

                SPDLOG_ERROR(spdlog::fmt_lib::format("initShaders: {} Shader compilation failed: {}", shaderType, log));

                free(log);
            } else {
                SPDLOG_INFO(spdlog::fmt_lib::format("initShaders: {} shader complied", shaderType));
            }

            glAttachShader(p, v);
        }

        i++;
    }

    glLinkProgram(p);

    GLint linked;
    glGetProgramiv(p, GL_LINK_STATUS, &linked);
    if (!linked) {
        GLsizei len;
        glGetProgramiv(p, GL_INFO_LOG_LENGTH, &len);
        char* log = (char*)malloc(len + 1);

        if (log == nullptr) {
            SPDLOG_ERROR("initShaders: Was not able to get memory to get error code for compiled shader");
            exit(EXIT_FAILURE);
        }

        glGetProgramInfoLog(p, len, &len, log);
        std::string logString(log);
        logString.pop_back();
        SPDLOG_ERROR(spdlog::fmt_lib::format("initShaders: Shader linking failed: {}", logString));
        free(log);
    } else {
        SPDLOG_INFO("initShaders: shader's linked");
    }
    glUseProgram(p);
    return p;

}

/**
 * Initialize Shaders
 * @note code from Yasmin and commit and some modification make by Timbre Freeman
 * @param s paths (as char strings) of the shader files, in this order <br>
 *          - Vertex (.vert) <br>
 *          - Fragment (.frag) <br>
 *          - Geometry (.geom) <br>
 * @param count number of shader files
 * @warning if count is above 3 it will exit
 * @warning if a invalid shader is pass in will call `tellWindowToClose()`
 * @return shader program
 */
GLuint initShaders(const std::vector<std::filesystem::path>& shaderPaths) {
    return initShaders(shaderPaths.data(), static_cast<int>(shaderPaths.size()));
}

void loadGLFWIcon(GLFWwindow* thisWindow, const std::filesystem::path iconFileNamePath) {
    SPDLOG_INFO(spdlog::fmt_lib::format("Setup icon \"{}\" for the window", iconFileNamePath.filename().string()));
    GLFWimage icons[1];
    icons[0].pixels = stbi_load(
            iconFileNamePath.string().c_str(),
            &icons[0].width,
            &icons[0].height,
            nullptr, 4);
    if (icons[0].pixels == nullptr) {
        SPDLOG_ERROR(spdlog::fmt_lib::format("Unable to load icon \"{}\"", iconFileNamePath.string()));
    } else {
        glfwSetWindowIcon(thisWindow, 1, icons);
        stbi_image_free(icons[0].pixels);
    }
}

/**
 * Generates a texture that is suited for attachments to a framebuffer
 * @note code from Yasmin and commit and some modification make by Timbre Freeman
 */
GLuint generateAttachmentTexture() {
    // What enum to use?
    GLenum attachment_type = GL_RGB;

    //Generate texture ID and load texture data
    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    glTexImage2D(GL_TEXTURE_2D, 0, attachment_type, screenWidth, screenHeight, 0, attachment_type, GL_UNSIGNED_BYTE, nullptr);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    return textureID;
}

/**
 * Load Texture
 * @note code from Yasmin and commit and some modification make by Timbre Freeman
 * @note (used devil to load the image)
 * @note do not forget to uncommitted the include lib at the top and uncommitted the setup in main
 * @param filename path to image file
 * @return GL Texture ID
 */
GLuint loadTexture(std::filesystem::path filename) {
  int width, height;
  stbi_uc *image = stbi_load(filename.string().c_str(), &width, &height, nullptr, 4);
  if (image == nullptr) {
    SPDLOG_ERROR(spdlog::fmt_lib::format("Couldn't load texture file \"{}\"", filename.string()));
    tellWindowToClose();
    return 0;
  }
  GLuint tid;
  glGenTextures(1, &tid);
  glBindTexture(GL_TEXTURE_2D, tid);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, image);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

  glBindTexture(GL_TEXTURE_2D, 0);

  /* Because we have already copied image data into texture data
  we can release memory used by image. */

  stbi_image_free(image);
  SPDLOG_INFO(spdlog::fmt_lib::format("Texture \"{}\" is ready", filename.filename().string()));
  return tid;
}

/**
 * Load Cube Map Texture
 * @note code from Yasmin and commit and some modification make by Timbre Freeman
 * @note (used devil to load the image)
 * @note do not forget to uncommitted the include lib at the top and uncommitted the setup in main
 * @param posX path to posX image file
 * @param negX path to negX image file
 * @param posY path to posY image file
 * @param negY path to negY image file
 * @param posZ path to posZ image file
 * @param negZ path to negZ image file
 * @return GL Texture ID
 */
//unsigned int loadCubeMapTexture(
//        const char* posX, const char* negX,
//        const char* posY, const char* negY,
//        const char* posZ, const char* negZ) {
//    if (devILIsSetup) {
//        ILboolean success;
//        unsigned int imageID;
//        GLuint textureID;
//        GLuint faceTarget[] = {
//                GL_TEXTURE_CUBE_MAP_POSITIVE_X, GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
//                GL_TEXTURE_CUBE_MAP_POSITIVE_Y, GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
//                GL_TEXTURE_CUBE_MAP_POSITIVE_Z, GL_TEXTURE_CUBE_MAP_NEGATIVE_Z
//        };
//        const char* files[6];
//
//        files[0] = posX;
//        files[1] = negX;
//        files[2] = posY;
//        files[3] = negY;
//        files[4] = posZ;
//        files[5] = negZ;
//
//        glActiveTexture(GL_TEXTURE0);
//        glGenTextures(1, &textureID); /* Texture name generation */
//        glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);
//
//        // Load Textures for Cube Map
//
//        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
//        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
//        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
//        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
//        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
//
//
//        glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
//
//        ilGenImages(1, &imageID);
//        ilBindImage(imageID); /* Binding of DevIL image name */
//
//        for (int i = 0; i < 6; ++i) {
//            ilEnable(IL_ORIGIN_SET);
//            ilOriginFunc(IL_ORIGIN_LOWER_LEFT);
//            success = ilLoadImage(files[i]);
//
//            if (!success) {
//                fprintf(stderr, "Error: loadCubeMapTexture: Couldn't load texture file \"%s\"\n", files[i]);
//                // The operation was not sucessfull hence free image and texture
//                ilDeleteImages(1, &imageID);
//            }
//
//            /* Convert image to RGBA */
//            ilConvertImage(IL_RGBA, IL_UNSIGNED_BYTE);
//
//            /* Create and load textures to OpenGL */
//            glTexImage2D(faceTarget[i], 0, GL_RGBA,
//                         ilGetInteger(IL_IMAGE_WIDTH),
//                         ilGetInteger(IL_IMAGE_HEIGHT),
//                         0, GL_RGBA, GL_UNSIGNED_BYTE, ilGetData());
//
//            fprintf(stdout,"Info: loadCubeMapTexture: Texture Loaded: %s\n", files[i]);
//        }
//        fprintf(stdout,"Info: loadCubeMapTexture: Cube Map Loaded Successfully.\n");
//
//
//        glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
//
//        /* Because we have already copied image data into texture data we can release memory used by image. */
//        ilDeleteImages(1, &imageID);
//
//        // add information to the log
//
//        return textureID;
//    } else {
//        fprintf(stderr, "Error: loadCubeMapTexture: DevIL is not setup\n");
//        tellWindowToClose();
//        return 0;
//    }
//}

/**
 * Get the tangents from the vertices, normals, texture and indices
 * @note code from Yasmin and commit and some modification make by Timbre Freeman
 * @param vertices the vertices
 * @param normals the normals
 * @param tangents the tangents (warning tangents will be overwritten)
 * @param numvertices the number of vertices
 * @param numindices the number of indices
 * @param indices the indices
 * @param tex the texture coordinates
 * @warning tangents will be overwritten
 */
void updateVertexTangents(const glm::vec4* vertices, const glm::vec3* normals, glm::vec4* tangents, int numvertices, int numindices,  const GLushort* indices, const glm::vec2* tex) {

    glm::vec3* tan1 = (glm::vec3*)malloc(sizeof(glm::vec3) * numindices);
    glm::vec3* tan2 = (glm::vec3*)malloc(sizeof(glm::vec3) * numindices);

    if (tan1 == nullptr || tan2 == nullptr) {
        SPDLOG_ERROR("Unable to get memory to update vertex targets");
        exit(EXIT_FAILURE);
    }

    for (int k = 0; k < numindices; k++){
        tan1[k] = glm::vec3(0.0f, 0.0f, 0.0f);
        tan2[k] = glm::vec3(0.0f, 0.0f, 0.0f);
    }

    // Compute the tangent vector
    for (int i = 0; i < numindices; i += 3)
    {
        int index_0 = indices[i];
        int index_1 = indices[i + 1];
        int index_2 = indices[i + 2];

        glm::vec3 p1(vertices[index_0]);
        glm::vec3 p2(vertices[index_1]);
        glm::vec3 p3(vertices[index_2]);

        glm::vec2 tc1 = tex[index_0];
        glm::vec2 tc2 = tex[index_1];
        glm::vec2 tc3 = tex[index_2];

        glm::vec3 q1 = p2 - p1;
        glm::vec3 q2 = p3 - p1;
        float s1 = tc2.x - tc1.x;
        float s2 = tc3.x - tc1.x;
        float t1 = tc2.y - tc1.y;
        float t2 = tc3.y - tc1.y;
        float r = 1.0f / (s1 * t2 - s2 * t1);

        glm::vec3 tangents1((t2 * q1.x - t1 * q2.x) * r, (t2 * q1.y - t1 * q2.y) * r, (t2 * q1.z - t1 * q2.z) * r);
        glm::vec3 tangents2((s1 * q2.x - s2 * q1.x) * r, (s1 * q2.y - s2 * q1.y) * r, (s1 * q2.z - s2 * q1.z) * r);

        tan1[index_0] += tangents1;
        tan1[index_1] += tangents1;
        tan1[index_2] += tangents1;
        tan2[index_0] += tangents2;
        tan2[index_1] += tangents2;
        tan2[index_2] += tangents2;
    }

    for (int i = 0; i < numvertices; ++i)
    {
        glm::vec3 n = normals[i];
        glm::vec3 t1 = tan1[i];
        glm::vec3 t2 = tan2[i];

        // Gram-Schmidt orthogonalize
        tangents[i] = glm::vec4(glm::normalize(t1 - (glm::dot(n, t1) * n)), 0.0f);
        // Store handedness in w
        tangents[i].w = (glm::dot(glm::cross(n, t1), t2) < 0.0f) ? -1.0f : 1.0f;
    }

    free(tan1);
    free(tan2);
}

/**
 * Get the tangents from the vertices, normals, texture and indices
 * @note code from Yasmin and commit and some modification make by Timbre Freeman
 * @param vertices the vertices
 * @param normals the normals
 * @param tangents the tangents (warning tangents will be overwritten)
 * @param numvertices the number of vertices
 * @param numindices the number of indices
 * @param indices the indices
 * @param tex the texture coordinates
 * @warning tangents will be overwritten
 */
void updateVertexTangents(const glm::vec4* vertices, const glm::vec3* normals, glm::vec4* tangents, int numvertices, int numindices,  const GLuint* indices, const glm::vec2* tex) {

    glm::vec3* tan1 = (glm::vec3*)malloc(sizeof(glm::vec3) * numindices);
    glm::vec3* tan2 = (glm::vec3*)malloc(sizeof(glm::vec3) * numindices);

    if (tan1 == nullptr || tan2 == nullptr) {
        SPDLOG_ERROR("Unable to get memory to update vertex targets");
        exit(EXIT_FAILURE);
    }

    for (int k = 0; k < numindices; k++){
        tan1[k] = glm::vec3(0.0f, 0.0f, 0.0f);
        tan2[k] = glm::vec3(0.0f, 0.0f, 0.0f);
    }

    // Compute the tangent vector
    for (int i = 0; i < numindices; i += 3)
    {
        int index_0 = indices[i];
        int index_1 = indices[i + 1];
        int index_2 = indices[i + 2];

        glm::vec3 p1(vertices[index_0]);
        glm::vec3 p2(vertices[index_1]);
        glm::vec3 p3(vertices[index_2]);

        glm::vec2 tc1 = tex[index_0];
        glm::vec2 tc2 = tex[index_1];
        glm::vec2 tc3 = tex[index_2];

        glm::vec3 q1 = p2 - p1;
        glm::vec3 q2 = p3 - p1;
        float s1 = tc2.x - tc1.x;
        float s2 = tc3.x - tc1.x;
        float t1 = tc2.y - tc1.y;
        float t2 = tc3.y - tc1.y;
        float r = 1.0f / (s1 * t2 - s2 * t1);

        glm::vec3 tangents1((t2 * q1.x - t1 * q2.x) * r, (t2 * q1.y - t1 * q2.y) * r, (t2 * q1.z - t1 * q2.z) * r);
        glm::vec3 tangents2((s1 * q2.x - s2 * q1.x) * r, (s1 * q2.y - s2 * q1.y) * r, (s1 * q2.z - s2 * q1.z) * r);

        tan1[index_0] += tangents1;
        tan1[index_1] += tangents1;
        tan1[index_2] += tangents1;
        tan2[index_0] += tangents2;
        tan2[index_1] += tangents2;
        tan2[index_2] += tangents2;
    }

    for (int i = 0; i < numvertices; ++i)
    {
        glm::vec3 n = normals[i];
        glm::vec3 t1 = tan1[i];
        glm::vec3 t2 = tan2[i];

        // Gram-Schmidt orthogonalize
        tangents[i] = glm::vec4(glm::normalize(t1 - (glm::dot(n, t1) * n)), 0.0f);
        // Store handedness in w
        tangents[i].w = (glm::dot(glm::cross(n, t1), t2) < 0.0f) ? -1.0f : 1.0f;
    }

    free(tan1);
    free(tan2);
}

/**
 * Update All The Vertex Normals
 * @param vertices the vertices used to cal the norms
 * @param norms the norms that will be updated (WARNING the data will be overridden)
 * @param indices to know which vertices used to cal the norms
 * @param numNormals the number of normals
 * @param numIndices the number of indices
 * @warning the data in norms will be overridden
 */
void updateVertexNormals(const glm::vec3* vertices, glm::vec3* norms, const GLushort* indices,
                         GLuint numNormals, GLuint numIndices) {

    glm::vec3 p1;
    glm::vec3 p2;
    glm::vec3 p3;
    glm::vec3 n;

    for (int i = 0; i < numNormals; i++) {
        norms[i] = glm::vec3(0.0, 0.0, 0.0);
    }

    for (int index = 0; index < numIndices; index+=3) {

        p1 = vertices[indices[index + 0]];
        p2 = vertices[indices[index + 1]];
        p3 = vertices[indices[index + 2]];

        n = glm::normalize(cross((p2 - p1), (p3 - p1)));

        norms[indices[index + 0]] += n;
        norms[indices[index + 1]] += n;
        norms[indices[index + 2]] += n;
    }

    for (int i = 0; i < numNormals; i++) {
        norms[i] = glm::normalize(norms[i]);
    }
}

/**
 * Update All The Vertex Normals
 * @param vertices the vertices used to cal the norms
 * @param norms the norms that will be updated (WARNING the data will be overridden)
 * @param indices to know which vertices used to cal the norms
 * @param numNormals the number of normals
 * @param numIndices the number of indices
 * @warning the data in norms will be overridden
 */
void updateVertexNormals(const glm::vec3* vertices, glm::vec3* norms, const GLuint* indices,
                         GLuint numNormals, GLuint numIndices) {

    glm::vec3 p1;
    glm::vec3 p2;
    glm::vec3 p3;
    glm::vec3 n;

    for (int i = 0; i < numNormals; i++) {
        norms[i] = glm::vec3(0.0, 0.0, 0.0);
    }

    for (int index = 0; index < numIndices; index+=3) {

        p1 = vertices[indices[index + 0]];
        p2 = vertices[indices[index + 1]];
        p3 = vertices[indices[index + 2]];

        n = glm::normalize(cross((p2 - p1), (p3 - p1)));

        norms[indices[index + 0]] += n;
        norms[indices[index + 1]] += n;
        norms[indices[index + 2]] += n;
    }

    for (int i = 0; i < numNormals; i++) {
        norms[i] = glm::normalize(norms[i]);
    }
}

/**
 * Center and make the model fit in a -1 to 1 box
 * @param vertices the vertices to update
 * @param numVertices the number of vertices
 * @warning the data in vertices will be overridden
 */
void unitizeModel(glm::vec3 vertices[], GLuint numVertices) {
    // Step 1: Compute the maximum and minimum of x, y, and z
    // coordinates of the model’s vertices.
    float min_x;
    float max_x;
    float min_y;
    float max_y;
    float min_z;
    float max_z;
    min_x = max_x = vertices[0].x;
    min_y = max_y = vertices[0].y;
    min_z = max_z = vertices[0].z;

    // finding the min and max for xyz
    for (int i = 0; i < numVertices; ++i) {
        glm::vec3 vertex = vertices[i];
        if (vertex.x < min_x) {
            min_x = vertex.x;
        } else if (vertex.x > max_x) {
            max_x = vertex.x;
        }

        if (vertex.y < min_y) {
            min_y = vertex.y;
        } else if (vertex.y > max_y) {
            max_y = vertex.y;
        }

        if (vertex.z < min_z) {
            min_z = vertex.z;
        } else if (vertex.z > max_z) {
            max_z = vertex.z;
        }
    }

    // Step 2: Calculate the center as follows:
    float center_x = (max_x + min_x) / 2;
    float center_y = (max_y + min_y) / 2;
    float center_z = (max_z + min_z) / 2;

    // Step 3: Calculate the width, height, and depth of the model.
    float width = std::abs(max_x - min_x);
    float height = std::abs(max_y - min_y);
    float depth = std::abs(max_z - min_z);

    // Step 4: Calculate the scale factor!.
    float scale = glm::max(depth, glm::max(width, height));

    for (int i = 0; i < numVertices; i++) {
        // Step 5: Center the model at the origin!
        // moving points to center of the screen
        vertices[i].x -= center_x;
        vertices[i].y -= center_y;
        vertices[i].z -= center_z;

        // Step 6:Divide each coordinate by the scale factor determined
        // in Step 4! This will fit the model in a bounding box of width,
        // height, and depth 1 each extending from -0.5 to +0.5
        // scale the model to fit in a box whose width, height, and depth extend from -0.5 to 0.5.
        vertices[i].x /= scale;
        vertices[i].y /= scale;
        vertices[i].z /= scale;

        // Step 7: Multiply each coordinate by 2.0! This will fit the
        // model in a bounding box of width, height, and depth 2 each
        // extending from -1.0 to +1.0

        //scale the model to fit in a box whose width, height, and depth extend from -1.0 to 1.0.
        vertices[i].x *= 2;
        vertices[i].y *= 2;
        vertices[i].z *= 2;
    }
}

#if OPENGL_DEBUG_FOR_GLFW
/**
 * OpenGL Debug message callback to output
 * @param source Where the error came from
 * @param type The type of error
 * @param id the id of the error
 * @param severity how bad the error was
 * @param length Unknown (my guest is the size of message)
 * @param message the message of the error
 * @param userParam Unknown
 * @see https://learnopengl.com/In-Practice/Debugging
 * @warning if severity is high tell glfw to close
 */
void APIENTRY glDebugOutput(GLenum source,
                            GLenum type,
                            unsigned int id,
                            GLenum severity,
                            [[maybe_unused]] GLsizei length,
                            const char *message,
                            [[maybe_unused]] const void *userParam)
{
    // ignore non-significant error/warning codes
    if(id == 131169 || id == 131185 || id == 131218 || id == 131204) return;
//    std::cout << "Info: OpenGL Debug incoming (ID:" << id << ")" << std::endl;

    std::string sourceMessage;
    std::string typeMessage;
    std::string severityMessage;

    switch (source)
    {
        case GL_DEBUG_SOURCE_API:             sourceMessage = "Source: API"; break;
        case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   sourceMessage = "Source: Window System"; break;
        case GL_DEBUG_SOURCE_SHADER_COMPILER: sourceMessage = "Source: Shader Compiler"; break;
        case GL_DEBUG_SOURCE_THIRD_PARTY:     sourceMessage = "Source: Third Party"; break;
        case GL_DEBUG_SOURCE_APPLICATION:     sourceMessage = "Source: Application"; break;
        case GL_DEBUG_SOURCE_OTHER:           sourceMessage = "Source: Other"; break;
        default:                              sourceMessage = "Source: Unknown"; break;
    }

    switch (type)
    {
        case GL_DEBUG_TYPE_ERROR:               typeMessage = "Type: Error"; break;
        case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: typeMessage = "Type: Deprecated Behaviour"; break;
        case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  typeMessage = "Type: Undefined Behaviour"; break;
        case GL_DEBUG_TYPE_PORTABILITY:         typeMessage = "Type: Portability"; break;
        case GL_DEBUG_TYPE_PERFORMANCE:         typeMessage = "Type: Performance"; break;
        case GL_DEBUG_TYPE_MARKER:              typeMessage = "Type: Marker"; break;
        case GL_DEBUG_TYPE_PUSH_GROUP:          typeMessage = "Type: Push Group"; break;
        case GL_DEBUG_TYPE_POP_GROUP:           typeMessage = "Type: Pop Group"; break;
        case GL_DEBUG_TYPE_OTHER:               typeMessage = "Type: Other"; break;
        default:                                typeMessage = "Type: Unknown"; break;
    }

    switch (severity)
    {
        case GL_DEBUG_SEVERITY_HIGH:         severityMessage = "Severity: high"; break;
        case GL_DEBUG_SEVERITY_MEDIUM:       severityMessage = "Severity: medium"; break;
        case GL_DEBUG_SEVERITY_LOW:          severityMessage = "Severity: low"; break;
        case GL_DEBUG_SEVERITY_NOTIFICATION: severityMessage = "Severity: notification"; break;
        default:                             severityMessage = "Severity: Unknown"; break;
    }

    if (severity == GL_DEBUG_SEVERITY_NOTIFICATION) {
        std::string s(message);
        s.erase(s.find_last_not_of(" \n\r\t")+1);
        SPDLOG_INFO(spdlog::fmt_lib::format("OpenGL Debug: "
                                            "{}, {}, {}, "
                                            "Debug message (ID:{:0d}): {}",
                                            sourceMessage,
                                            typeMessage,
                                            severityMessage,
                                            id, s));
    } else if (severity == GL_DEBUG_SEVERITY_HIGH || severity == GL_DEBUG_SEVERITY_MEDIUM) {
        std::string s(message);
        s.erase(s.find_last_not_of(" \n\r\t")+1);
        SPDLOG_ERROR(spdlog::fmt_lib::format("OpenGL Debug: message (ID:{:0d}): {}", id, s));
        SPDLOG_ERROR(spdlog::fmt_lib::format("OpenGL Debug: {}", sourceMessage));
        SPDLOG_ERROR(spdlog::fmt_lib::format("OpenGL Debug: {}", typeMessage));
        SPDLOG_ERROR(spdlog::fmt_lib::format("OpenGL Debug: {}", severityMessage));
    } else {
        std::string s(message);
        s.erase(s.find_last_not_of(" \n\r\t")+1);
        SPDLOG_DEBUG(spdlog::fmt_lib::format("OpenGL Debug: "
                                             "{}, {}, {}, "
                                             "Debug message (ID:{:0d}): {}",
                                             sourceMessage,
                                             typeMessage,
                                             severityMessage,
                                             id, s));
    }

    if (severity == GL_DEBUG_SEVERITY_HIGH) {
        tellWindowToClose();
    }
}
#endif
