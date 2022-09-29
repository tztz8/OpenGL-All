//
// Created by tztz8 on 9/26/22.
//

#include "ModelLoader.h"
#include "OpenGLHelperMethods.h"

void firstPass(SimpleModel *pModel, std::ifstream *ifstream);

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#include <spdlog/spdlog.h>
#include <fstream>
#include <regex>

SimpleModel* readOBJ(std::filesystem::path filename) {

    SPDLOG_INFO(spdlog::fmt_lib::format("readOBJ: Start load of filepath \"{}\"", filename.string()));

    SimpleModel *model;

    // Open File
    std::ifstream modelFile(filename);
    // is the file available and open
    if (!modelFile.is_open()) {
        SPDLOG_ERROR(spdlog::fmt_lib::format("readOBJ: Unable to open filepath \"{}\"", filename.string()));
        modelFile.close();
        return nullptr;
    }
    // is the file empty
    if (!modelFile.good()) {
        SPDLOG_ERROR(spdlog::fmt_lib::format("readOBJ: model file is empty: {}",
                                             filename.filename().string()));
        return nullptr;
    }

    // allocate a new model
    model = (SimpleModel *) malloc(sizeof(SimpleModel));

    model->numvertices = 0;
    model->vertices = nullptr;
    model->numnormals = 0;
    model->normals = nullptr;
    model->numtextures = 0;
    model->textures = nullptr;
    model->numtriangles = 0;

    // make a first pass through the file to get a count of the number
    // of vertices, normals, texcoords & triangles
    firstPass(model, &modelFile);

    modelFile.close();
    SPDLOG_INFO(spdlog::fmt_lib::format("readOBJ: \"{}\" is ready", filename.filename().string()));
    return model;
}

void firstPass(SimpleModel *pModel, std::ifstream *ifstream) {
    GLuint numvertices;        /* number of vertices in model */
    GLuint numnormals;         /* number of normals in model */
    GLuint numtextures;         /* number of textures in model */
    GLuint numtriangles;       /* number of triangles in model */
    std::vector<glm::vec3> temp_vertices;
    std::vector<glm::vec3> temp_normals;
    std::vector<glm::vec2> temp_textures;
    std::vector<GLuint> temp_indices;

    int v, n, t;

    numvertices = numnormals = numtextures = numtriangles = 0;

    std::string line;
    while(ifstream->good()) {
        std::getline(*ifstream, line);
        if (!line.empty()) {
            switch(line.at(0)) {
                case 'v':
                    switch(line.at(1)) {
                        case '\0': // vertex
//                            std::basic_regex vertex(R"(\d \d \d)");
                            numvertices++;
                            break;
                        case 'n': // normal
                            numnormals++;
                            break;
                        case 't': // texture
                            numtextures++;
                            break;
                    }
                    break;
                case 'f': // face
                    //std::regex face("%d//%d");
                    // TODO: fix
                    numtriangles++;
                    break;
                default:
                    break;
            }
        }
    }

    pModel->numvertices = numvertices;
    pModel->numnormals = numnormals;
    pModel->numtextures = numtextures;

    if (numvertices != numnormals || numvertices != numtextures) {
        SPDLOG_WARN(spdlog::fmt_lib::format(
                "firstPass: The number of vertices done not match v:{}, vn:{}, vt:{}",
                numvertices, numnormals, numtextures));
    }

    pModel->numtriangles = numtriangles;
    pModel->numindices = numtriangles * 3;
}
