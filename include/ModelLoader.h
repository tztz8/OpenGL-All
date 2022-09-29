//
// Created by tztz8 on 9/26/22.
//

#ifndef OPENGL_ALL_MODELLOADER_H
#define OPENGL_ALL_MODELLOADER_H

#include <GL/glew.h>
#include <filesystem>
#include "glm/vec3.hpp"
#include "glm/vec2.hpp"

using SimpleModel = struct _SimpleModel {
    GLuint numvertices;         /* number of vertices in model */
    glm::vec3 *vertices;         /* array of vertices  */


    GLuint numnormals;          /* number of normals in model */
    glm::vec3 *normals;        /* array of normals */

    GLuint numtextures;          /* number of textures in model */
    glm::vec2 *textures;        /* array of tex coordinates */

    GLuint numtriangles;    /* number of triangles in model */

    GLuint numindices;
    GLuint *indices;

    GLuint vao;
    GLuint vbo;
    GLuint ebo;
};

SimpleModel* readOBJ(std::filesystem::path filename);


#endif //OPENGL_ALL_MODELLOADER_H
