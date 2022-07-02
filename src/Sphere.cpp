//
// Created by tztz8 on 5/16/22.
//

#include "Sphere.h"
//#include <random>
#include <vector>
#include <cstdio>
#include <GL/glew.h>
#define GLM_FORCE_RADIANS
#include <glm/mat4x4.hpp> // glm::mat4
#include <glm/gtc/matrix_transform.hpp> // glm::translate, glm::rotate, glm::scale, glm::perspective
#include "OpenGLHelperMethods.h"

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#include <spdlog/spdlog.h>


const double kPI = 3.1415926535897932384626433832795;

Sphere::Sphere(int step) :
    step(step),
    numVertices((step + 1) * (step + 1)),
    numTriangles((2 * step * (step - 1))),
    numIndices((6 * step * step)),
    sphere_vao(0) // setting 0 so it is initialized
    {
    // TODO: Debug
    SPDLOG_DEBUG(spdlog::fmt_lib::format("make using step: {}, numVertices: {}, numTriangles: {}, numIndices: {}, do not forget to call create",
                                         this->step,
                                         this->numVertices,
                                         this->numTriangles,
                                         this->numIndices));
//    fprintf(stdout,
//            "Debug: Sphere: Class make using isBasket: %s, step: %d, numVertices: %d, numTriangles: %d, numIndices: %d, do not forget to call create\n",
//            this->isBasket ? "true" : "false",
//            this->step,
//            this->numVertices,
//            this->numTriangles,
//            this->numIndices);
}

void Sphere::create() {
    std::vector<GLushort> indices(this->numIndices);
    std::vector<glm::vec4> points(this->numVertices);
    std::vector<glm::vec3> normals(this->numVertices);
    std::vector<glm::vec2> TexCoord(this->numVertices);

    double theta = 0.0;
    double phi = 0.0;
    size_t i = 0;

    // Generate Vertices (Points)
    for (double b = -this->step / 2.0; b <= this->step / 2.0; b++) {
            theta = (1.0 * b / this->step) * kPI;
            for (int a = 0; a <= this->step; ++a) {
                phi = (1.0 * a / this->step) * 2 * kPI;
                points[i] = glm::vec4(
                        cos(theta) * sin(phi),
                        sin(theta),
                        cos(theta) * cos(phi),
                        1.0
                );
                TexCoord[i] = glm::vec2(
                        1.0 * a / this->step,
                        ((sin(theta) / 2.0) + 0.5)
                );
                i++;
            }
        }

    // TODO: Debug
    SPDLOG_DEBUG(spdlog::fmt_lib::format("NumVertices: {}, i: {}", this->numVertices, i));

    size_t size_t_numVertices = static_cast<size_t>(this->numVertices);
    for (size_t j = 0; j < size_t_numVertices; ++j) {
        // Generate Vertices (normal)
        normals[j] = glm::normalize(glm::vec3(points[j]));
    }

    size_t index = 0;
    for (int j = 0; j < (this->numVertices-this->step-1); j += this->step + 1) {
        for (int k = j; k < (j + this->step); ++k) {
            indices[index++] = k;
            indices[index++] = k + this->step + 1;
            indices[index++] = (k + 1) + this->step + 1;

            indices[index++] = k;
            indices[index++] = (k + 1) + this->step + 1;
            indices[index++] = k + 1;
        }
    }

    std::vector<glm::vec4> tangents(this->numVertices);
    updateVertexTangents(points.data(), normals.data(), tangents.data(),
                         this->numVertices, this->numIndices,
                        indices.data(), TexCoord.data());
    // TODO: Debug
//    fprintf(stdout, "Debug: Sphere Class: NumVertices: %d, tangents size: %zu\n", this->numVertices, tangents.size());
//    for (int j = 0; j < tangents.size(); ++j) {
//        fprintf(stdout, "Debug: Sphere Class: tangents%d: <%f, %f, %f, %f>\n", j,
//                tangents[j].x, tangents[j].y, tangents[j].z, tangents[j].w);
//    }

    // TODO: Debug
    SPDLOG_DEBUG(spdlog::fmt_lib::format("NumIndices: {}, index: {}", this->numIndices, index));

    glGenVertexArrays(1, &this->sphere_vao);
    glBindVertexArray(this->sphere_vao);

    unsigned int handle[5];
    glGenBuffers(5, handle);

    glBindBuffer(GL_ARRAY_BUFFER, handle[0]);
    glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(glm::vec4), points.data(), GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)0, 4, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);  // Vertex position

    glBindBuffer(GL_ARRAY_BUFFER, handle[1]);
    glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(glm::vec3), normals.data(), GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(1);  // Vertex normal

    glBindBuffer(GL_ARRAY_BUFFER, handle[2]);
    glBufferData(GL_ARRAY_BUFFER, TexCoord.size() * sizeof(glm::vec2), TexCoord.data(), GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)2, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(2);  // Vertex texture

    glBindBuffer(GL_ARRAY_BUFFER, handle[3]);
    glBufferData(GL_ARRAY_BUFFER, tangents.size() * sizeof(glm::vec4), tangents.data(), GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)3, 4, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(3);  // Vertex tangents

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, handle[4]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLushort), indices.data(), GL_STATIC_DRAW);

    glBindVertexArray(0);
}

void Sphere::draw() {
    glBindVertexArray(this->sphere_vao);
    glDrawElements(GL_TRIANGLES, this->numIndices, GL_UNSIGNED_SHORT, 0);
//    glDrawArrays(GL_POINTS, 0, NUMVERTICES);
}

void Sphere::updateStep(int step) {
    glDeleteVertexArrays(1, &this->sphere_vao);
    this->step = step,
    this->numVertices = ((step + 1) * (step + 1));
    this->numTriangles = ((2 * step * (step - 1)));
    this->numIndices = ((6 * step * step));
    SPDLOG_DEBUG(spdlog::fmt_lib::format("make using step: {}, numVertices: {}, numTriangles: {}, numIndices: {}, do not forget to call create",
                                         this->step,
                                         this->numVertices,
                                         this->numTriangles,
                                         this->numIndices));
    this->create();
}

int Sphere::getStep() {
    return this->step;
}
