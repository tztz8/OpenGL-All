//
// Created by tztz8 on 5/16/22.
//

#ifndef EWU_CSCD470_ASSIGNMENT4_SPHERE_H
#define EWU_CSCD470_ASSIGNMENT4_SPHERE_H


class Sphere {
private:
    unsigned int sphere_vao;
    const int step;
    const int numVertices;
    const int numTriangles;
    const int numIndices;
public:
    /**
     * Create the Sphere class
     * @param step number of steps for the sphere
     */
    Sphere(int step);
    /**
     * Call the openGL commands to draw the Sphere
     */
    void draw();
    /**
     * Make all the data needed for OpenGL
     * @warning this must wait for after glewInit()
     * @see glewInit()
     */
    void create();
};


#endif //EWU_CSCD470_ASSIGNMENT4_SPHERE_H
