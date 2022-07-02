/**
 * @brief Holds the definshon of Sphere Class
 * @author Timbre Freeman (tztz8)
 */

#ifndef EWU_CSCD470_ASSIGNMENT4_SPHERE_H
#define EWU_CSCD470_ASSIGNMENT4_SPHERE_H

/**
 * @brief Sphere Class for OpenGL
 * This is a normal sphere that also setup for textuers
 */
class Sphere {
private:
    unsigned int sphere_vao;
    int step;
    int numVertices;
    int numTriangles;
    int numIndices;
public:
    /**
     * Make a class with a setNumber of step (detile of the sphere)
     * @brief Construct a new Sphere object
     * @param step number of steps for the sphere
     */
    Sphere(int step);
    /**
     * Call the openGL commands to draw the Sphere
     * @brief draw using opengl
     */
    void draw();
    /**
     * Make and gives all the data needed for OpenGL to OpenGL (GPU)
     * @brief setup the data with OpenGL
     * @warning this must wait for after glewInit()
     * @see glewInit()
     */
    void create();
    void updateStep(int step);
    int getStep();
};


#endif //EWU_CSCD470_ASSIGNMENT4_SPHERE_H
