
#ifndef _VBO_H_IS_DEF_
#define _VBO_H_IS_DEF_

#ifdef __APPLE__
#include <OpenGL/gl.h>          // Header File For The OpenGL32 Library
#include <OpenGL/glu.h>         // Header File For The GLu32 Library
#include <GLUT/glut.h>          // Header File For The GLut Library
#else
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>          // Header File For The OpenGL32 Library
#include <GL/glu.h>         // Header File For The GLu32 Library
#include <GL/glut.h>          // Header File For The GLut Library
#endif

// In _SPHERE_MODE_, atoms are displayed as a mesh of (many) triangles
// and look pretty nice. However, this mode incurs a high memory
// consumption on the GPU.
//
// If _SPHERE_MODE_ is undefined, then the display falls back to an
// ugly point-based rendering.
//

#define _SPHERE_MODE_

typedef enum { SPHERE_SKIN, PACMAN_SKIN } atom_skin_t;

#define MAX_VERTICES   (1024*1024)

extern GLfloat vbo_vertex[];
extern GLfloat vbo_normal[];
extern GLfloat vbo_color[];

extern GLuint triangles_index[];
extern GLuint nb_indexes;

extern GLuint vbovid;

extern GLuint nb_vertices;

extern GLuint vertices_per_atom;

void setAtomsSkin(atom_skin_t skin);
void addAtom(GLfloat cx, GLfloat cy, GLfloat cz, double radius);
void buildVBO();
void clearVBO();
void renderAtoms();
void updateAtomCoordinatesToAccel(void);
void vboFinalize();

#endif
