
#define _XOPEN_SOURCE 600

#include "atom.h"
#include "vbo.h"
#include "ocl.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <float.h>

#define SEGMENTS 14

GLfloat vbo_vertex[MAX_VERTICES*3];
GLfloat vbo_normal[MAX_VERTICES*3];
GLfloat vbo_color[MAX_VERTICES*3];

GLuint tridx[MAX_VERTICES*2]; //approx
GLuint nb_indexes = 0;

GLuint nb_vertices = 0;
GLuint vertices_per_atom = 0;

static GLuint vi = 0;
static GLuint ni = 0;
static GLuint ci = 0;

GLuint vbovid;
static GLuint vbonid, vbocid, vboidx;

atom_skin_t atom_skin = SPHERE_SKIN;

static float Ratom = 0;
static float Gatom = 250.0/255;
static float Batom = 250.0/255;


void vboFinalize()
{
  glDeleteBuffers(1, &vbonid);
  glDeleteBuffers(1, &vbocid);
  glDeleteBuffers(1, &vboidx);
  glDeleteBuffers(1, &vbovid);
}

void setAtomsSkin(atom_skin_t skin)
{
  atom_skin = skin;
}

void clearVBO()
{
  nb_vertices = 0;
  nb_indexes = 0;
  vi = 0;
  ni = 0;
  ci = 0;
}

void buildVBO()
{
  glGenBuffers(1, &vbocid);
  glBindBuffer(GL_ARRAY_BUFFER, vbocid);
  glColorPointer(3, GL_FLOAT, 0, 0);
  glBufferData(GL_ARRAY_BUFFER, nb_vertices*3*sizeof(float), vbo_color, GL_STATIC_DRAW);

#ifdef _SPHERE_MODE_
  glGenBuffers(1, &vbonid);
  glBindBuffer(GL_ARRAY_BUFFER, vbonid);
  glNormalPointer(GL_FLOAT, 3*sizeof(float), 0);
  glBufferData(GL_ARRAY_BUFFER, nb_vertices*3*sizeof(float), vbo_normal, GL_STATIC_DRAW);
#endif

  glGenBuffers(1, &vbovid);
  glBindBuffer(GL_ARRAY_BUFFER, vbovid);
  glVertexPointer(3, GL_FLOAT, 0, 0);
  glBufferData(GL_ARRAY_BUFFER, nb_vertices*3*sizeof(float), NULL, GL_STATIC_DRAW);

#ifdef _SPHERE_MODE_
  glGenBuffers(1, &vboidx);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboidx);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, nb_indexes*sizeof(GLuint), tridx, GL_DYNAMIC_DRAW);
#endif

  glEnableClientState(GL_COLOR_ARRAY);
#ifdef _SPHERE_MODE_
  glEnableClientState(GL_NORMAL_ARRAY);
#endif
  glEnableClientState(GL_VERTEX_ARRAY);
}

void renderAtoms()
{
#ifdef _SPHERE_MODE_
  glEnableClientState(GL_COLOR_ARRAY);
  glEnableClientState(GL_NORMAL_ARRAY);
  glEnableClientState(GL_VERTEX_ARRAY);

  glDrawElements(GL_TRIANGLES, nb_indexes, GL_UNSIGNED_INT, 0);

  glDisableClientState(GL_COLOR_ARRAY);
  glDisableClientState(GL_NORMAL_ARRAY);
  glDisableClientState(GL_VERTEX_ARRAY);

#else
  glEnableClientState(GL_COLOR_ARRAY);
  glEnableClientState(GL_VERTEX_ARRAY);

  glPointSize(5);
  glDrawArrays(GL_POINTS, 0, nb_vertices);

  glDisableClientState(GL_COLOR_ARRAY);
  glDisableClientState(GL_VERTEX_ARRAY);
#endif
}

// Called after skin has been changed, so update vertices, triangles, normals and colors
void updateAtomCoordinatesToAccel(void)
{
  // Update VBO buffer
  ocl_updateVBOFromHost();

#ifdef _SPHERE_MODE_
  // Update triangle buffer
  //
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboidx);
  glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, nb_indexes*sizeof(GLuint), tridx);
#endif

  // Update color buffer
  //
  glBindBuffer(GL_ARRAY_BUFFER, vbocid);
  glBufferSubData(GL_ARRAY_BUFFER, 0, nb_vertices*3*sizeof(float), vbo_color);

#ifdef _SPHERE_MODE_
  // Update normal buffer
  //
  glBindBuffer(GL_ARRAY_BUFFER, vbonid);
  glBufferSubData(GL_ARRAY_BUFFER, 0, nb_vertices*3*sizeof(float), vbo_normal);
#endif
}

static void addVertice(GLfloat x, GLfloat y, GLfloat z, GLfloat nx, GLfloat ny, GLfloat nz)
{
  vbo_vertex[vi++] = x;
  vbo_vertex[vi++] = y;
  vbo_vertex[vi++] = z;
  //vbo_vertex[vi++] = 1.0f;

#ifdef _SPHERE_MODE_
  vbo_normal[ni++] = nx;
  vbo_normal[ni++] = ny;
  vbo_normal[ni++] = nz;
  //  vbo_normal[ni++] = 1.0f;
#endif

  vbo_color[ci++] = Ratom;
  vbo_color[ci++] = Gatom;
  vbo_color[ci++] = Batom;
  //  vbo_color[ci++] = 1.0f;
}

static void addSphere(GLfloat cx, GLfloat cy, GLfloat cz, double radius)
{
  int i,j;
  float lon,lat;
  float loninc,latinc;
  float x,y,z;
  GLuint bl, br, tl, tr;
  GLuint vertices = nb_vertices;

  loninc = 2*M_PI/(SEGMENTS);
  latinc = M_PI/SEGMENTS;

  bl = nb_vertices;

  // Create SEGMENTS/2-1 fake points to match #vertices of ghosts...
  for(i=0; i<=SEGMENTS/2; i++) {
    addVertice(cx, cy-radius, cz, 0, -1, 0);
    nb_vertices++;
  }

  lon = 0;
  lat = -M_PI/2 + latinc;
  y = sin(lat);
  for (i=0; i<=SEGMENTS-1; i++) {
    x = cos(lon)*cos(lat);
    z = -sin(lon)*cos(lat);

    addVertice(cx+x*radius, cy+y*radius, cz+z*radius, x, y, z);

    if(i == 0)
      tl = nb_vertices;
    else {
      tr = nb_vertices;
      tridx[nb_indexes++] = bl;
      tridx[nb_indexes++] = tl;
      tridx[nb_indexes++] = tr;
      tl = tr;
    }

    nb_vertices++;
    lon += loninc;
  }

  tridx[nb_indexes++] = bl;
  tridx[nb_indexes++] = tl;
  tridx[nb_indexes++] = nb_vertices-SEGMENTS;

  for (j=1; j<SEGMENTS-1; j++) { 
    lon = 0;
    lat += latinc;
    for (i=0; i<SEGMENTS; i++) {
 
      x = cos(lon)*cos(lat);
      y = sin(lat);
      z = -sin(lon)*cos(lat);

      addVertice(cx+x*radius, cy+y*radius, cz+z*radius, x, y, z);

      if(i == 0) {
	bl = nb_vertices-SEGMENTS;
	tl = nb_vertices;
      } else {
	tr = nb_vertices;
	br = bl+1;
	tridx[nb_indexes++] = bl;
	tridx[nb_indexes++] = tl;
	tridx[nb_indexes++] = br;

	tridx[nb_indexes++] = tl;
	tridx[nb_indexes++] = tr;
	tridx[nb_indexes++] = br;

	tl++;
	bl++;
      }

      nb_vertices++;
      lon += loninc;
    }

    tr = nb_vertices - SEGMENTS;
    br = tr - SEGMENTS;

    tridx[nb_indexes++] = bl;
    tridx[nb_indexes++] = tl;
    tridx[nb_indexes++] = br;

    tridx[nb_indexes++] = tl;
    tridx[nb_indexes++] = tr;
    tridx[nb_indexes++] = br;

    // Duplicate Equator (for the eating effect)
    if(j == SEGMENTS/2-1) {
      lon = 0;
      for (i=0; i<SEGMENTS; i++) {
	x = cos(lon)*cos(lat);
	y = sin(lat);
	z = -sin(lon)*cos(lat);

	addVertice(cx+x*radius, cy+y*radius, cz+z*radius, x, y, z);
	nb_vertices++;

	lon += loninc;
      }
    }
  }

  tl = nb_vertices;

  for (i=0; i<SEGMENTS; i++) {

    if(i == 0)
      bl = nb_vertices - SEGMENTS;
    else {
      br = bl+1;
      tridx[nb_indexes++] = tl;
      tridx[nb_indexes++] = bl;
      tridx[nb_indexes++] = br;
      bl = br;
    }
  }

  tridx[nb_indexes++] = tl;
  tridx[nb_indexes++] = bl;
  tridx[nb_indexes++] = nb_vertices-SEGMENTS;

  // Create SEGMENTS/2-1 fake points to match #vertices of ghosts...
  for(i=0; i<=SEGMENTS/2; i++) {
    addVertice(cx, cy+radius, cz, 0, 1, 0);
    nb_vertices++;
  }

  vertices_per_atom = nb_vertices - vertices;
}

static void addGhost(GLfloat cx, GLfloat cy, GLfloat cz, double radius)
{
  int i,j;
  float lon,lat;
  float loninc,latinc;
  float x,y,z;
  GLuint bl, br, tl, tr;
  GLuint vertices = nb_vertices;

  loninc = 2*M_PI/(SEGMENTS);
  latinc = M_PI/SEGMENTS;

  tl = nb_vertices;

  addVertice(cx, cy+radius, cz, 0, 1, 0);
  nb_vertices++;

  lon = 0;
  lat = M_PI/2 - latinc;
  y = sin(lat);
  for (i=0; i<SEGMENTS; i++) {
    x = cos(lon)*cos(lat);
    z = -sin(lon)*cos(lat);

    addVertice(cx+x*radius, cy+y*radius, cz+z*radius, x, y, z);

    if(i == 0)
      bl = nb_vertices;
    else {
      br = nb_vertices;
      tridx[nb_indexes++] = tl;
      tridx[nb_indexes++] = bl;
      tridx[nb_indexes++] = br;
      bl = br;
    }

    nb_vertices++;
    lon += loninc;
  }

  tridx[nb_indexes++] = tl;
  tridx[nb_indexes++] = bl;
  tridx[nb_indexes++] = nb_vertices-SEGMENTS;
 
  for (j=1; j<SEGMENTS/2; j++) { 
    lon = 0;
    lat -= latinc;
    for (i=0; i<SEGMENTS; i++) {
 
      x = cos(lon)*cos(lat);
      y = sin(lat);
      z = -sin(lon)*cos(lat);

      addVertice(cx+x*radius, cy+y*radius, cz+z*radius, x, y, z);

      if(i == 0) {
	tl = nb_vertices-SEGMENTS;
	bl = nb_vertices;
      } else {
	br = nb_vertices;
	tr = tl+1;
	tridx[nb_indexes++] = bl;
	tridx[nb_indexes++] = tl;
	tridx[nb_indexes++] = br;

	tridx[nb_indexes++] = tl;
	tridx[nb_indexes++] = tr;
	tridx[nb_indexes++] = br;

	tl++;
	bl++;
      }

      nb_vertices++;
      lon += loninc;
    }

    br = nb_vertices - SEGMENTS;
    tr = br - SEGMENTS;

    tridx[nb_indexes++] = bl;
    tridx[nb_indexes++] = tl;
    tridx[nb_indexes++] = br;

    tridx[nb_indexes++] = tl;
    tridx[nb_indexes++] = tr;
    tridx[nb_indexes++] = br;
  }

  // Duplicate equator
  lon = 0;
  for (i=0; i<SEGMENTS; i++) {
    x = cos(lon)*cos(lat);
    y = sin(lat);
    z = -sin(lon)*cos(lat);

    addVertice(cx+x*radius, cy+y*radius, cz+z*radius, x, y, z);
    nb_vertices++;

    lon += loninc;
  }

  for (; j<SEGMENTS-1; j++) { 
    lon = 0;
    y -= 0.15;
    for (i=0; i<SEGMENTS; i++) {
 
      x = cos(lon)*cos(lat);
      z = -sin(lon)*cos(lat);

      addVertice(cx+x*radius, cy+y*radius, cz+z*radius, x, y, z);

      if(i == 0) {
	tl = nb_vertices-SEGMENTS;
	bl = nb_vertices;
      } else {
	br = nb_vertices;
	tr = tl+1;
	tridx[nb_indexes++] = bl;
	tridx[nb_indexes++] = tl;
	tridx[nb_indexes++] = br;

	tridx[nb_indexes++] = tl;
	tridx[nb_indexes++] = tr;
	tridx[nb_indexes++] = br;

	tl++;
	bl++;
      }

      nb_vertices++;
      lon += loninc;
    }

    br = nb_vertices - SEGMENTS;
    tr = br - SEGMENTS;

    tridx[nb_indexes++] = bl;
    tridx[nb_indexes++] = tl;
    tridx[nb_indexes++] = br;

    tridx[nb_indexes++] = tl;
    tridx[nb_indexes++] = tr;
    tridx[nb_indexes++] = br;
  }

  lon = loninc/2;
  y -= 0.2;
  tl = nb_vertices-SEGMENTS;
  for (i=0; i<SEGMENTS; i++) {
 
    x = cos(lon)*cos(lat);
    z = -sin(lon)*cos(lat);

    addVertice(cx+x*radius, cy+y*radius, cz+z*radius, x, y, z);

    bl = nb_vertices;
    if(i == SEGMENTS-1)
      tr = tl -SEGMENTS+1;
    else
      tr = tl+1;
    tridx[nb_indexes++] = tl;
    tridx[nb_indexes++] = tr;
    tridx[nb_indexes++] = bl;
    tl++;

    nb_vertices++;
    lon += loninc;
  }

  // Add a fake vertex
  addVertice(cx+x*radius, cy+y*radius, cz+z*radius, x, y, z);
  nb_vertices++;

  vertices_per_atom = nb_vertices - vertices;
}

void addAtom(GLfloat cx, GLfloat cy, GLfloat cz, double radius)
{
#ifdef _SPHERE_MODE_
  static unsigned odd = 0;
  switch(atom_skin) {
    case PACMAN_SKIN :
      if(odd) {
	Ratom = 1.0;
	Gatom = 1.0;
	Batom = 1.0;
	addGhost(cx, cy, cz, radius);
      } else {
	Ratom = 1.0;
	Gatom = 1.0;
	Batom = 0.0;
	addSphere(cx, cy, cz, radius);
      }
      odd ^= 1;
      break;
    case SPHERE_SKIN :
    default :
      Ratom = 0.0;
      Gatom = 250.0/255;
      Batom = 250.0/255;
      addSphere(cx, cy, cz, radius);
  }
#else
  Ratom = 1.0;
  Gatom = 0.0;
  Batom = 0.0;
  addVertice(cx, cy, cz, 0.0f, 0.0f, 0.0f);
  nb_vertices++;
  vertices_per_atom = 1;
#endif
}
