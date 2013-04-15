
#define _XOPEN_SOURCE 600

#include "atom.h"
#include "vbo.h"
#include "ocl.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <float.h>

char *MD_FILE = "default.conf";

#define ALIGN     16
#define ROUND(n)  (((size_t)(n)+ALIGN-1)&(~(size_t)(ALIGN-1)))

#define x_atom(a) (atom_pos.x[a])
#define y_atom(a) (atom_pos.y[a])
#define z_atom(a) (atom_pos.z[a])

#define x_speed(a) (atom_speed.dx[a])
#define y_speed(a) (atom_speed.dy[a])
#define z_speed(a) (atom_speed.dz[a])

// atom positions (looks like 3 arrays, but allocated once contiguously)
typedef struct {
  float *x;
  float *y;
  float *z;
} AtomPosType;

// atom speeds (looks like 3 arrays, but allocated once contiguously)
typedef struct {
  float *dx;
  float *dy;
  float *dz;
} AtomSpeedType; 


int natoms = 0;
AtomPosType atom_pos;
AtomSpeedType atom_speed;

float min_ext[3], max_ext[3];  /* Range of atomic coordinates:
                                  (left,lower,back), (right,top,front) */

unsigned long TIMER_VAL = 50;   /* atom position update interval (in milliseconds) */
unsigned eating_enabled = 0;
unsigned move_enabled = 0;
unsigned detect_collision = 0;
unsigned force_enabled = 0;
unsigned gravity_enabled = 0;

extern cl_command_queue queue;
extern cl_mem vbo_buffer;
extern cl_mem pos_buffer;
extern cl_mem speed_buffer;
extern cl_mem min_buffer;
extern cl_mem max_buffer;

static cl_kernel eating_kernel;
static cl_kernel move_vertices_kernel;
static cl_kernel update_position_kernel;
static cl_kernel border_col_kernel;
static cl_kernel atom_col_kernel;
static cl_kernel atom_force_kernel;
static cl_kernel gravity_kernel;
static cl_int err;


#define error(...) do { fprintf(stderr, "Error: " __VA_ARGS__); exit(EXIT_FAILURE); } while(0)
#define check(err, ...)					\
  do {							\
    if(err != CL_SUCCESS) {				\
      fprintf(stderr, "(%d) Error: " __VA_ARGS__, err);	\
      exit(EXIT_FAILURE);				\
    }							\
  } while(0)

float *atomPosAddr(void)
{
  return atom_pos.x;
}

float *atomSpeedAddr(void)
{
  return atom_speed.dx;
}

size_t atomPosSize(void)
{
  return ROUND(natoms)*sizeof(float)*3;
}

size_t atomSpeedSize(void)
{
  return ROUND(natoms)*sizeof(float)*3;
}


static float rand_float(float mn, float mx)
{
    float r = random() / (float) RAND_MAX;
    return mn + (mx-mn)*r;
}

void initializeAtoms(void)
{
  int l, j;
  FILE *fp;
  float lon,lat;
  float speed;
  int read_speed;

  fp = fopen(MD_FILE, "r");
  if(fp == NULL)
    error("Cannot open %s file\n", MD_FILE);

  // Read the # of atoms
  fscanf(fp, "%d", &natoms);

  // allocate atom positions
  atom_pos.x = (float *)malloc(atomPosSize());
  atom_pos.y = atom_pos.x + ROUND(natoms);
  atom_pos.z = atom_pos.y + ROUND(natoms);

  // allocate atom speeds
  atom_speed.dx = (float *)malloc(atomSpeedSize());
  atom_speed.dy = atom_speed.dx + ROUND(natoms);
  atom_speed.dz = atom_speed.dy + ROUND(natoms);

  // Maximum & minimum extent of system in Angstroms
  for (l=0; l<3; l++) {
    fscanf(fp, "%f %f", &min_ext[l], &max_ext[l]);
    min_ext[l] -= ATOM_RADIUS;
    max_ext[l] += ATOM_RADIUS;
  }

  // speeds defined within the file?
  fscanf(fp, "%d", &read_speed);

  // Atom coordinates
  for (j=0; j<natoms; j++) {
    fscanf(fp, "%f %f %f",
           &x_atom(j),
	   &y_atom(j),
	   &z_atom(j));

    if(read_speed) {
      fscanf(fp, "%f %f %f",
	     &x_speed(j),
	     &y_speed(j),
	     &z_speed(j));
    } else { 
      // Velocity norm is between 5% and 20% of atom radius, in a random direction
      speed = rand_float(ATOM_RADIUS*0.05, ATOM_RADIUS*0.2);
      lat = rand_float(-M_PI/2, M_PI/2);
      lon = rand_float(0.0, M_PI*2);
      x_speed(j) = cos(lon)*cos(lat)*speed;
      y_speed(j) = sin(lat)*speed;
      z_speed(j) = -sin(lon)*cos(lat)*speed;
    }
  }
  fclose(fp);
}

void buildAtoms(void)
{
  int i;

  for (i=0; i < natoms; i++)
    addAtom(x_atom(i), y_atom(i), z_atom(i), ATOM_RADIUS);
}

void initializeComputeDevices(void)
{
  // Init OpenCL
  ocl_init(CL_DEVICE_TYPE_ALL, "physics.cl");

  ocl_kernelCreate("eating", &eating_kernel);
  ocl_kernelCreate("move_vertices", &move_vertices_kernel);
  ocl_kernelCreate("update_position", &update_position_kernel);
  ocl_kernelCreate("border_collision", &border_col_kernel);
  ocl_kernelCreate("atom_collision", &atom_col_kernel);
  ocl_kernelCreate("lennard_jones", &atom_force_kernel);
  ocl_kernelCreate("gravity", &gravity_kernel);
}

static void border_collision(void)
{
  cl_event prof_event;
  size_t global;                      // global domain size for our calculation
  size_t local;                       // local domain size for our calculation
  float radius = ATOM_RADIUS;         // collision when closer than atom radius

  // Set the arguments to our compute kernel
  //
  err  = clSetKernelArg(border_col_kernel, 0, sizeof(cl_mem), &pos_buffer);
  err  |= clSetKernelArg(border_col_kernel, 1, sizeof(cl_mem), &speed_buffer);
  err  |= clSetKernelArg(border_col_kernel, 2, sizeof(cl_mem), &min_buffer);
  err  |= clSetKernelArg(border_col_kernel, 3, sizeof(cl_mem), &max_buffer);
  err  |= clSetKernelArg(border_col_kernel, 4, sizeof(float), &radius);
  err  |= clSetKernelArg(border_col_kernel, 5, sizeof(natoms), &natoms);
  check(err, "Failed to set kernel arguments! %d\n", err);

  global = 1; // TODO: CHANGE!!!
  local = 1; // Set workgroup size to 1

  err = clEnqueueNDRangeKernel(queue, border_col_kernel, 1, NULL, &global, &local, 0, NULL, &prof_event);
  check(err, "Failed to execute kernel!\n");
}

static void update_position(void)
{
  cl_event prof_event;
  size_t global;                      // global domain size for our calculation
  size_t local;                       // local domain size for our calculation

  // Set the arguments to our compute kernel
  //
  err  = clSetKernelArg(update_position_kernel, 0, sizeof(cl_mem), &pos_buffer);
  err  |= clSetKernelArg(update_position_kernel, 1, sizeof(cl_mem), &speed_buffer);
  err  |= clSetKernelArg(update_position_kernel, 2, sizeof(natoms), &natoms);
  check(err, "Failed to set kernel arguments! %d\n", err);

  global = 1; // TODO: CHANGE!!!
  local = 1; // Set workgroup size to 1

  err = clEnqueueNDRangeKernel(queue, update_position_kernel, 1, NULL, &global, &local, 0, NULL, &prof_event);
  check(err, "Failed to execute kernel!\n");
}

static void atom_collision(void)
{
  cl_event prof_event;
  size_t global;                      // global domain size for our calculation
  size_t local;                       // local domain size for our calculation
  float radius = ATOM_RADIUS;         // collision when closer to atom radius

  // Set the arguments to our compute kernel
  //
  err  = clSetKernelArg(atom_col_kernel, 0, sizeof(cl_mem), &pos_buffer);
  err  |= clSetKernelArg(atom_col_kernel, 1, sizeof(cl_mem), &speed_buffer);
  err  |= clSetKernelArg(atom_col_kernel, 2, sizeof(float), &radius);
  check(err, "Failed to set kernel arguments! %d\n", err);

  global = 1; // TODO: CHANGE!!!
  local = 1; // Set workgroup size to 1

  err = clEnqueueNDRangeKernel(queue, atom_col_kernel, 1, NULL, &global, &local, 0, NULL, &prof_event);
  check(err, "Failed to execute kernel!\n");
}

static void atom_force(void)
{
  cl_event prof_event;
  size_t global;                      // global domain size for our calculation
  size_t local;                       // local domain size for our calculation
  float radius = ATOM_RADIUS;

  // Set the arguments to our compute kernel
  //
  err  = clSetKernelArg(atom_force_kernel, 0, sizeof(cl_mem), &pos_buffer);
  err  |= clSetKernelArg(atom_force_kernel, 1, sizeof(cl_mem), &speed_buffer);
  err  |= clSetKernelArg(atom_force_kernel, 2, sizeof(float), &radius);
  check(err, "Failed to set kernel arguments! %d\n", err);

  global = 1; // TODO: CHANGE!!!
  local = 1; // Set workgroup size to 1

  err = clEnqueueNDRangeKernel(queue, atom_force_kernel, 1, NULL, &global, &local, 0, NULL, &prof_event);
  check(err, "Failed to execute kernel!\n");
}

static void gravity(void)
{
  cl_event prof_event;
  size_t global;                      // global domain size for our calculation
  size_t local;                       // local domain size for our calculation
  float g = 0.005;                   // gravity acceleration

  // Set the arguments to our compute kernel
  //
  err  = clSetKernelArg(gravity_kernel, 0, sizeof(cl_mem), &pos_buffer);
  err  |= clSetKernelArg(gravity_kernel, 1, sizeof(cl_mem), &speed_buffer);
  err  |= clSetKernelArg(gravity_kernel, 2, sizeof(float), &g);
  check(err, "Failed to set kernel arguments! %d\n", err);

  global = 1; // TODO: CHANGE!!!
  local = 1; // Set workgroup size to 1

  err = clEnqueueNDRangeKernel(queue, gravity_kernel, 1, NULL, &global, &local, 0, NULL, &prof_event);
  check(err, "Failed to execute kernel!\n");
}

static void move_vertices(void)
{
  cl_event prof_event;
  size_t global;                      // global domain size for our calculation
  size_t local;                       // local domain size for our calculation

  // Set the arguments to our compute kernel
  //
  err  = clSetKernelArg(move_vertices_kernel, 0, sizeof(cl_mem), &vbo_buffer);
  err  |= clSetKernelArg(move_vertices_kernel, 1, sizeof(cl_mem), &speed_buffer);
  err  |= clSetKernelArg(move_vertices_kernel, 2, sizeof(unsigned), &vertices_per_atom);
  check(err, "Failed to set kernel arguments! %d\n", err);

  global = 1; // TODO: CHANGE!!!
  local = 1; // Set workgroup size to 1

  err = clEnqueueNDRangeKernel(queue, move_vertices_kernel, 1, NULL, &global, &local, 0, NULL, &prof_event);
  check(err, "Failed to execute kernel!\n");
}

#define PERIOD    9
static float dy = -0.01;
static unsigned long step = 0;

static void eating(void)
{
  cl_event prof_event;
  size_t global;                      // global domain size for our calculation
  size_t local;                       // local domain size for our calculation

  if(step++ % PERIOD == 0)
    dy *= -1;

  // Set the arguments to our compute kernel
  //
  err  = clSetKernelArg(eating_kernel, 0, sizeof(cl_mem), &vbo_buffer);
  err  |= clSetKernelArg(eating_kernel, 1, sizeof(float), &dy);
  err  |= clSetKernelArg(eating_kernel, 2, sizeof(unsigned), &vertices_per_atom);
  check(err, "Failed to set kernel arguments! %d\n", err);

  global = nb_vertices*3; // One thread per vertex coordinate -> 3 * #vertices
  local = 1; // Set workgroup size to 1

  err = clEnqueueNDRangeKernel(queue, eating_kernel, 1, NULL, &global, &local, 0, NULL, &prof_event);
  check(err, "Failed to execute kernel!\n");
}

void resetAnimation(void)
{
  dy = -0.01;
  step = 0;
}

void animateGPU(void)
{
  glFinish();
  err = clEnqueueAcquireGLObjects(queue, 1, &vbo_buffer, 0, NULL, NULL);
  check(err, "Failed to acquire lock!\n");

  if(eating_enabled)
    eating();

  if(move_enabled) {

    if(force_enabled)
      atom_force();

    if(gravity_enabled)
      gravity();

    if(detect_collision)
      //  check collisions between atoms
      atom_collision();

    // check border collision, invert speed component if needed
    border_collision();

    // update position of atoms
    update_position();

    // move all the vertices
    move_vertices();
  }

  // Wait for the command commands to get serviced before reading back results
  //
  clFinish(queue);
  err = clEnqueueReleaseGLObjects(queue, 1, &vbo_buffer, 0, NULL, NULL);
  check(err, "Failed to release lock!\n");
}

void zeroSpeeds(void)
{
  for (int j=0; j<natoms; j++) {
    x_speed(j) = 0.0f;
    y_speed(j) = 0.0f;
    z_speed(j) = 0.0f;
  }

  err = clEnqueueWriteBuffer(queue, speed_buffer, CL_TRUE, 0,
			     atomSpeedSize(), atomSpeedAddr(), 0, NULL, NULL);
  check(err, "Failed to write to speed_buffer array!\n");

  clFinish(queue);
}

void atomFinalize(void)
{
  clReleaseKernel(eating_kernel);
  clReleaseKernel(move_vertices_kernel);
  clReleaseKernel(border_col_kernel);
  clReleaseKernel(atom_col_kernel);
  clReleaseKernel(atom_force_kernel);
  clReleaseKernel(gravity_kernel);
}
