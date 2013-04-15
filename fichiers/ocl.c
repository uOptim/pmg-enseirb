
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#ifdef __APPLE__
#include <OpenGL/CGLContext.h>
#include <OpenGL/CGLCurrent.h>
#else
#include </usr/include/GL/glx.h>
#endif
#include "ocl.h"
#include "vbo.h"
#include "atom.h"

#define MAX_DEVICES 5

cl_context context;                 // compute context
cl_program program;                 // compute program
cl_command_queue queue;

cl_mem vbo_buffer;
cl_mem pos_buffer;
cl_mem speed_buffer;
cl_mem min_buffer;
cl_mem max_buffer;

#define error(...) do { fprintf(stderr, "Error: " __VA_ARGS__); exit(EXIT_FAILURE); } while(0)
#define check(err, ...)					\
  do {							\
    if(err != CL_SUCCESS) {				\
      fprintf(stderr, "(%d) Error: " __VA_ARGS__, err);	\
      exit(EXIT_FAILURE);				\
    }							\
  } while(0)

static size_t file_size(const char *filename) {
	struct stat sb;
	if (stat(filename, &sb) < 0) {
		perror ("stat");
		abort ();
	}
	return sb.st_size;
}

static char *load(const char *filename) {
	FILE *f;
	char *b;
	size_t s;
	size_t r;
	s = file_size (filename);
	b = malloc (s+1);
	if (!b) {
		perror ("malloc");
		exit (1);
	}
	f = fopen (filename, "r");
	if (f == NULL) {
		perror ("fopen");
		exit (1);
	}
	r = fread (b, s, 1, f);
	if (r != 1) {
		perror ("fread");
		exit (1);
	}
	b[s] = '\0';
	return b;
}

void ocl_init(cl_device_type device_type, char *program_name)
{

  cl_platform_id pf[3];
  cl_uint nb_platforms = 0;
  cl_uint p = 0;

  cl_int err;                            // error code returned from api calls
    
  cl_device_id devices[MAX_DEVICES];
  cl_uint nb_devices = 0;

  cl_int dev = 0;

  // Get list of OpenCL platforms detected
  //
  err = clGetPlatformIDs(3, pf, &nb_platforms);
  check(err, "Failed to get platform IDs");

  //printf("%d OpenCL platforms detected\n", nb_platforms);

  // Print name & vendor for each platform
  //
  for (unsigned int _p=0; _p<nb_platforms; _p++) {
    cl_uint num;
    int platform_valid = 1;
    char name[1024], vendor[1024];

    err = clGetPlatformInfo(pf[_p], CL_PLATFORM_NAME, 1024, name, NULL);
    check(err, "Failed to get Platform Info");

    err = clGetPlatformInfo(pf[_p], CL_PLATFORM_VENDOR, 1024, vendor, NULL);
    check(err, "Failed to get Platform Info");

    printf("Platform %d: %s - %s\n", _p, name, vendor);

    if(strstr(vendor, "NVIDIA")) {
      p = _p;
      printf("Choosing platform %d\n", p);
    }
  }

  // Get list of devices
  //
  err = clGetDeviceIDs(pf[p], device_type, MAX_DEVICES, devices, &nb_devices);
  //printf("nb devices = %d\n", nb_devices);

  for(int d = 0; d < nb_devices; d++) {
    cl_device_type dtype;
    char name[1024];

    err = clGetDeviceInfo(devices[d], CL_DEVICE_NAME, 1024, name, NULL);
    check(err, "Cannot get name of device");
    err = clGetDeviceInfo(devices[d], CL_DEVICE_TYPE, sizeof(cl_device_type), &dtype, NULL);
    check(err, "Cannot get type of device");

    printf("Device %d : %s [%s]\n", d, (dtype == CL_DEVICE_TYPE_GPU) ? "GPU" : "CPU", name);
    if(dtype == CL_DEVICE_TYPE_GPU)
      dev = d;
  }

  printf("Choosing device %d\n", dev);


  // Setup context properties so that the Vertex Buffer Object can be shared between OpenGL and OpenCL
  //
#ifdef __APPLE__
  CGLContextObj cgl_context = CGLGetCurrentContext();
  CGLShareGroupObj sharegroup = CGLGetShareGroup(cgl_context);
  cl_context_properties properties[] = {
    CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
    (cl_context_properties)sharegroup,
    0};
#else
  cl_context_properties properties[] = {
    CL_GL_CONTEXT_KHR, (cl_context_properties) glXGetCurrentContext(),
    CL_GLX_DISPLAY_KHR, (cl_context_properties) glXGetCurrentDisplay(), 
    CL_CONTEXT_PLATFORM, (cl_context_properties) pf[p], 
    0};
#endif

  // Create compute context with "device_type" devices
  //
  context = clCreateContext (properties, 1, devices + dev, NULL, NULL, &err);
  check(err, "Failed to create compute context");

  // Load program source
  //
  const char	*opencl_prog;
  opencl_prog = load(program_name);

  // Build program
  //
  program = clCreateProgramWithSource(context, 1, &opencl_prog, NULL, &err);
  check(err, "Failed to create program");

  err = clBuildProgram (program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    size_t len;

    // Display compiler error log
    //
    clGetProgramBuildInfo(program, devices[dev], CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
    {
      char buffer[len+1];

      clGetProgramBuildInfo(program, devices[dev], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, NULL);
      fprintf(stderr, "%s\n", buffer);
    }
    error("Failed to build program!\n");
  }

  // Share vertex buffer with OpenGL
  //
  vbo_buffer = clCreateFromGLBuffer(context, CL_MEM_READ_WRITE, vbovid, NULL);
  if(!vbo_buffer)
    error("Failed to map vbo buffer!\n");

  pos_buffer = clCreateBuffer(context,  CL_MEM_READ_WRITE, atomPosSize(), NULL, NULL);
  if(!pos_buffer)
    error("Failed to map coord buffer!\n");

  speed_buffer = clCreateBuffer(context,  CL_MEM_READ_WRITE, atomSpeedSize(), NULL, NULL);
  if(!speed_buffer)
    error("Failed to map coord buffer!\n");

  min_buffer = clCreateBuffer(context,  CL_MEM_READ_ONLY, 3 * sizeof(float), NULL, NULL);
  if(!min_buffer)
    error("Failed to map coord buffer!\n");

  max_buffer = clCreateBuffer(context,  CL_MEM_READ_ONLY, 3 * sizeof(float), NULL, NULL);
  if(!max_buffer)
    error("Failed to map coord buffer!\n");


  // Create an OpenCL command queue
  //
  queue = clCreateCommandQueue(context, devices[dev], CL_QUEUE_PROFILING_ENABLE, &err);
  check(err,"Failed to create a command queue!\n");

  // Transfer VBO buffer
  //
  {
    glFinish();
    err = clEnqueueAcquireGLObjects(queue, 1, &vbo_buffer, 0, NULL, NULL);
    check(err, "Failed to acquire lock!\n");

    err = clEnqueueWriteBuffer(queue, vbo_buffer, CL_TRUE, 0,
			       sizeof(float) * 3 * nb_vertices, vbo_vertex, 0, NULL, NULL);
    check(err, "Failed to write to vbo_buffer array!\n");

    clFinish(queue);
    err = clEnqueueReleaseGLObjects(queue, 1, &vbo_buffer, 0, NULL, NULL);
    check(err, "Failed to release lock!\n");
  }

  // Transfer coord buffer
  //
  err = clEnqueueWriteBuffer(queue, pos_buffer, CL_TRUE, 0,
			     atomPosSize(), atomPosAddr(), 0, NULL, NULL);
  check(err, "Failed to write to pos_buffer array!\n");

  // Transfert speed buffer
  //
  err = clEnqueueWriteBuffer(queue, speed_buffer, CL_TRUE, 0,
			     atomSpeedSize(), atomSpeedAddr(), 0, NULL, NULL);
  check(err, "Failed to write to speed_buffer array!\n");

  err = clEnqueueWriteBuffer(queue, min_buffer, CL_TRUE, 0,
			     sizeof(float) * 3, min_ext, 0, NULL, NULL);
  check(err, "Failed to write to min_buffer array!\n");

  err = clEnqueueWriteBuffer(queue, max_buffer, CL_TRUE, 0,
			     sizeof(float) * 3, max_ext, 0, NULL, NULL);
  check(err, "Failed to write to max_buffer array!\n");

  clFinish(queue);
}

void ocl_kernelCreate(char *entry_point, cl_kernel *kernel)
{
  cl_int err;

  // Create the compute kernel in the program we wish to run
  //
  err = 0;
  *kernel = clCreateKernel(program, entry_point, &err);
  check(err, "Failed to create compute kernel!\n");
}

void ocl_readAtomCoordinatesFromAccel(void)
{
  cl_int err;

  // Transfer position buffer back
  //
  err = clEnqueueReadBuffer(queue, pos_buffer, CL_TRUE, 0,
			    atomPosSize(), atomPosAddr(), 0, NULL, NULL);
  check(err, "Failed to read from atom positions array!\n");

  clFinish(queue);
}

void ocl_updateVBOFromHost(void)
{
  cl_int err;

  // Transfer VBO buffer
  //
  glFinish();
  err = clEnqueueAcquireGLObjects(queue, 1, &vbo_buffer, 0, NULL, NULL);
  check(err, "Failed to acquire lock!\n");

  err = clEnqueueWriteBuffer(queue, vbo_buffer, CL_TRUE, 0,
			     sizeof(float) * 3 * nb_vertices, vbo_vertex, 0, NULL, NULL);
  check(err, "Failed to write to vbo_buffer array!\n");

  clFinish(queue);
  err = clEnqueueReleaseGLObjects(queue, 1, &vbo_buffer, 0, NULL, NULL);
  check(err, "Failed to release lock!\n");
}

void ocl_finalize(void)
{
  clReleaseMemObject(vbo_buffer);
  clReleaseMemObject(min_buffer);
  clReleaseMemObject(max_buffer);
  clReleaseMemObject(pos_buffer);
  clReleaseMemObject(speed_buffer);

  clReleaseCommandQueue(queue);
  clReleaseProgram(program);
  clReleaseContext(context);
}
