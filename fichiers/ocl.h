
#ifndef _OCL_IS_DEF
#define _OCL_IS_DEF

#ifdef __APPLE__
#  include <OpenCL/opencl.h>
#else
#  include <CL/opencl.h>
#endif

void ocl_init(cl_device_type device_type, char *program_name); 
/* device_type may be CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU or CL_DEVICE_TYPE_ALL */

void ocl_kernelCreate(char *entry_point, cl_kernel *kernel);
void ocl_readAtomCoordinatesFromAccel(void);
void ocl_updateVBOFromHost(void);

void ocl_finalize(void);

#endif
