
#ifndef _ATOM_IS_DEF_
#define _ATOM_IS_DEF_

#include <stdlib.h>

#define ATOM_RADIUS 0.2f


extern int natoms;               /* number of atoms */

extern char *MD_FILE;

extern float min_ext[3], max_ext[3];  /* Range of atomic coordinates:
                                         (left,lower,back), (right,top,front) */

extern unsigned long TIMER_VAL;   /* atom position update interval (in milliseconds) */

extern unsigned eating_enabled;
extern unsigned move_enabled;
extern unsigned detect_collision;
extern unsigned force_enabled;
extern unsigned gravity_enabled;

void initializeAtoms(void);
float *atomPosAddr(void);
float *atomSpeedAddr(void);
size_t atomPosSize(void);
size_t atomSpeedSize(void);
void buildAtoms(void);
void initializeComputeDevices(void);
void animateGPU(void);
void resetAnimation(void);
void zeroSpeeds(void);
void atomFinalize(void);

#endif
