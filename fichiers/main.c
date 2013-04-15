
#define _XOPEN_SOURCE 600

#include "atom.h"
#include "vbo.h"
#include "ocl.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <float.h>
#include <string.h>


GLuint glutWindowHandle = 0;
GLdouble fovy, aspect, near_clip, far_clip;  
                          /* parameters for gluPerspective() */

// mouse controls
GLfloat angle = 0.0;
float dis;

int winx=800, winy=800;   /* Window size */
float eye[3];             /* position of eye point */
float center[3];          /* position of look reference point */
float up[3];              /* up direction for camera */

unsigned full_speed = 0;

int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -1.f;

/* Function prototypes ************************************************/
void reshape(int, int);
void drawScene(void);
void display(void);
void initView(float *, float *);

/**********************************************************************/
void reshape (int w, int h) {
/***********************************************************************
  Callback for glutReshapeFunc()
***********************************************************************/

  /* set the GL viewport to match the full size of the window */
  glViewport(0, 0, (GLsizei)w, (GLsizei)h);
  aspect = w/(float)h;
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(fovy,aspect,near_clip,far_clip);
  glMatrixMode(GL_MODELVIEW);
}

static unsigned true_redisplay = 1;

/**********************************************************************/
void drawScene() {
/***********************************************************************
  Called by display() to draw the view of the current scene.
***********************************************************************/
  if(true_redisplay) {
    glLoadIdentity();
 
   /* Define viewing transformation */
    gluLookAt(
	      (GLdouble)eye[0],(GLdouble)eye[1],(GLdouble)eye[2],
	      (GLdouble)center[0],(GLdouble)center[1],(GLdouble)center[2],
	      (GLdouble)up[0],(GLdouble)up[1],(GLdouble)up[2]);

    glTranslatef(0.0, 0.0, translate_z);
    glTranslatef(center[0], center[1], center[2]);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);
    glTranslatef(-center[0], -center[1], -center[2]);

    true_redisplay = 0;
  }

  renderAtoms();

  // Draw a nice floor :)
  //
  glEnable (GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glBegin(GL_TRIANGLE_FAN);
  glColor4f(.6f, 0.6f, 1.0f, 0.4f);
  glVertex3f(center[0], min_ext[1], center[2]); glNormal3f(0.0, 1.0, 0.0);
  glVertex3f(min_ext[0], min_ext[1], min_ext[2]); glNormal3f(0.0, 1.0, 0.0);
  glVertex3f(min_ext[0], min_ext[1], max_ext[2]); glNormal3f(0.0, 1.0, 0.0);
  glVertex3f(max_ext[0], min_ext[1], max_ext[2]); glNormal3f(0.0, 1.0, 0.0);
  glVertex3f(max_ext[0], min_ext[1], min_ext[2]); glNormal3f(0.0, 1.0, 0.0);
  glVertex3f(min_ext[0], min_ext[1], min_ext[2]); glNormal3f(0.0, 1.0, 0.0);
  glEnd();
}

void display()
{
  glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
  drawScene();
  glutSwapBuffers();
}

void timer(int arg)
{
  animateGPU();
  glutPostRedisplay();

  glutTimerFunc(TIMER_VAL, timer, arg) ;
}

void idle(void)
{
  animateGPU();
  glutPostRedisplay();
}

void initView (float *min_ext, float *max_ext)
{
  GLfloat light_diffuse[]   = {1.0, 1.0, 1.0, 1.0};
  GLfloat light_position1[] = {0.5, 0.5, 1.0, 0.0};
  float dif_ext[3];
  int i;

  /* Define normal light */
  glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
  glLightfv(GL_LIGHT0, GL_POSITION, light_position1);

  /* Enable a single OpenGL light */
  glEnable(GL_LIGHTING);
  glEnable(GL_LIGHT0);

  /* Use depth buffering for hidden surface elimination */
  glEnable(GL_DEPTH_TEST);

  /* get diagonal and average distance of extent */
  for (i=0; i<3; i++)
    dif_ext[i] = max_ext[i] - min_ext[i];
  dis = 0.0;
  for (i=0; i<3; i++)
    dis += dif_ext[i]*dif_ext[i];
  dis = (float)sqrt((double)dis);

  /* set center in world space */
  for (i=0; i<3; i++)
    center[i] = min_ext[i] + dif_ext[i]/2.0;

  /* set initial eye & look at location in world space */
  eye[0] = center[0];
  eye[1] = center[1];
  eye[2] = center[2] + dis;
  up[0] = 0.0;
  up[1] = 1.0;
  up[2] = 0.0;

  /* set parameters for gluPerspective() */
  /* Near- & far clip-plane distances */
  near_clip = (GLdouble)( 0.5*(dis-0.5*dif_ext[2]) );
  far_clip  = (GLdouble)( 2.0*(dis+0.5*dif_ext[2]) );

  /* Field of view */
  fovy = (GLdouble)( 0.5*dif_ext[1]/(dis-0.5*dif_ext[2]) );
  fovy = (GLdouble)( 2*atan((double)fovy)/M_PI*180.0 );
  fovy = (GLdouble)(1.2*fovy);

  /* Enable the color material mode */
  glEnable(GL_COLOR_MATERIAL);
}

void appDestroy()
{
  atomFinalize();
  ocl_finalize();
  vboFinalize();

  if(glutWindowHandle)
    glutDestroyWindow(glutWindowHandle);

  exit(0);
}

void change_skin(atom_skin_t skin)
{
  // Resfresh positions of atoms in host memory
  ocl_readAtomCoordinatesFromAccel();

  // Reset all vertices & triangles indexes
  clearVBO();

  // Set skin to Pacmans and ghosts
  setAtomsSkin(skin);

  // Re-calculate vertices and triangles
  buildAtoms();

  // Resfreh vertices in Accelerator memory
  updateAtomCoordinatesToAccel();

  // Reset eating animation
  resetAnimation();

  glutPostRedisplay();
}

void appKeyboard(unsigned char key, int x, int y)
{
    //this way we can exit the program cleanly
    switch(key)
    {
    case '<' : translate_z -= 0.1; true_redisplay = 1; glutPostRedisplay(); break;
    case '>' : translate_z += 0.1; true_redisplay = 1; glutPostRedisplay(); break;
    case '+' : TIMER_VAL = TIMER_VAL - TIMER_VAL/10; break;
    case '-' : TIMER_VAL = TIMER_VAL + (TIMER_VAL/10 ? : 1); break;
    case 'e' :
    case 'E' : eating_enabled = 1 - eating_enabled;
               printf("eating mode: %d\n", eating_enabled); break;
    case 'n' :
    case 'N' : change_skin(SPHERE_SKIN); break; // atom mode
    case 'p' : 
    case 'P' : change_skin(PACMAN_SKIN); break; // pacman mode
    case 'z' :
    case 'Z' : zeroSpeeds(); break;
    case 'm' :
    case 'M' : move_enabled = 1 - move_enabled;
               printf("move: %d\n", move_enabled); break;
    case 'f' :
    case 'F' : force_enabled = 1 - force_enabled;
               printf("force: %d\n", force_enabled); break;
    case 'g' :
    case 'G' : gravity_enabled = 1 - gravity_enabled;
               printf("gravity: %d\n", gravity_enabled); break;
    case 'c' :
    case 'C' : detect_collision = 1 - detect_collision;
               printf("collision detect: %d\n", detect_collision); break;
    case '\033': // escape quits
    case '\015': // Enter quits    
    case 'Q':    // Q quits
    case 'q':    // q (or escape) quits
      // Cleanup up and quit
      appDestroy();
      break;
    }
}

void appMouse(int button, int state, int x, int y)
{
    //handle mouse interaction for rotating/zooming the view
    if (state == GLUT_DOWN) {
        mouse_buttons |= 1<<button;
    } else if (state == GLUT_UP) {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

void appMotion(int x, int y)
{

  //  printf("mouse motion\n");

    //hanlde the mouse motion for zooming and rotating the view
    float dx, dy;
    dx = x - mouse_old_x;
    dy = y - mouse_old_y;

    if (mouse_buttons & 1) {
        rotate_x += dy * 0.2;
        rotate_y += dx * 0.2;
    } else if (mouse_buttons & 4) {
        translate_z += dy * 0.1;
    }

    mouse_old_x = x;
    mouse_old_y = y;

    true_redisplay = 1;
    glutPostRedisplay();
}

int main(int argc, char **argv)
{
  glutInit(&argc, argv);

  // Filter args
  //
  argv++;
  while (argc > 1) {
    if(!strcmp(*argv, "--full-speed") || !strcmp(*argv, "-fs")) {
      full_speed = 1;
    } else
      break;
    argc--; argv++;
  }

  if(argc > 1)
    MD_FILE = *argv;

  /* Read atomic coordinates from an MD-configuration file */
  initializeAtoms();

  /* Set up an window */
  /* Initialize display mode */
  glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGBA|GLUT_DEPTH);
  /* Specify window size */
  glutInitWindowSize(winx, winy);
  /* Open window */
  glutWindowHandle = glutCreateWindow("Lennard-Jones Atoms");

  /* Initialize view */
  initView(min_ext, max_ext);

  /* Set a glut callback functions */
  glutDisplayFunc(display);
  glutReshapeFunc(reshape);
  if(full_speed)
    glutIdleFunc(idle);
  else
    glutTimerFunc(TIMER_VAL, timer, 0);

  glutKeyboardFunc(appKeyboard);
  glutMouseFunc(appMouse);
  glutMotionFunc(appMotion);

  /* make the geometry of the current frame's atoms */
  buildAtoms();
  buildVBO();

  printf("#atoms = %d, #vertices = %d, #triangles = %d, #vertices per atom = %d\n",
	 natoms, nb_vertices, nb_indexes/3, vertices_per_atom);

  /* Init acceleration devices */
  initializeComputeDevices();

  /* Start main display loop */
  glutMainLoop();

  return 0;
}
