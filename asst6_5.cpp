////////////////////////////////////////////////////////////////////////
//
//   Harvard University
//   CS175 : Computer Graphics
//   Professor Steven Gortler
//
////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <cstddef>
#include <vector>
#include <math.h>
#include <string>
#include <memory>
#include <stdexcept>
#include <list>
#if __GNUG__
#   include <tr1/memory>
#endif

#ifdef __MAC__
#   include <OpenGL/gl3.h>
#   include <GLUT/glut.h>
#else
#   include <GL/glew.h>
#   include <GL/glut.h>
#endif

#include "ppm.h"
#include "cvec.h"
#include "matrix4.h"
#include "rigtform.h"
#include "glsupport.h"
#include "geometrymaker.h"
#include "arcball.h"
#include "scenegraph.h"

#include "asstcommon.h"
#include "drawer.h"
#include "picker.h"
#include "sgutils.h"
#include "geometry.h"

using namespace std;
using namespace tr1;

#define KF_UNDEF -1

// G L O B A L S ///////////////////////////////////////////////////

// --------- IMPORTANT --------------------------------------------------------
// Before you start working on this assignment, set the following variable
// properly to indicate whether you want to use OpenGL 2.x with GLSL 1.0 or
// OpenGL 3.x+ with GLSL 1.5.
//
// Set g_Gl2Compatible = true to use GLSL 1.0 and g_Gl2Compatible = false to
// use GLSL 1.5. Use GLSL 1.5 unless your system does not support it.
//
// If g_Gl2Compatible=true, shaders with -gl2 suffix will be loaded.
// If g_Gl2Compatible=false, shaders with -gl3 suffix will be loaded.
// To complete the assignment you only need to edit the shader files that get
// loaded
// ----------------------------------------------------------------------------
#ifdef __MAC__
  const bool g_Gl2Compatible = false;
#else
  const bool g_Gl2Compatible = true;
#endif


static const float g_frustMinFov = 60.0;  // A minimal of 60 degree field of view
static float g_frustFovY = g_frustMinFov; // FOV in y direction (updated by updateFrustFovY)

static const float g_frustNear = -0.1;    // near plane
static const float g_frustFar = -50.0;    // far plane
static const float g_groundY = -2.0;      // y coordinate of the ground
static const float g_groundSize = 10.0;   // half the ground length

enum SkyMode {WORLD_SKY=0, SKY_SKY=1};

static int g_windowWidth = 512;
static int g_windowHeight = 512;
static bool g_mouseClickDown = false;    // is the mouse button pressed
static bool g_mouseLClickButton, g_mouseRClickButton, g_mouseMClickButton;
static bool g_spaceDown = false;         // space state, for middle mouse emulation
static int g_mouseClickX, g_mouseClickY; // coordinates for mouse click event
static int g_activeShader = 0;

static SkyMode g_activeCameraFrame = WORLD_SKY;

static bool g_displayArcball = true;
static double g_arcballScreenRadius = 100; // number of pixels
static double g_arcballScale = 1;

static bool g_pickingMode = false;

// -------- Shaders

static shared_ptr<Material> g_redDiffuseMat,
                            g_blueDiffuseMat,
                            g_bumpFloorMat,
                            g_arcballMat,
                            g_pickingMat,
                            g_lightMat;

shared_ptr<Material> g_overridingMaterial;

// linked list of frame vectors
static list<vector<RigTForm> > key_frames;
static int cur_frame = -1;

// --------- Geometry
typedef SgGeometryShapeNode MyShapeNode;

// Vertex buffer and index buffer associated with the ground and cube geometry
static shared_ptr<Geometry> g_ground, g_cube, g_sphere;

// --------- Scene

static const Cvec3 g_light1(2.0, 3.0, 14.0), g_light2(-2, 3.0, -14.0);  // define two lights positions in world space

static shared_ptr<SgTransformNode> g_light1Node, g_light2Node;

static shared_ptr<SgRootNode> g_world;
static shared_ptr<SgRbtNode> g_skyNode, g_groundNode, g_robot1Node, g_robot2Node;

static shared_ptr<SgRbtNode> g_currentCameraNode;
static shared_ptr<SgRbtNode> g_currentPickedRbtNode;

static int g_msBetweenKeyFrames = 2000;
static int g_animateFramesPerSecond = 60;
static bool animating = false;

///////////////// END OF G L O B A L S //////////////////////////////////////////////////

static void make_frame() {
  vector<shared_ptr<SgRbtNode> > graph_vector;
  dumpSgRbtNodes(g_world, graph_vector);

  vector<RigTForm> new_frame;
  for (int i = 0; i < graph_vector.size(); ++i) {
    new_frame.push_back(graph_vector[i]->getRbt());
  }

  if (cur_frame == KF_UNDEF || cur_frame == key_frames.size() - 1) {
    // undef is -1, so adding one sets the position to 0
    key_frames.push_back(new_frame);
    ++cur_frame;
  }
  else {
    list<vector<RigTForm> >::iterator it = key_frames.begin();
    advance(it, cur_frame);
    key_frames.insert(it, new_frame);
    ++cur_frame;
  }
  return;
}

static void next_frame() {
  if (cur_frame == KF_UNDEF || cur_frame == key_frames.size() - 1) {
    cout << "can't advance frame" << endl;
    return;
  }
  ++cur_frame;
  list<vector<RigTForm> >::iterator it = key_frames.begin();
  advance(it, cur_frame);
  // this linked list of arrays is getting the previous vectors stacked on top of each other
  fillSgRbtNodes(g_world, *it);
  return;
}

static void prev_frame() {
  if (cur_frame < 1) {
    cout << "can't rewind frame "<< endl;
    return;
  }
  --cur_frame;
  list<vector<RigTForm> >::iterator it = key_frames.begin();
  advance(it, cur_frame);
  fillSgRbtNodes(g_world, *it);
  return;
}

static void delete_frame() {
  if (cur_frame == KF_UNDEF) {
    return;
  }
  list<vector<RigTForm> >::iterator it = key_frames.begin();
  advance(it, cur_frame);
  key_frames.erase(it);
  if (key_frames.empty()) {
    cur_frame = KF_UNDEF;
    return;
  }
  else if (cur_frame != 0) {
    --cur_frame;
  }
  fillSgRbtNodes(g_world, *it);

  return;
}

static void write_frame() {
  list<vector<RigTForm> >::iterator it = key_frames.begin();
  FILE* output = fopen("animation.txt", "w");
  int n = (*it).size();
  fprintf(output, "%d %d\n", key_frames.size(), n);
  while (it != key_frames.end()) {
    vector<RigTForm> frame = *it;
    for (int i = 0; i < frame.size(); ++i) {
      RigTForm r = frame[i];
      Cvec3 transFact = r.getTranslation();
      Quat linFact = r.getRotation();
      fprintf(output, "%.3f %.3f %.3f %.3f %.3f %.3f %.3f\n",
          transFact[0], transFact[1], transFact[2],
          linFact[0], linFact[1], linFact[2], linFact[3]
      );
    }
    ++it;
  }
  fclose(output);
}

static void read_frame() {
  FILE* input = fopen("animation.txt", "r");
  if (input == NULL) {
    return;
  }

  int nFrames;
  int nRbts;
  fscanf(input, "%d %d\n", &nFrames, &nRbts);
  key_frames.clear();

  for (int i = 0; i < nFrames; ++i) {
    vector<RigTForm> frame;
    for (int j = 0; j < nRbts; ++j) {
      Cvec3 transFact;
      Quat linFact;
      fscanf(input, "%lf %lf %lf %lf %lf %lf %lf\n",
          &transFact[0], &transFact[1], &transFact[2],
          &linFact[0], &linFact[1], &linFact[2], &linFact[3]
      );
      RigTForm r = RigTForm(transFact, linFact);
      frame.push_back(r);
    }
    key_frames.push_back(frame);
  }
  cur_frame = 0;
  fillSgRbtNodes(g_world, key_frames.front());
  fclose(input);

}

static Quat slerp(Quat src, Quat dest, float alpha);
static Cvec3 lerp(Cvec3 src, Cvec3 dest, float alpha);
static Quat cond_neg(Quat q);
static Quat qpow(Quat q, float alpha);


Cvec3 getDTrans(Cvec3 c_i_1, Cvec3 c_i_neg_1, Cvec3 c_i) {
  return (c_i_1 - c_i_neg_1)/6 + c_i;
}

Cvec3 getETrans(Cvec3 c_i_2, Cvec3 c_i_1, Cvec3 c_i) {
  return (c_i_2 - c_i)/-6 + c_i_1;
}

Cvec3 bezierTrans(Cvec3 c_i_neg_1, Cvec3 c_i, Cvec3 c_i_1, Cvec3 c_i_2, int i, float t) {

  Cvec3 d = getDTrans(c_i_1, c_i_neg_1, c_i);
  Cvec3 e = getETrans(c_i_2, c_i_1, c_i);

  Cvec3 f = c_i*(1 - t + i) + d*(t - i);
  Cvec3 g = d*(1 - t + i) + e*(t - i);
  Cvec3 h = e*(1 - t + i) + c_i_1*(t - i);
  Cvec3 m = f*(1 - t + i) + g*(t - i);
  Cvec3 n = g*(1 - t + i) + h*(t - i);

  return m*(1 - t + i) + n*(t - i);
}

Quat getDRot(Quat c_i_1, Quat c_i_neg_1, Quat c_i) {
  return qpow(cond_neg(c_i_1 * inv(c_i_neg_1)), 1.0/6.0) * c_i;
}

Quat getERot(Quat c_i_2, Quat c_i_1, Quat c_i) {
  return qpow(cond_neg(c_i_2 * inv(c_i)), -1.0/6.0) * c_i_1;
}

Quat bezierRot(Quat c_i_neg_1, Quat c_i, Quat c_i_1, Quat c_i_2, int i, float t) {

  Quat d = getDRot(c_i_1, c_i_neg_1, c_i);
  Quat e = getERot(c_i_2, c_i_1, c_i);

  Quat f = slerp(c_i, d, t -i);
  Quat g = slerp(d, e, t - i);
  Quat h = slerp(e, c_i_1, t - i);
  Quat m = slerp(f, g, t - i);
  Quat n = slerp(g, h, t - i);

  return slerp(m, n, t - i);
}

bool interpolateAndDisplay(float t) {
  list<vector<RigTForm> >::iterator it = key_frames.begin();
  advance(it, (int) t);

  ++it;
  vector<RigTForm> frame_1 = *it;
  ++it;
  vector<RigTForm> frame_2 = *it;
  ++it;
  if (it == key_frames.end()) {
    return true;
  }
  vector<RigTForm> post_frame = *it;
  // minus operator not overloaded for iterators. sad face.
  --it; --it; --it;
  vector<RigTForm> pre_frame = *it;


  // d ci ci+1 e
  float alpha = t - (int) t;
  vector<RigTForm> frame;
  int n = frame_1.size();
  for (int i = 0; i < n; ++i) {
    Cvec3 c_i_neg_1 = pre_frame[i].getTranslation();
    Cvec3 c_i = frame_1[i].getTranslation();
    Cvec3 c_i_1 = frame_2[i].getTranslation();
    Cvec3 c_i_2 = post_frame[i].getTranslation();

    Quat c_i_neg_1_r = pre_frame[i].getRotation();
    Quat c_i_r = frame_1[i].getRotation();
    Quat c_i_1_r = frame_2[i].getRotation();
    Quat c_i_2_r = post_frame[i].getRotation();

    Cvec3 trans = bezierTrans(c_i_neg_1, c_i, c_i_1, c_i_2, (int) t, t);
    Quat rot = bezierRot(c_i_neg_1_r, c_i_r, c_i_1_r, c_i_2_r, (int) t, t);
    frame.push_back(RigTForm(trans, rot));
  }
  fillSgRbtNodes(g_world, frame);
  glutPostRedisplay();

  return false;
}

static void animateTimerCallback(int ms) {
  float t = (float) ms / (float) g_msBetweenKeyFrames;

  bool endReached = interpolateAndDisplay(t);
  if (!endReached) {
    glutTimerFunc(1000/g_animateFramesPerSecond,
        animateTimerCallback,
        ms + 1000/g_animateFramesPerSecond);
  }
  else {
    animating = false;
    cur_frame = key_frames.size() - 2;
    glutPostRedisplay();
  }
}

static void initGround() {
  int ibLen, vbLen;
  getPlaneVbIbLen(vbLen, ibLen);

  // Temporary storage for cube Geometry
  vector<VertexPNTBX> vtx(vbLen);
  vector<unsigned short> idx(ibLen);

  makePlane(g_groundSize*2, vtx.begin(), idx.begin());
  g_ground.reset(new SimpleIndexedGeometryPNTBX(&vtx[0], &idx[0], vbLen, ibLen));
}

static void initCubes() {
  int ibLen, vbLen;
  getCubeVbIbLen(vbLen, ibLen);

  // Temporary storage for cube Geometry
  vector<VertexPNTBX> vtx(vbLen);
  vector<unsigned short> idx(ibLen);

  makeCube(1, vtx.begin(), idx.begin());
  g_cube.reset(new SimpleIndexedGeometryPNTBX(&vtx[0], &idx[0], vbLen, ibLen));
}

static void initSphere() {
  int ibLen, vbLen;
  getSphereVbIbLen(20, 10, vbLen, ibLen);

  // Temporary storage for sphere Geometry
  vector<VertexPNTBX> vtx(vbLen);
  vector<unsigned short> idx(ibLen);
  makeSphere(1, 20, 10, vtx.begin(), idx.begin());
  g_sphere.reset(new SimpleIndexedGeometryPNTBX(&vtx[0], &idx[0], vtx.size(), idx.size()));
}

static void initRobots() {
  // Init whatever geometry needed for the robots
}

// takes a projection matrix and send to the the shaders
inline void sendProjectionMatrix(Uniforms& uniforms, const Matrix4& projMatrix) {
  uniforms.put("uProjMatrix", projMatrix);
}

// update g_frustFovY from g_frustMinFov, g_windowWidth, and g_windowHeight
static void updateFrustFovY() {
  if (g_windowWidth >= g_windowHeight)
    g_frustFovY = g_frustMinFov;
  else {
    const double RAD_PER_DEG = 0.5 * CS175_PI/180;
    g_frustFovY = atan2(sin(g_frustMinFov * RAD_PER_DEG) * g_windowHeight / g_windowWidth, cos(g_frustMinFov * RAD_PER_DEG)) / RAD_PER_DEG;
  }
}

static Cvec3 lerp(Cvec3 src, Cvec3 dest, float alpha) {
  assert(0 <= alpha && alpha <= 1.0);
  float xout = ((1-alpha) * src[0]) + (alpha * dest[0]);
  float yout = ((1-alpha) * src[1]) + (alpha * dest[1]);
  float zout = ((1-alpha) * src[2]) + (alpha * dest[2]);
  return Cvec3(xout, yout, zout);
}

static Quat cond_neg(Quat q) {
  if (q[0] < 0) {
    return Quat(-q[0], -q[1], -q[2], -q[3]);
  }
  return q;
}

static Quat qpow(Quat q, float alpha) {
  Cvec3 axis = Cvec3(q[1], q[2], q[3]);

  float theta = atan2(sqrt(norm2(axis)), q[0]);

  if (norm2(axis) <= .001) {
    return Quat();
  }
  axis = normalize(axis);

  float q_outw = cos(alpha * theta);
  float q_outx = axis[0] * sin(alpha * theta);
  float q_outy = axis[1] * sin(alpha * theta);
  float q_outz = axis[2] * sin(alpha * theta);

  return normalize(Quat(q_outw, q_outx, q_outy, q_outz));
}

static Quat slerp(Quat src, Quat dest, float alpha) {
  assert(0 <= alpha && alpha <= 1.0);
  return normalize(qpow(cond_neg(dest * inv(src)), alpha) * src);
}

static Matrix4 makeProjectionMatrix() {
  return Matrix4::makeProjection(
           g_frustFovY, g_windowWidth / static_cast <double> (g_windowHeight),
           g_frustNear, g_frustFar);
}

enum ManipMode {
  ARCBALL_ON_PICKED,
  ARCBALL_ON_SKY,
  EGO_MOTION
};

static ManipMode getManipMode() {
  // if nothing is picked or the picked transform is the transfrom we are viewing from
  if (g_currentPickedRbtNode == NULL || g_currentPickedRbtNode == g_currentCameraNode) {
    if (g_currentCameraNode == g_skyNode && g_activeCameraFrame == WORLD_SKY)
      return ARCBALL_ON_SKY;
    else
      return EGO_MOTION;
  }
  else
    return ARCBALL_ON_PICKED;
}

static bool shouldUseArcball() {
  return getManipMode() != EGO_MOTION;
}

// The translation part of the aux frame either comes from the current
// active object, or is the identity matrix when
static RigTForm getArcballRbt() {
  switch (getManipMode()) {
  case ARCBALL_ON_PICKED:
    return getPathAccumRbt(g_world, g_currentPickedRbtNode);
  case ARCBALL_ON_SKY:
    return RigTForm();
  case EGO_MOTION:
    return getPathAccumRbt(g_world, g_currentCameraNode);
  default:
    throw runtime_error("Invalid ManipMode");
  }
}

static void updateArcballScale() {
  RigTForm arcballEye = inv(getPathAccumRbt(g_world, g_currentCameraNode)) * getArcballRbt();
  double depth = arcballEye.getTranslation()[2];
  if (depth > -CS175_EPS)
    g_arcballScale = 0.02;
  else
    g_arcballScale = getScreenToEyeScale(depth, g_frustFovY, g_windowHeight);
}

static void drawArcBall(Uniforms& uniforms) {
  // switch to wire frame mode
  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

  RigTForm arcballEye = inv(getPathAccumRbt(g_world, g_currentCameraNode)) * getArcballRbt();
  Matrix4 MVM = rigTFormToMatrix(arcballEye) * Matrix4::makeScale(Cvec3(1, 1, 1) * g_arcballScale * g_arcballScreenRadius);
  sendModelViewNormalMatrix(uniforms, MVM, normalMatrix(MVM));

  uniforms.put("uColor", Cvec3 (0.27, 0.82, 0.35));

  // switch back to solid mode
  g_arcballMat->draw(*g_sphere, uniforms);
}

static void drawStuff(bool picking) {

  Uniforms uniforms;
  // if we are not translating, update arcball scale
  if (!(g_mouseMClickButton || (g_mouseLClickButton && g_mouseRClickButton) || (g_mouseLClickButton && !g_mouseRClickButton && g_spaceDown)))
    updateArcballScale();

  // build & send proj. matrix to vshader
  const Matrix4 projmat = makeProjectionMatrix();
  sendProjectionMatrix(uniforms, projmat);

  const RigTForm eyeRbt = getPathAccumRbt(g_world, g_currentCameraNode);
  const RigTForm invEyeRbt = inv(eyeRbt);

  // const Cvec3 eyeLight1 = Cvec3(invEyeRbt * Cvec4(g_light1, 1));
  // const Cvec3 eyeLight2 = Cvec3(invEyeRbt * Cvec4(g_light2, 1));

  const Cvec3 eyeLight1 = getPathAccumRbt(g_world, g_light1Node).getTranslation();
  const Cvec3 eyeLight2 = getPathAccumRbt(g_world, g_light2Node).getTranslation();

  uniforms.put("uLight", (Cvec3) (invEyeRbt * Cvec4(eyeLight1,1)));
  uniforms.put("uLight2", (Cvec3) (invEyeRbt * Cvec4(eyeLight2,1)));

  if (!picking) {
    Drawer drawer(invEyeRbt, uniforms);
    g_world->accept(drawer);

    if (g_displayArcball && shouldUseArcball())
      drawArcBall(uniforms);
  }
  else {
    Picker picker(invEyeRbt, uniforms);
    g_overridingMaterial = g_pickingMat;
    g_world->accept(picker);
    g_overridingMaterial.reset();
    glFlush();
    g_currentPickedRbtNode = picker.getRbtNodeAtXY(g_mouseClickX, g_mouseClickY);
    if (g_currentPickedRbtNode == g_groundNode)
      g_currentPickedRbtNode = shared_ptr<SgRbtNode>(); // set to NULL

    cout << (g_currentPickedRbtNode ? "Part picked" : "No part picked") << endl;
  }
}

static void display() {
  // No more glUseProgram

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  drawStuff(false); // no more curSS

  glutSwapBuffers();

  checkGlErrors();
}

static void pick() {
  // We need to set the clear color to black, for pick rendering.
  // so let's save the clear color
  GLdouble clearColor[4];
  glGetDoublev(GL_COLOR_CLEAR_VALUE, clearColor);

  glClearColor(0, 0, 0, 0);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // No more glUseProgram
  drawStuff(true); // no more curSS

  // Uncomment below and comment out the glutPostRedisplay in mouse(...) call back
  // to see result of the pick rendering pass
  // glutSwapBuffers();

  //Now set back the clear color
  glClearColor(clearColor[0], clearColor[1], clearColor[2], clearColor[3]);

  checkGlErrors();
}

static void reshape(const int w, const int h) {
  g_windowWidth = w;
  g_windowHeight = h;
  glViewport(0, 0, w, h);
  cerr << "Size of window is now " << w << "x" << h << endl;
  g_arcballScreenRadius = max(1.0, min(h, w) * 0.25);
  updateFrustFovY();
  glutPostRedisplay();
}

static Cvec3 getArcballDirection(const Cvec2& p, const double r) {
  double n2 = norm2(p);
  if (n2 >= r*r)
    return normalize(Cvec3(p, 0));
  else
    return normalize(Cvec3(p, sqrt(r*r - n2)));
}

static RigTForm moveArcball(const Cvec2& p0, const Cvec2& p1) {
  const Matrix4 projMatrix = makeProjectionMatrix();
  const RigTForm eyeInverse = inv(getPathAccumRbt(g_world, g_currentCameraNode));
  const Cvec3 arcballCenter = getArcballRbt().getTranslation();
  const Cvec3 arcballCenter_ec = Cvec3(eyeInverse * Cvec4(arcballCenter, 1));

  if (arcballCenter_ec[2] > -CS175_EPS)
    return RigTForm();

  Cvec2 ballScreenCenter = getScreenSpaceCoord(arcballCenter_ec,
                                               projMatrix, g_frustNear, g_frustFovY, g_windowWidth, g_windowHeight);
  const Cvec3 v0 = getArcballDirection(p0 - ballScreenCenter, g_arcballScreenRadius);
  const Cvec3 v1 = getArcballDirection(p1 - ballScreenCenter, g_arcballScreenRadius);

  return RigTForm(Quat(0.0, v1[0], v1[1], v1[2]) * Quat(0.0, -v0[0], -v0[1], -v0[2]));
}

static RigTForm doMtoOwrtA(const RigTForm& M, const RigTForm& O, const RigTForm& A) {
  return A * M * inv(A) * O;
}

static RigTForm getMRbt(const double dx, const double dy) {
  RigTForm M;

  if (g_mouseLClickButton && !g_mouseRClickButton && !g_spaceDown) {
    if (shouldUseArcball())
      M = moveArcball(Cvec2(g_mouseClickX, g_mouseClickY), Cvec2(g_mouseClickX + dx, g_mouseClickY + dy));
    else
      M = RigTForm(Quat::makeXRotation(-dy) * Quat::makeYRotation(dx));
  }
  else {
    double movementScale = getManipMode() == EGO_MOTION ? 0.02 : g_arcballScale;
    if (g_mouseRClickButton && !g_mouseLClickButton) {
      M = RigTForm(Cvec3(dx, dy, 0) * movementScale);
    }
    else if (g_mouseMClickButton || (g_mouseLClickButton && g_mouseRClickButton) || (g_mouseLClickButton && g_spaceDown)) {
      M = RigTForm(Cvec3(0, 0, -dy) * movementScale);
    }
  }

  switch (getManipMode()) {
  case ARCBALL_ON_PICKED:
    break;
  case ARCBALL_ON_SKY:
    M = inv(M);
    break;
  case EGO_MOTION:
    if (g_mouseLClickButton && !g_mouseRClickButton && !g_spaceDown) // only invert rotation
      M = inv(M);
    break;
  }
  return M;
}

static RigTForm makeMixedFrame(const RigTForm& objRbt, const RigTForm& eyeRbt) {
  return transFact(objRbt) * linFact(eyeRbt);
}

// l = w X Y Z
// o = l O
// a = w A = l (Z Y X)^1 A = l A'
// o = a (A')^-1 O
//   => a M (A')^-1 O = l A' M (A')^-1 O

static void motion(const int x, const int y) {
  if (!g_mouseClickDown)
    return;

  const double dx = x - g_mouseClickX;
  const double dy = g_windowHeight - y - 1 - g_mouseClickY;

  const RigTForm M = getMRbt(dx, dy);   // the "action" matrix

  // the matrix for the auxiliary frame (the w.r.t.)
  RigTForm A = makeMixedFrame(getArcballRbt(), getPathAccumRbt(g_world, g_currentCameraNode));

  shared_ptr<SgRbtNode> target;
  switch (getManipMode()) {
  case ARCBALL_ON_PICKED:
    target = g_currentPickedRbtNode;
    break;
  case ARCBALL_ON_SKY:
    target = g_skyNode;
    break;
  case EGO_MOTION:
    target = g_currentCameraNode;
    break;
  }

  A = inv(getPathAccumRbt(g_world, target, 1)) * A;

  target->setRbt(doMtoOwrtA(M, target->getRbt(), A));

  g_mouseClickX += dx;
  g_mouseClickY += dy;
  glutPostRedisplay();  // we always redraw if we changed the scene
}

static void mouse(const int button, const int state, const int x, const int y) {
  g_mouseClickX = x;
  g_mouseClickY = g_windowHeight - y - 1;  // conversion from GLUT window-coordinate-system to OpenGL window-coordinate-system

  g_mouseLClickButton |= (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN);
  g_mouseRClickButton |= (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN);
  g_mouseMClickButton |= (button == GLUT_MIDDLE_BUTTON && state == GLUT_DOWN);

  g_mouseLClickButton &= !(button == GLUT_LEFT_BUTTON && state == GLUT_UP);
  g_mouseRClickButton &= !(button == GLUT_RIGHT_BUTTON && state == GLUT_UP);
  g_mouseMClickButton &= !(button == GLUT_MIDDLE_BUTTON && state == GLUT_UP);

  g_mouseClickDown = g_mouseLClickButton || g_mouseRClickButton || g_mouseMClickButton;

  if (g_pickingMode && button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
    pick();
    g_pickingMode = false;
    cerr << "Picking mode is off" << endl;
    glutPostRedisplay(); // request redisplay since the arcball will have moved
  }
  glutPostRedisplay();
}

static void keyboardUp(const unsigned char key, const int x, const int y) {
  switch (key) {
  case ' ':
    g_spaceDown = false;
    break;
  }
  glutPostRedisplay();
}

static void keyboard(const unsigned char key, const int x, const int y) {
  if (animating) {
    return;
  }
  switch (key) {
  case ' ':
    g_spaceDown = true;
    break;
  case 27:
    exit(0);                                  // ESC
  case 'h':
    cout << " ============== H E L P ==============\n\n"
    << "h\t\thelp menu\n"
    << "s\t\tsave screenshot\n"
    << "f\t\tToggle flat shading on/off.\n"
    << "p\t\tUse mouse to pick a part to edit\n"
    << "v\t\tCycle view\n"
    << "drag left mouse to rotate\n" << endl;
    break;
  case 's':
    glFlush();
    writePpmScreenshot(g_windowWidth, g_windowHeight, "out.ppm");
    break;
  case 'f':
    cout << "Nein!" << endl;
    break;
  case 'v':
  {
    shared_ptr<SgRbtNode> viewers[] = {g_skyNode, g_robot1Node, g_robot2Node};
    for (int i = 0; i < 3; ++i) {
      if (g_currentCameraNode == viewers[i]) {
        g_currentCameraNode = viewers[(i+1)%3];
        break;
      }
    }
  }
  break;
  case 'p':
    g_pickingMode = !g_pickingMode;
    cerr << "Picking mode is " << (g_pickingMode ? "on" : "off") << endl;
    break;
  case 'm':
    g_activeCameraFrame = SkyMode((g_activeCameraFrame+1) % 2);
    cerr << "Editing sky eye w.r.t. " << (g_activeCameraFrame == WORLD_SKY ? "world-sky frame\n" : "sky-sky frame\n") << endl;
    break;
  case 'c':
    cout << "clicked c" << endl;
    break;
  case 'u':
    cout << "clicked u" << endl;
    break;
  case '>':
    next_frame();
    cout << "clicked >" << endl;
    break;
  case '<':
    prev_frame();
    cout << "clicked <" << endl;
    break;
  case 'n':
    cout << "making snapshot of current scene graph" << endl;
    make_frame();
    break;
  case 'd':
    cout << "clicked d" << endl;
    delete_frame();
    break;
  case 'i':
    cout << "Reading animation from animation.txt" << endl;
    read_frame();
    break;
  case 'w':
    cout << "Writing animation to animation.txt" << endl;
    write_frame();
    break;
  case 'y':
    if (key_frames.size() < 4) {
      cout << "Cannot play animation with fewer than 4 keyframes." << endl;
      break;
    }
    animating = !animating;
    animateTimerCallback(0);
    break;
  case '+':
    g_msBetweenKeyFrames -= 100;
    cout << g_msBetweenKeyFrames << " ms between keyframes." << endl;
    break;
  case '-':
    g_msBetweenKeyFrames += 100;
    cout << g_msBetweenKeyFrames << " ms between keyframes." << endl;
    break;
  }
  glutPostRedisplay();
}

static void initGlutState(int argc, char * argv[]) {
  glutInit(&argc, argv);                                  // initialize Glut based on cmd-line args
#ifdef __MAC__
  glutInitDisplayMode(GLUT_3_2_CORE_PROFILE|GLUT_RGBA|GLUT_DOUBLE|GLUT_DEPTH); // core profile flag is required for GL 3.2 on Mac
#else
  glutInitDisplayMode(GLUT_RGBA|GLUT_DOUBLE|GLUT_DEPTH);  //  RGBA pixel channels and double buffering
#endif
  glutInitWindowSize(g_windowWidth, g_windowHeight);      // create a window
  glutCreateWindow("Assignment 4");                       // title the window

  glutIgnoreKeyRepeat(true);                              // avoids repeated keyboard calls when holding space to emulate middle mouse

  glutDisplayFunc(display);                               // display rendering callback
  glutReshapeFunc(reshape);                               // window reshape callback
  glutMotionFunc(motion);                                 // mouse movement callback
  glutMouseFunc(mouse);                                   // mouse click callback
  glutKeyboardFunc(keyboard);
  glutKeyboardUpFunc(keyboardUp);
}

static void initGLState() {
  glClearColor(128./255., 200./255., 255./255., 0.);
  glClearDepth(0.);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glCullFace(GL_BACK);
  glEnable(GL_CULL_FACE);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_GREATER);
  glReadBuffer(GL_BACK);
  if (!g_Gl2Compatible)
    glEnable(GL_FRAMEBUFFER_SRGB);
}

static void initMaterials() {
  // Create some prototype materials
  Material diffuse("./shaders/basic-gl3.vshader", "./shaders/diffuse-gl3.fshader");
  Material solid("./shaders/basic-gl3.vshader", "./shaders/solid-gl3.fshader");

  // copy diffuse prototype and set red color
  g_redDiffuseMat.reset(new Material(diffuse));
  g_redDiffuseMat->getUniforms().put("uColor", Cvec3f(1, 0, 0));

  // copy diffuse prototype and set blue color
  g_blueDiffuseMat.reset(new Material(diffuse));
  g_blueDiffuseMat->getUniforms().put("uColor", Cvec3f(0, 0, 1));

  // normal mapping material
  g_bumpFloorMat.reset(new Material("./shaders/normal-gl3.vshader", "./shaders/normal-gl3.fshader"));
  g_bumpFloorMat->getUniforms().put("uTexColor", shared_ptr<ImageTexture>(new ImageTexture("Fieldstone.ppm", true)));
  g_bumpFloorMat->getUniforms().put("uTexNormal", shared_ptr<ImageTexture>(new ImageTexture("FieldstoneNormal.ppm", false)));

  // copy solid prototype, and set to wireframed rendering
  g_arcballMat.reset(new Material(solid));
  g_arcballMat->getUniforms().put("uColor", Cvec3f(0.27f, 0.82f, 0.35f));
  g_arcballMat->getRenderStates().polygonMode(GL_FRONT_AND_BACK, GL_LINE);

  // copy solid prototype, and set to color white
  g_lightMat.reset(new Material(solid));
  g_lightMat->getUniforms().put("uColor", Cvec3f(1, 1, 1));

  // pick shader
  g_pickingMat.reset(new Material("./shaders/basic-gl3.vshader", "./shaders/pick-gl3.fshader"));
};


static void initGeometry() {
  initGround();
  initCubes();
  initSphere();
  initRobots();
}

static void constructRobot(shared_ptr<SgTransformNode> base, shared_ptr<Material> material) {
  const double ARM_LEN = 0.7,
               ARM_THICK = 0.25,
               LEG_LEN = 1,
               LEG_THICK = 0.25,
               TORSO_LEN = 1.5,
               TORSO_THICK = 0.25,
               TORSO_WIDTH = 1,
               HEAD_SIZE = 0.7;
  const int NUM_JOINTS = 10,
            NUM_SHAPES = 10;

  struct JointDesc {
    int parent;
    float x, y, z;
  };

  JointDesc jointDesc[NUM_JOINTS] = {
    {-1}, // torso
    {0,  TORSO_WIDTH/2, TORSO_LEN/2, 0}, // upper right arm
    {0, -TORSO_WIDTH/2, TORSO_LEN/2, 0}, // upper left arm
    {1,  ARM_LEN, 0, 0}, // lower right arm
    {2, -ARM_LEN, 0, 0}, // lower left arm
    {0, TORSO_WIDTH/2-LEG_THICK/2, -TORSO_LEN/2, 0}, // upper right leg
    {0, -TORSO_WIDTH/2+LEG_THICK/2, -TORSO_LEN/2, 0}, // upper left leg
    {5, 0, -LEG_LEN, 0}, // lower right leg
    {6, 0, -LEG_LEN, 0}, // lower left
    {0, 0, TORSO_LEN/2, 0} // head
  };

  struct ShapeDesc {
    int parentJointId;
    float x, y, z, sx, sy, sz;
    shared_ptr<Geometry> geometry;
  };

  ShapeDesc shapeDesc[NUM_SHAPES] = {
    {0, 0,         0, 0, TORSO_WIDTH, TORSO_LEN, TORSO_THICK, g_cube}, // torso
    {1, ARM_LEN/2, 0, 0, ARM_LEN/2, ARM_THICK/2, ARM_THICK/2, g_sphere}, // upper right arm
    {2, -ARM_LEN/2, 0, 0, ARM_LEN/2, ARM_THICK/2, ARM_THICK/2, g_sphere}, // upper left arm
    {3, ARM_LEN/2, 0, 0, ARM_LEN, ARM_THICK, ARM_THICK, g_cube}, // lower right arm
    {4, -ARM_LEN/2, 0, 0, ARM_LEN, ARM_THICK, ARM_THICK, g_cube}, // lower left arm
    {5, 0, -LEG_LEN/2, 0, LEG_THICK/2, LEG_LEN/2, LEG_THICK/2, g_sphere}, // upper right leg
    {6, 0, -LEG_LEN/2, 0, LEG_THICK/2, LEG_LEN/2, LEG_THICK/2, g_sphere}, // upper left leg
    {7, 0, -LEG_LEN/2, 0, LEG_THICK, LEG_LEN, LEG_THICK, g_cube}, // lower right leg
    {8, 0, -LEG_LEN/2, 0, LEG_THICK, LEG_LEN, LEG_THICK, g_cube}, // lower left leg
    {9, 0, HEAD_SIZE/2 * 1.5, 0, HEAD_SIZE/2, HEAD_SIZE/2, HEAD_SIZE/2, g_sphere}, // head
  };

  shared_ptr<SgTransformNode> jointNodes[NUM_JOINTS];

  for (int i = 0; i < NUM_JOINTS; ++i) {
    if (jointDesc[i].parent == -1)
      jointNodes[i] = base;
    else {
      jointNodes[i].reset(new SgRbtNode(RigTForm(Cvec3(jointDesc[i].x, jointDesc[i].y, jointDesc[i].z))));
      jointNodes[jointDesc[i].parent]->addChild(jointNodes[i]);
    }
  }
  for (int i = 0; i < NUM_SHAPES; ++i) {
    shared_ptr<SgGeometryShapeNode> shape(
      new MyShapeNode(shapeDesc[i].geometry,
                      material, // USE MATERIAL as opposed to color
                      Cvec3(shapeDesc[i].x, shapeDesc[i].y, shapeDesc[i].z),
                      Cvec3(0, 0, 0),
                      Cvec3(shapeDesc[i].sx, shapeDesc[i].sy, shapeDesc[i].sz)));
    jointNodes[shapeDesc[i].parentJointId]->addChild(shape);
  }
}

static void initScene() {
  g_world.reset(new SgRootNode());

  g_light1Node.reset(new SgRbtNode(RigTForm(g_light1)));
  g_light2Node.reset(new SgRbtNode(RigTForm(g_light2)));

  g_skyNode.reset(new SgRbtNode(RigTForm(Cvec3(0.0, 0.25, 4.0))));

  g_groundNode.reset(new SgRbtNode());
  g_groundNode->addChild(shared_ptr<MyShapeNode>(
                           new MyShapeNode(g_ground, g_bumpFloorMat, Cvec3(0, g_groundY, 0))));

  g_robot1Node.reset(new SgRbtNode(RigTForm(Cvec3(-2, 1, 0))));
  g_robot2Node.reset(new SgRbtNode(RigTForm(Cvec3(2, 1, 0))));

  constructRobot(g_robot1Node, g_redDiffuseMat); // a Red robot
  constructRobot(g_robot2Node, g_blueDiffuseMat); // a Blue robot

  g_world->addChild(g_skyNode);
  g_world->addChild(g_groundNode);
  g_world->addChild(g_robot1Node);
  g_world->addChild(g_robot2Node);
  g_world->addChild(g_light1Node);
  g_world->addChild(g_light2Node);

  g_light1Node->addChild(shared_ptr<MyShapeNode>(
                           new MyShapeNode(g_sphere, g_lightMat, Cvec3(0,0,0))));

  g_light2Node->addChild(shared_ptr<MyShapeNode>(
                           new MyShapeNode(g_sphere, g_lightMat, Cvec3(0,0,0))));

  g_currentCameraNode = g_skyNode;
}

int main(int argc, char * argv[]) {
  try {
    initGlutState(argc,argv);

    // on Mac, we shouldn't use GLEW.

#ifndef __MAC__
    glewInit(); // load the OpenGL extensions
#endif

    cout << (g_Gl2Compatible ? "Will use OpenGL 2.x / GLSL 1.0" : "Will use OpenGL 3.x / GLSL 1.5") << endl;

#ifndef __MAC__
    if ((!g_Gl2Compatible) && !GLEW_VERSION_3_0)
      throw runtime_error("Error: card/driver does not support OpenGL Shading Language v1.3");
    else if (g_Gl2Compatible && !GLEW_VERSION_2_0)
      throw runtime_error("Error: card/driver does not support OpenGL Shading Language v1.0");
#endif

    initGLState();
    initMaterials();
    initGeometry();
    initScene();

    glutMainLoop();
    return 0;
  }
  catch (const runtime_error& e) {
    cout << "Exception caught: " << e.what() << endl;
    return -1;
  }
}
