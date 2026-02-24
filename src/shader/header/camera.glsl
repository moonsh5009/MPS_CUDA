#include "common.glsl"

layout(set = SET_INDEX.UNIFORM, binding = UNIFORM_BINDING_INDEX.CAMERA, std140) uniform CameraBlock {
    mat4 view_mat;
    mat4 proj_mat;
    mat4 view_inv_mat;
    mat4 proj_inv_mat;
    vec3 position;
    vec3 view_dir;
    ivec4 viewport;
    vec2 frustum;
} u_camera;

mat4 getViewMat() {
    return u_camera.view_mat;
}

mat4 getProjMat() {
    return u_camera.proj_mat;
}

mat4 getViewInvMat() {
    return u_camera.view_inv_mat;
}

mat4 getProjInvMat() {
    return u_camera.proj_inv_mat;
}

vec3 getPosition() {
    return u_camera.position;
}

vec3 getViewDir() {
    return u_camera.view_dir;
}

ivec4 getViewport() {
    return u_camera.viewport;
}

ivec2 getScreenSize() {
    return ivec2(u_camera.viewport.z, u_camera.viewport.w);
}

bool isPerspective() {
    return u_camera.proj_mat[2][3] != 0.0;
}

vec2 getFrustum() {
    return u_camera.frustum;
}