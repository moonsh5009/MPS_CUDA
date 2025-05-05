#version 450 core

layout (std140, binding = 0) uniform Camera
{
    mat4 viewMat;
    mat4 projMat;
    mat4 viewInvMat;
    mat4 projInvMat;
} uCamera;

layout (std140, binding = 1) uniform Light
{
    vec3 pos;
    vec4 color;
} uLight;

struct VertexShaderOut
{
    vec2 uv;
    vec3 pos;
    float radius;
    vec4 color;
    vec3 unit2Light;
    vec3 lightColor;
};