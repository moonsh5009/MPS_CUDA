#version 420 core

layout (std140, binding = 1) uniform Camera
{
    mat4 viewMat;
    mat4 projMat;
    mat4 viewInvMat;
    mat4 projInvMat;
} uCamera;
layout (std140, binding = 2) uniform Light
{
    vec3 pos;
    vec4 color;
} uLight;

layout (location = 0) in vec3 position;

out vec3 lightDir;
out vec3 viewDir;

void main()
{
    float radius = 180.;

    vec4 posClip = uCamera.viewMat * vec4(position, 1.f);
    float dist = length(posClip.xyz);
    
    vec4 lightClip = uCamera.viewMat * vec4(uLight.pos, 1.f);
    lightDir = normalize((lightClip - posClip).xyz);
    viewDir = normalize(-posClip.xyz);

    gl_Position = uCamera.projMat * posClip;
    gl_PointSize = radius / gl_Position.w;
}