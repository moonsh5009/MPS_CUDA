#version 450 core
// Begin include: MeshShader.glsl

layout(std140, binding = 0) uniform Camera
{
    mat4 viewMat;
    mat4 projMat;
    mat4 viewInvMat;
    mat4 projInvMat;
} uCamera;

layout(std140, binding = 1) uniform Light
{
    vec3 pos;
    vec4 color;
} uLight;

struct VertexShaderOut
{
    vec3 pos;
    vec4 frontColor;
    vec4 backColor;
    vec3 unit2Light;
    vec3 lightColor;
};// End include: MeshShader.glsl

layout(location = 0) in VertexShaderOut vsOut;

layout(location = 0) out vec4 result;

void main(void)
{
    result = vsOut.frontColor;
    //result = vec4(0.0, 0.0, 0.0, 1.0);
}