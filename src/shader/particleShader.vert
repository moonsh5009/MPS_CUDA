#include <particleShader.glsl>

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

layout (location = 0) in vec2 inUV;
layout (location = 1) in vec3 inPos;
layout (location = 2) in float inRadius;
layout (location = 3) in vec4 inColor;

out VertexShaderOut vsOut;

void main()
{
    vsOut.uv = inUV;
	vsOut.pos = inPos;
	vsOut.radius = inRadius;
	vsOut.color = inColor;
	
	vec4 viewPos = uCamera.viewMat * vec4(inPos.xy, -inPos.z, 1.0);
    vec4 lightPos = uCamera.viewMat * vec4(uLight.pos, 1.f);
	vsOut.unit2Light = normalize((lightPos - viewPos).xyz);
	vsOut.lightColor = uLight.color.xyz;

    gl_Position = uCamera.projMat * (viewPos + vec4(inUV * inRadius, 0.0f, 0.0f));
}