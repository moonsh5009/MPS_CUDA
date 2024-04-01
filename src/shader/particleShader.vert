#include <particleShader.glsl>

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

layout (std430, binding = 0) restrict readonly buffer ParticlePos { double aParticlePos[]; };
layout (std430, binding = 1) restrict readonly buffer ParticleRadius { double aParticleRadius[]; };
layout (std430, binding = 2) restrict readonly buffer ParticleColor { vec4 aParticleColor[]; };

layout (location = 0) in vec2 inUV;

out VertexShaderOut vsOut;

void main()
{
    vsOut.uv = inUV;
	vsOut.pos = vec3(float(aParticlePos[gl_InstanceID * 3 + 0]), float(aParticlePos[gl_InstanceID * 3 + 1]), float(aParticlePos[gl_InstanceID * 3 + 2]));
	vsOut.radius = float(aParticleRadius[gl_InstanceID]);
	vsOut.color = vec4(0.3f, 0.8f, 0.8f, 1.0f);//aParticleColor[gl_InstanceID];
	
	vec4 viewPos = uCamera.viewMat * vec4(vsOut.pos.xy, -vsOut.pos.z, 1.0);
    vec4 lightPos = uCamera.viewMat * vec4(vec3(uLight.pos), 1.f);
	vsOut.unit2Light = normalize((lightPos - viewPos).xyz);
	vsOut.lightColor = uLight.color.xyz;

    gl_Position = uCamera.projMat * (viewPos + vec4(inUV * vsOut.radius, 0.0f, 0.0f));
}