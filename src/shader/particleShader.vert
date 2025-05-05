#version 450 core
#include "ParticleShader.glsl"

layout (std430, binding = 0) restrict readonly buffer ParticlePos { float[3] aParticlePos[]; };
layout (std430, binding = 1) restrict readonly buffer ParticleRadius { float aParticleRadius[]; };
layout (std430, binding = 2) restrict readonly buffer ParticleColor { vec4 aParticleColor[]; };

out VertexShaderOut vsOut;

void main()
{
	vec2 UVs[4] = 
	{
		vec2(-1.0, -1.0),
		vec2(1.0, -1.0),
		vec2(-1.0, 1.0),
		vec2(1.0, 1.0)
	};
	float[3] vertex_array = aParticlePos[gl_InstanceIndex];

    vsOut.uv = UVs[gl_VertexIndex];
	vsOut.pos = vec3(vertex_array[0], vertex_array[1], vertex_array[2]);
	vsOut.radius = aParticleRadius[gl_InstanceIndex];
	vsOut.color = aParticleColor[gl_InstanceIndex];
	
	vec4 viewPos = uCamera.viewMat * vec4(vsOut.pos, 1.0);
    vec4 lightPos = uCamera.viewMat * vec4(vec3(uLight.pos), 1.f);
	vsOut.unit2Light = normalize((lightPos - viewPos).xyz);
	vsOut.lightColor = uLight.color.xyz;
    
    vsOut.pos = vec3(viewPos);
    gl_Position = uCamera.projMat * (viewPos + vec4(vsOut.uv * vsOut.radius, 0.0f, 0.0f));
}