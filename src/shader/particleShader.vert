#include <ParticleShader.glsl>

layout (std430, binding = 0) restrict readonly buffer ParticlePos { double aParticlePos[]; };
layout (std430, binding = 1) restrict readonly buffer ParticleRadius { double aParticleRadius[]; };
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

    vsOut.uv = UVs[gl_VertexID];
	vsOut.pos = vec3(float(aParticlePos[gl_InstanceID * 3 + 0]), float(aParticlePos[gl_InstanceID * 3 + 1]), -float(aParticlePos[gl_InstanceID * 3 + 2]));
	vsOut.radius = float(aParticleRadius[gl_InstanceID]);
	vsOut.color = aParticleColor[gl_InstanceID];
	
	vec4 viewPos = uCamera.viewMat * vec4(vsOut.pos, 1.0);
    vec4 lightPos = uCamera.viewMat * vec4(vec3(uLight.pos), 1.f);
	vsOut.unit2Light = normalize((lightPos - viewPos).xyz);
	vsOut.lightColor = uLight.color.xyz;
    
    vsOut.pos = vec3(viewPos);
    gl_Position = uCamera.projMat * (viewPos + vec4(vsOut.uv * vsOut.radius, 0.0f, 0.0f));
}