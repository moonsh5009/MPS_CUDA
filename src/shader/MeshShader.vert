#include <MeshShader.glsl>

layout (std430, binding = 0) restrict readonly buffer Face { uint aFace[]; };
layout (std430, binding = 1) restrict readonly buffer Vertex { double aVertex[]; };
layout (std430, binding = 2) restrict readonly buffer FrontColor { vec4 aFrontColor[]; };
layout (std430, binding = 3) restrict readonly buffer BackColor { vec4 aBackColor[]; };

out VertexShaderOut vsOut;

void main()
{
	uint idxFace = gl_VertexID / 3;
	uint iVertex = gl_VertexID - idxFace * 3;

	uint idxVertex = aFace[idxFace * 3 + iVertex];

	vsOut.pos = vec3(float(aVertex[idxVertex * 3 + 0]), float(aVertex[idxVertex * 3 + 1]), -float(aVertex[idxVertex * 3 + 2]));
	vsOut.frontColor = aFrontColor[idxFace];
	vsOut.backColor = aBackColor[idxFace];
	
	vec4 viewPos = uCamera.viewMat * vec4(vsOut.pos, 1.0);
    vec4 lightPos = uCamera.viewMat * vec4(vec3(uLight.pos), 1.f);
	vsOut.unit2Light = normalize((lightPos - viewPos).xyz);
	vsOut.lightColor = uLight.color.xyz;
    
    vsOut.pos = vec3(viewPos);
    gl_Position = uCamera.projMat * viewPos;
}