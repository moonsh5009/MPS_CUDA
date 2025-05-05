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

layout (std430, binding = 0) restrict readonly buffer Face { uint aFace[]; };
layout (std430, binding = 1) restrict readonly buffer Vertex { float[3] aVertex[]; };
layout (std430, binding = 2) restrict readonly buffer FrontColor { vec4 aFrontColor[]; };
layout (std430, binding = 3) restrict readonly buffer BackColor { vec4 aBackColor[]; };

layout(location = 0) out VertexShaderOut vsOut;

void main()
{
	uint idxFace = gl_VertexIndex / 3;
	uint iVertex = gl_VertexIndex - idxFace * 3;

	uint idxVertex = aFace[idxFace * 3 + iVertex];
	float[3] vertex_array = aVertex[idxVertex];
	vsOut.pos = vec3(vertex_array[0], vertex_array[1], vertex_array[2]);
	vsOut.frontColor = aFrontColor[idxFace];
	vsOut.backColor = aBackColor[idxFace];
	
	vec4 viewPos = uCamera.viewMat * vec4(vsOut.pos, 1.0);
    vec4 lightPos = uCamera.viewMat * vec4(vec3(uLight.pos), 1.f);
	vsOut.unit2Light = normalize((lightPos - viewPos).xyz);
	vsOut.lightColor = uLight.color.xyz;
    
    vsOut.pos = vec3(viewPos);
    gl_Position = uCamera.projMat * viewPos;
}