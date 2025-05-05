#version 450 core
#include "MeshShader.glsl"

in VertexShaderOut vsOut;

out vec4 result;

void main(void)
{
    result = vsOut.frontColor;
    //result = vec4(0.0, 0.0, 0.0, 1.0);
}