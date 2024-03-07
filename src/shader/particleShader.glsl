#version 420 core

struct VertexShaderOut
{
    vec2 uv;
    vec3 pos;
    float radius;
    vec4 color;
    vec3 unit2Light;
    vec3 lightColor;
};