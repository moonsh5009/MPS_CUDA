#include <particleShader.glsl>

in VertexShaderOut vsOut;

out vec4 result;

void main(void)
{
    vec3 norm = vec3(vsOut.uv, 0.0f);
    norm.z = dot(vsOut.uv, vsOut.uv);
    if (norm.z > 1.0)
        discard;

    norm.z = sqrt(1.0 - norm.z);
    vec3 pos = vsOut.pos + norm * vsOut.radius;

    float ambientStrength = 0.1;
    float diffuseStrength = 0.8;
    float specularStrength = 0.3;

    vec3 zDir = vec3(0.0, 0.0, 1.0);
    vec3 reflectDir = reflect(-vsOut.unit2Light, norm);

    float ambient = ambientStrength;
    float diffuse = max(dot(norm, vsOut.unit2Light), 0.0) * diffuseStrength;
    float specular = pow(max(dot(zDir, reflectDir), 0.0), 64.0) * specularStrength;
    
    //result = vec4(((ambient + diffuse) * vsOut.color.xyz + specular * vsOut.lightColor), 1.0);
    result = vec4(((ambient + diffuse) * vec3(0.3, 0.8, 0.8) + specular * vec3(1.0, 1.0, 1.0)), 1.0);
}