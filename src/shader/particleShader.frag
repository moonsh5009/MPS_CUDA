#version 420 core

in vec3 lightDir;
in vec3 viewDir;

out vec4 result;

void main(void)
{
    vec3 normal;
    normal.xy = vec2(gl_PointCoord.x * 2.f - 1.f, -gl_PointCoord.y * 2.f + 1.f);
    float mag = dot(normal.xy, normal.xy);
    if (mag > 1.0)
        discard;
    normal.z = sqrt(1.0 - mag);
    
    vec3 lightColor = vec3(1.0, 1.0, 1.0);
    vec3 reflectDir = reflect(-lightDir, normal);
    float specularStrength = 0.3f;

    float ambient = 0.1f;
    float diffuse = max(0.0, dot(lightDir, normal));
    float specular = specularStrength * pow(max(dot(viewDir, reflectDir), 0.0), 64.f);

    result = vec4(0.3f, 0.8f, 0.8f, 1.f) * vec4(min(ambient + diffuse + specular, 1.f) * lightColor, 1.f);
} 