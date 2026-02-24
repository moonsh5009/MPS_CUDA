#include "common.glsl"

struct LightColor {
    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
};

struct Light {
    vec4 pos;
    LightColor color;
    mat4 shadow[4];
    vec4 layer_distance;
    int layer_num;
    int is_shadow_map;
};

layout(set = SET_INDEX.UNIFORM, binding = UNIFORM_BINDING_INDEX.LIGHT, std140) uniform LightBlock {
    Light u_light;
};

LightColor calcBlinnPhongLight(vec3 vertex_norm, vec3 eye_dir, float shininess) {
    float blinn_phong_shininess = max(0.0, shininess * 2.0);
    vec3 reverse_dir_light = -u_light.pos.xyz;
    vec3 halfway_dir = normalize(reverse_dir_light + eye_dir);

    LightColor result;
    result.ambient = u_light.color.ambient;
    result.diffuse = u_light.color.diffuse * max(dot(reverse_dir_light, vertex_norm), 0.0);
    result.specular = u_light.color.specular * pow(max(dot(halfway_dir, vertex_norm), 0.0), blinn_phong_shininess);
    return result;
}