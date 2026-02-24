#include "camera.glsl"
#include "light.glsl"

struct ENUM_COLOR_TYPE {
    int CONST;
    int ATTRIBUTE;
};
const ENUM_COLOR_TYPE COLOR_TYPE = ENUM_COLOR_TYPE(0, 1);

struct RenderConfig {
    vec4 const_color;
    int color_type;
    int transparent_type;
    
    int model_type;

    int use_point_size;
    float point_size;
    int use_line_width;
    float line_width;
};

layout(set = SET_INDEX.UNIFORM, binding = UNIFORM_BINDING_INDEX.RENDER_CONFIG, std140) uniform RenderConfigBlock {
    RenderConfig u_render_config;
};

bool RenderConfig_useConstColor() {
    return u_render_config.color_type == COLOR_TYPE.CONST;
}

bool RenderConfig_useConstTransparent() {
    return u_render_config.transparent_type == COLOR_TYPE.CONST;
}

bool RenderConfig_usePointSize() {
    return u_render_config.use_point_size != 0;
}

bool RenderConfig_useLineWidth() {
    return u_render_config.use_line_width != 0;
}

vec4 RenderConfig_getConstColor() {
    return u_render_config.const_color;
}

float RenderConfig_getPointSize() {
    return u_render_config.point_size;
}

float RenderConfig_getLineWidth() {
    return u_render_config.line_width;
}

int RenderConfig_getModelType() {
    return u_render_config.model_type;
}

vec4 RenderConfig_getSpecularColor() {
	return vec4( 0.2, 0.2, 0.2, 1.0);
}

vec4 RenderConfig_getColor(vec4 attribute_color) {
    vec4 color;
    if (u_render_config.color_type == COLOR_TYPE.CONST) {
        color.rgb = u_render_config.const_color.rgb;
    } else {
        color.rgb = attribute_color.rgb;
    }
    if (u_render_config.transparent_type == COLOR_TYPE.CONST) {
        color.a = 1.0;
    } else {
        color.a = attribute_color.a;
    }
    return color;
}

vec4 RenderConfig_getPhongReflectionColor(vec2 vec_tex_coord, vec3 norm, vec4 color) {
    vec3 view_dir = getViewDir();
    LightColor light_color = calcBlinnPhongLight(norm, view_dir, 0.2);
    vec4 material_diffuse = color;
    
    return vec4(
        (light_color.ambient.rgb + light_color.diffuse.rgb) * material_diffuse.rgb +
        (light_color.specular.rgb * RenderConfig_getSpecularColor().rgb),
        material_diffuse.a);
}