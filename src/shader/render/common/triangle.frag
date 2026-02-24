#include "../../header/render_config.glsl"

layout(location = 0) in flat uint in_instance_id;
layout(location = 1) in flat uint in_element_id;
layout(location = 2) in vec4 in_color;
layout(location = 3) in vec3 in_normal;

layout(location = 0) out vec4 out_color;

void main() {
    out_color = RenderConfig_getPhongReflectionColor(vec2(0.0, 0.0), in_normal, in_color);
}