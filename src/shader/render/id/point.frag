#include "../../header/render_config.glsl"

layout(location = 0) in flat uint in_instance_id;
layout(location = 1) in flat uint in_element_id;
layout(location = 2) in vec4 in_color;
layout(location = 3) in float in_size;
layout(location = 4) in vec2 in_clip_normal;

layout(location = 0) out uvec4 out_id;

void main() {
    float z_sq = in_clip_normal.x * in_clip_normal.x + in_clip_normal.y * in_clip_normal.y;
    if (z_sq > 1.0) {
        discard;
    }

    out_id = uvec4(RenderConfig_getModelType(), in_instance_id, in_element_id, 0);
}