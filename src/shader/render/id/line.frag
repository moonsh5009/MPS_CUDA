#include "../../header/render_config.glsl"

layout(location = 0) in flat uint in_instance_id;
layout(location = 1) in flat uint in_element_id;
layout(location = 2) in vec4 in_color;
layout(location = 3) in float in_thickness;
layout(location = 4) in float in_distance_to_center;

layout(location = 0) out uvec4 out_id;

void main() {
    if (in_distance_to_center > in_thickness * 0.5) {
        discard;
    }
    out_id = uvec4(RenderConfig_getModelType(), in_instance_id, in_element_id, 0);
}