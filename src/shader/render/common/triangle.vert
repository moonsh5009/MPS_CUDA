#include "../../header/render_config.glsl"
#include "../../header/camera.glsl"

layout(location = 0) in dvec3 v_pos;
layout(location = 2) in dvec3 v_normal;
layout(location = 4) in vec4 v_color;
layout(location = 5) in vec2 v_tex_coord;

layout(location = 0) out flat uint out_instance_id;
layout(location = 1) out flat uint out_element_id;
layout(location = 2) out vec4 out_color;
layout(location = 3) out vec3 out_normal;

void main() {
    out_instance_id = gl_InstanceIndex;
    out_element_id = gl_VertexIndex;
    out_color = RenderConfig_getColor(v_color);
    out_normal = vec3(v_normal);
    
    gl_Position = getProjMat() * getViewMat() * vec4(v_pos, 1.0);
}