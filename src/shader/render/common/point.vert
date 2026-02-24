#include "../../header/render_config.glsl"
#include "../../header/camera.glsl"
#include "../../header/vbo.glsl"

const vec2 s_vertex_to_clip_normal[6] = {
    vec2(-1.0, 1.0),
    vec2(-1.0, -1.0),
    vec2(1.0, 1.0),
    vec2(1.0, 1.0),
    vec2(-1.0, -1.0),
    vec2(1.0, -1.0)
};

layout (set = SET_INDEX.STREAM, binding = 0, std430) restrict readonly buffer VertexOffsets { uint vertex_offsets[]; };
layout (set = SET_INDEX.STREAM, binding = 1, std430) restrict readonly buffer Indices { uint indices[]; };
layout (set = SET_INDEX.STREAM, binding = 2, std430) restrict readonly buffer Positions { double[3] positions[]; };
layout (set = SET_INDEX.STREAM, binding = 3, std430) restrict readonly buffer Attributes { PointAttribute attributes[]; };

layout(location = 0) out flat uint out_instance_id;
layout(location = 1) out flat uint out_element_id;
layout(location = 2) out vec4 out_color;
layout(location = 3) out float out_size;
layout(location = 4) out vec2 out_clip_normal;

vec4 getColor(uint i) {
    if (RenderConfig_useConstColor() && RenderConfig_useConstTransparent()) {
        return RenderConfig_getConstColor();
    }
    return RenderConfig_getColor(unpackUnorm4x8(attributes[i].color));
}

float getPointSize(uint i) {
    if (RenderConfig_usePointSize()) {
        return RenderConfig_getPointSize();
    }
    return attributes[i].size;
}

void main() {
    uint point_index = gl_VertexIndex / 6;
    uint i = indices[point_index] + vertex_offsets[gl_InstanceIndex];

    uint vertex_index = gl_VertexIndex % 6;
    vec2 clip_normal = s_vertex_to_clip_normal[vertex_index];

    vec3 position = vec3(positions[i][0], positions[i][1], positions[i][2]);

    out_instance_id = gl_InstanceIndex;
    out_element_id = i;
    out_color = getColor(i);
    out_size = getPointSize(i) * 0.5;
    out_clip_normal = clip_normal;
    
    vec2 screen_size = vec2(getScreenSize());
    vec4 clip_pos = getProjMat() * getViewMat() * vec4(position, 1.0);
    vec3 clip_pos_xyz = clip_pos.xyz / clip_pos.w;
    clip_pos_xyz.z += 1.0 / 65536.0; // Push the point slightly forward to avoid clipping issues
    gl_Position = vec4(clip_pos_xyz, 1.0) + vec4(out_size * out_clip_normal * 2.0 / screen_size, 0.0, 0.0);
}