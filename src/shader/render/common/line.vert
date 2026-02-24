#include "../../header/render_config.glsl"
#include "../../header/camera.glsl"
#include "../../header/vbo.glsl"

const uint VERTEX_TO_LINE_VERTEX[6] = { 0, 0, 1, 1, 0, 1 };
const float VERTEX_TO_SIDE_FACTOR[6] = { -1., 1., -1., -1., 1., 1. };
const float LINE_BIAS = 0.5 / 65536.0;
const float MIN_NEAR = 1.0e-2;

layout (set = SET_INDEX.STREAM, binding = 0, std430) restrict readonly buffer VertexOffsets { uint vertex_offsets[]; };
layout (set = SET_INDEX.STREAM, binding = 1, std430) restrict readonly buffer Indices { uint indices[]; };
layout (set = SET_INDEX.STREAM, binding = 2, std430) restrict readonly buffer Positions { double[3] positions[]; };
layout (set = SET_INDEX.STREAM, binding = 3, std430) restrict readonly buffer Attributes { LineAttribute attributes[]; };

layout(location = 0) out flat uint out_instance_id;
layout(location = 1) out flat uint out_element_id;
layout(location = 2) out vec4 out_color;
layout(location = 3) out float out_thickness;
layout(location = 4) out float out_distance_to_center;

vec4 getColor(uint i) {
    if (RenderConfig_useConstColor() && RenderConfig_useConstTransparent()) {
        return RenderConfig_getConstColor();
    }
    return RenderConfig_getColor(unpackUnorm4x8(attributes[i].color));
}

float getLineWidth(uint i) {
    if (RenderConfig_useLineWidth()) {
        return RenderConfig_getLineWidth();
    }
    return attributes[i].width;
}

vec2 clipToViewZ(vec3 view_pos0, vec3 view_pos1) {
    if (!isPerspective()) {
        return vec2(0.0, 1.0);
    }

    float z_near = -getFrustum().x;
    if (view_pos0.z > z_near && view_pos1.z > z_near) {
        return vec2(2.0, -1.0);
    }
    if (view_pos0.z <= z_near && view_pos1.z <= z_near) {
        return vec2(0.0, 1.0);
    }
    
    float t = (z_near - view_pos0.z) / (view_pos1.z - view_pos0.z);
    if (view_pos0.z > z_near) {
        return vec2(clamp(t, 0.0, 1.0), 1.0);
    }
    return vec2(0.0, clamp(t, 0.0, 1.0));
}

void main() {
    uint vertex_offset = vertex_offsets[gl_InstanceIndex];
    uint line_index = gl_VertexIndex / 6;
    uint line_index_2 = line_index * 2;
    uint i0 = indices[line_index_2 + 0] + vertex_offset;
    uint i1 = indices[line_index_2 + 1] + vertex_offset;
    if (i0 == i1) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }
    
    uint vertex_index = gl_VertexIndex % 6;
    uint line_vertex_index = VERTEX_TO_LINE_VERTEX[vertex_index];
    float line_side_factor = VERTEX_TO_SIDE_FACTOR[vertex_index];

    vec3 position_0 = vec3(positions[i0][0], positions[i0][1], positions[i0][2]);
    vec3 position_1 = vec3(positions[i1][0], positions[i1][1], positions[i1][2]);

    vec4 view_x0 = getViewMat() * vec4(position_0, 1.0);
    vec4 view_x1 = getViewMat() * vec4(position_1, 1.0);
    vec2 depth_range = clipToViewZ(view_x0.xyz, view_x1.xyz);
    if (depth_range.x > depth_range.y) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }
    
    vec3 clipped_view_pos0 = mix(view_x0.xyz, view_x1.xyz, depth_range.x);
    vec3 clipped_view_pos1 = mix(view_x0.xyz, view_x1.xyz, depth_range.y);
    vec4 clip_x0 = getProjMat() * vec4(clipped_view_pos0, 1.0);
    vec4 clip_x1 = getProjMat() * vec4(clipped_view_pos1, 1.0);

    float inv_clip_x0_w = 1.0 / clip_x0.w;
    float inv_clip_x1_w = 1.0 / clip_x1.w;
    vec3 clip_x0_xyz = clip_x0.xyz * inv_clip_x0_w;
    vec3 clip_x1_xyz = clip_x1.xyz * inv_clip_x1_w;

    vec2 screen_size = vec2(getScreenSize());
    vec2 line_dir_screen = normalize((clip_x1_xyz.xy - clip_x0_xyz.xy) * screen_size);
    vec2 clip_normal = vec2(-line_dir_screen.y, line_dir_screen.x) * 2.0 / screen_size;

    out_instance_id = gl_InstanceIndex;
    out_element_id = line_index;

    vec3 clipped_pos;
    float inv_w;
    float line_distance;
    if (line_vertex_index == 0) {
        out_color = getColor(i0);
        out_thickness = getLineWidth(i0) * 0.5;
        clipped_pos = clip_x0_xyz;
        inv_w = inv_clip_x0_w;
    } else {
        out_color = getColor(i1);
        out_thickness = getLineWidth(i1) * 0.5;
        clipped_pos = clip_x1_xyz;
        inv_w = inv_clip_x1_w;
    }
    clipped_pos.z += LINE_BIAS;

    float dist = out_thickness * line_side_factor;
    out_distance_to_center = dist;
    gl_Position = vec4(clipped_pos, 1.0) + vec4(clip_normal * dist, 0.0, 0.0);
}