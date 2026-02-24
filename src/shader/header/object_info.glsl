#include "common.glsl"

layout(set = SET_INDEX.STORAGE, binding = STORAGE_BINDING_INDEX.OBJECT_INFO, std140) uniform ObjectInfoBlock {
    ivec2 s_object_info[];
};

ivec2 getObjectInfo(int instance_id) {
	return s_object_info[instance_id];
}