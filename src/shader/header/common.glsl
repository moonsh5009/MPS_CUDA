#version 450 core
#extension GL_ARB_gpu_shader_fp64 : enable
#extension GL_ARB_shading_language_packing : require

struct ENUM_SET_INDEX {
    int UNIFORM;
    int STORAGE;
    int STREAM;
};
const ENUM_SET_INDEX SET_INDEX = ENUM_SET_INDEX(0, 1, 1);

struct ENUM_UNIFORM_BINDING_INDEX {
    int RENDER_CONFIG;
    int CAMERA;
    int LIGHT;
};
const ENUM_UNIFORM_BINDING_INDEX UNIFORM_BINDING_INDEX = ENUM_UNIFORM_BINDING_INDEX(0, 1, 2);

struct ENUM_STORAGE_BINDING_INDEX {
    int OBJECT_INFO;
};
const ENUM_STORAGE_BINDING_INDEX STORAGE_BINDING_INDEX = ENUM_STORAGE_BINDING_INDEX(0);