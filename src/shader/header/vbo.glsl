#include "common.glsl"

struct ENUM_POINT_TYPE {
    uint DEFAULT;
    uint CIRCULAR;
    uint CIRCLE_OUTLINE;
    uint CIRCLE_FILLED;
    uint ARC_OUTLINE;
    uint ARC_FILLED;
};
const ENUM_POINT_TYPE POINT_TYPE = ENUM_POINT_TYPE(0, 1, 2, 3, 4, 5);

struct PointAttribute {
    uint color;
    float size;
};

struct LineAttribute {
    uint color;
    float width;
};