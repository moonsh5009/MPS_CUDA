---
description: "Render domain architecture reference."
---

# Render Domain

## Architecture

```
RenderCore → Scene → RenderingEngine → RenderPass → Renderer → DrawCall
                  → RenderModelContainer → RenderModel (Mesh, AABB, FixedElement)
                  → RenderTarget (MSAA, ID)
                  → RenderUniform (RenderConfig, Camera, Light)
```

### Render Pipeline (5 stages)

```
Prefix → Opaque → Transparent → ID → Post
```

### Enum Types (src/MCore_interface/RenderTypeDef.h)

```cpp
enum class RenderPassType    { Prefix, Opaque, Transparent, ID, Post, Size };
enum class RenderModelType   { None, Mesh, AABB, FixedElement, Size };
enum class RenderTargetType  { MSAA, ID, Size };
enum class RenderUniformType { RENDER_CONFIG, CAMERA, LIGHT, Size };
```

### StreamBuffer System

Each `RenderModel` owns 3 StreamBuffers:

| StreamBuffer | Data Passing | Key Buffers |
|-------------|-------------|-------------|
| `TriangleStreamBuffer` | VBO | IBO + Position + Normal + Attribute + DrawIndexedIndirect |
| `LineStreamBuffer` | SSBO (bind group 1) | IBO + Position + VertexOffset + Attribute(opt) + DrawIndirect |
| `PointStreamBuffer` | SSBO (bind group 1) | IBO + Position + VertexOffset + Attribute(opt) + DrawIndirect |

## Key Reference Files

| File | Description |
|------|-------------|
| `src/MCore_interface/RenderTypeDef.h` | All render type enums |
| `src/MCore_interface/IRenderModel.h` | `IRenderModel` interface, StreamBuffer members |
| `src/MCore_interface/StreamBuffer.h` | StreamBuffer types |
| `src/MCore_render/RenderPassDef.h` | `DECLARE/IMPLEMENT_RENDER_PASS` macros |
| `src/MCore_render/RenderModelDef.h` | `DECLARE/IMPLEMENT_RENDER_MODEL` macros |
| `src/MCore_render/RendererBase.h/.cpp` | Renderer base class |
| `src/MCore_render/OpaqueRenderPass.h/.cpp` | RenderPass example |
| `src/MCore_render/OpaqueRenderer.h/.cpp` | Renderer example (pipeline construction) |
| `src/MCore_render/MeshModel.h/.cpp` | RenderModel example |
| `src/MCore_render/AABBModel.h/.cpp` | RenderModel example (specialized) |
| `src/MCore_render/RenderingEngine.cpp` | Pass execution order |
