---
name: add-renderer
description: "Add new Render component (RenderModel, RenderPass, Renderer, Converter)."
tools: Read, Write, Edit, Glob, Grep, Bash
model: sonnet
skills:
  - code-style
  - render
---

# Agent: Add Renderer

## Input

Confirm with user:
1. **Component type**: (A) Model only, (B) Pass + Renderer, (C) Full, (D) Converter only
2. **Name**
3. **StreamBuffer types** (Triangle, Line, Point)
4. **(Converter)** Target DeviceContainer type

## Procedure

Read reference files for the patterns, then generate:

### (A) RenderModel
- Add enum in `src/MCore_interface/RenderTypeDef.h`
- Generate `src/MCore_render/<Name>Model.h/.cpp`

### (B) RenderPass + Renderer
- Add enum in `src/MCore_interface/RenderTypeDef.h` (if new pass type)
- Generate `src/MCore_render/<Name>Renderer.h/.cpp`
- Generate `src/MCore_render/<Name>RenderPass.h/.cpp`
- Add to draw sequence in `src/MCore_render/RenderingEngine.cpp`

### (C) Shaders (if needed)
- Generate `src/shader/render/<pass_name>/` (.vert/.frag files)

### (D) SimulateToRenderConverter
- Generate `src/MPS_system/<Name>ToRenderConverter.h/.cpp`

### Reference Files

| Pattern | Read This |
|---------|-----------|
| RenderModel | `src/MCore_render/MeshModel.h/.cpp` |
| RenderModel (specialized) | `src/MCore_render/AABBModel.h/.cpp` |
| RenderPass | `src/MCore_render/OpaqueRenderPass.h/.cpp` |
| Renderer | `src/MCore_render/OpaqueRenderer.h/.cpp` |
| ID Pass | `src/MCore_render/IDRenderPass.h/.cpp` |
| Converter | `src/MPS_system/MeshToRenderConverter.h/.cpp` |
| Shader examples | `src/shader/render/common/`, `src/shader/render/id/` |

## Validation

- [ ] Enum added before `Size`
- [ ] DECLARE/IMPLEMENT macro pairs correct
- [ ] HeaderPre.h / HeaderPost.h pairs correct
- [ ] Converter sets all pointers to nullptr for empty data
- [ ] New Pass placed correctly in RenderingEngine draw order
