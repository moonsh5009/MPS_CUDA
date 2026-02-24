---
description: "Project coding conventions. Reference when generating or modifying code."
---

# Code Style Guide

## Naming Convention

| Target | Rule | Example |
|--------|------|---------|
| Class/Struct | `PascalCase` | `CameraTransform`, `MeshPool` |
| Interface | `I` prefix | `IDBPool`, `IRenderPass` |
| Function/Method | `PascalCase` | `Initialize()`, `GetTransform()` |
| Member variable | `m_` prefix | `m_transform`, `m_keys` |
| Pointer member | `m_p` prefix | `m_pModelContainer` |
| Static member | `s_` prefix | `s_instance` |
| Constant/Macro | `UPPER_SNAKE_CASE` | `BLOCK_SIZE` |
| Enum value | `PascalCase` | `RenderPassType::Opaque` |
| Namespace | `lowercase` | `mcore::render`, `mvk`, `mcuda` |
| Filename | `PascalCase` | `MeshPool.h`, `OpaqueRenderPass.cpp` |
| Def file | `*Def.h` suffix | `RenderTypeDef.h`, `CameraDef.h` |
| Kernel function | `kernel_` prefix | `kernel_InitForce` |

## Namespace Hierarchy

```
mcore                    // Top-level (interfaces, common types)
mcore::database          // DB framework
mcore::render            // Render framework
mcore::simulate          // Simulate framework
mcore::system            // System framework
mcore::util              // Utilities
mvk                      // Vulkan wrappers
mvk::util                // Vulkan utilities
mvk::vbo                 // Vertex buffer attributes
mcuda                    // CUDA wrappers
```

## Formatting

- **Indentation**: Tab
- **Braces**: Allman style (open brace on new line)
- **Header guard**: `#pragma once`
- **DLL export**: `HeaderPre.h` / `__MY_EXT_CLASS__` / `HeaderPost.h`
