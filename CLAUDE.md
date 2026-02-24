# MPS_CUDA - Physics Simulation Engine

C++ Vulkan/CUDA physics simulation engine. Simulates mesh-based physics (gravity, cloth, rigid body) on GPU and renders with Vulkan.

## Language Policy

- **Conversation**: Korean
- **Documentation files** (CLAUDE.md, skills, agents, etc. `.md`): English

## Tech Stack

- **C++20** (MSVC v143, Visual Studio 2022)
- **CUDA 12.6** (simulation kernels)
- **Vulkan** (vulkan.hpp C++ wrapper, dynamic rendering)
- **GLFW** (window/input), **GLM** (math), **Eigen** (linear algebra)

## Module Dependency Graph

```
App → MCore_util, MCore_system, MPS_database, MPS_simulate, MPS_system
MPS_database  → MCore_database
MPS_simulate  → MCore_simulate, MCore_util
MPS_system    → MCore_system, MPS_simulate
MCore_system  → MCore_render, MCore_simulate, MCore_database
MCore_render  → MCore_util
MCore_database → MCore_interface
MCore_interface → MCore_util (INTERFACE, header-only)
MCore_util    → Vulkan, CUDA, GLM, Eigen, GLFW
```

## Directory Structure

```
src/
  App/                  # Entry point (Main.cpp), DLL loading, window loop
  MCore_interface/      # Pure interfaces (IDBData, IRenderModel, IDeviceContainer, ISimulateStep, etc.)
  MCore_util/           # Vulkan/CUDA wrappers (DeviceBuffer, ShaderModule, Pipeline, etc.)
  MCore_database/       # DB framework (Data, Pool, Session, Transaction, PoolFactory)
  MCore_render/         # Render framework (RenderPass, Renderer, RenderModel, Camera)
  MCore_simulate/       # Simulate framework (DeviceContainer, SimulateStep, SimulateManager)
  MCore_system/         # System framework (System, SimulateToRenderConverter)
  MPS_database/         # Impl: MeshData/Pool, KineticData/Pool, ClothData/Pool, ForceDynamics, etc.
  MPS_simulate/         # Impl: DeviceMeshContainer, ApplyForceStep, CUDA kernels (.cu)
  MPS_system/           # Impl: PhysicsSimulateManager, MeshToRenderConverter, etc.
  shader/               # GLSL shaders (header/, render/common/, render/id/)
```

## Build System

- **CMake 3.28+** with `CMakePresets.json`
- Configure: `cmake --preset msvc-x64`
- Build: `cmake --build build --config Debug`
- Each module builds as a **DLL** (`BUILD_MODULE_NAME` preprocessor define)
- DLL export pattern: `HeaderPre.h` / `__MY_EXT_CLASS__` / `HeaderPost.h`
- New source files auto-detected via `file(GLOB)`

## Key Types

| Type | Definition | Description |
|------|-----------|-------------|
| `DBKey` | `uint32_t` | DB object identifier |
| `DBTypeID` | `uint32_t` | DB type ID (MESH=10, KINETIC=12, etc.) |
| `IndexType` | `uint32_t` | Index (GPU compatible) |
| `mcore::Vector3` | `Eigen::Vector3d` | 3D vector (double) |
| `REAL` | `double` | Simulation real number type |

## Skills (`.claude/skills/`)

Domain-specific references (architecture + key files):
- `/code-style` - Naming conventions, formatting
- `/database` - DB domain architecture
- `/render` - Render domain architecture
- `/simulate` - Simulate domain architecture
- `/system` - System domain architecture
- `/commit` - Commit message conventions
- `/sync` - Documentation sync procedure

## Agents (`.claude/agents/`)

Isolated subagents for code generation:
- `add-database` - Add new DB type (Data + Pool + Registry)
- `add-simulator` - Add new Simulator (DeviceContainer + SimulateStep + CUDA kernel)
- `add-renderer` - Add new Render component (Model, Pass, Renderer, Converter)
- `build-fix` - Build and auto-fix compile/runtime errors
