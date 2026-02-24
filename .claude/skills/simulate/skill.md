---
description: "Simulate domain architecture reference."
---

# Simulate Domain

## Architecture

```
SimulateManager (MCore_simulate, Factory build)
  → DeviceContainer (DB data → GPU memory)
  → SimulateStep (order-based, CUDA stream concurrent execution)
  → DeviceConstant (constant memory)
                                 ↓ (system module)
  SimulateToRenderConverter → RenderModel StreamBuffer binding
```

### Step Execution Pipeline

Steps with the **same order** run concurrently on separate CUDA streams. Different order groups run sequentially with sync barriers.

```
┌─────────────────────────────────────┐
│ Order 100: [ApplyForceStep]         │ stream 0
├─────────────── sync ────────────────┤
│ Order 200: [DynamicsSolveStep,      │ stream 0, stream 1
│             ClothDynamicsContributor]│
├─────────────── sync ────────────────┤
│ Order 300: [IntegratePositionStep]  │ stream 0
└─────────────────────────────────────┘
```

### Data Flow

```
DB change → DeviceContainer.OnAddDB/OnModifyDB/OnDeleteDB → GPU buffer update
  → SimulateStep.OnUpdatedContainer() → SimulateStep.Execute(stream, dt)
    → CUDA kernel → SimulateToRenderConverter.OnConvert() → StreamBuffer binding
```

### Device Memory Types

| Type | Description |
|------|-------------|
| `mcuda::DeviceBuffer<T>` | CUDA-only device memory |
| `mcuda::VKDeviceBuffer<T>` | Vulkan interop device memory |
| `mcuda::DeviceSingleArray<T>` | Dynamic array (CUDA) |
| `mcuda::VKDeviceSingleArray<T>` | Dynamic array (Vulkan interop) |
| `mcuda::HostSingleArray<T>` | Host mirror array |

Rule: Buffers used in rendering → `VK*` types; simulation-only → regular types.

## Key Reference Files

| File | Description |
|------|-------------|
| `src/MCore_simulate/DeviceContainer.h` | Base template + `REGISTRY_DEVICE_CONTAINER` |
| `src/MCore_simulate/SimulateStep.h` | Base template + `REGISTRY_SIMULATE_STEP` |
| `src/MCore_simulate/SimulateStepFactory.h` | Step factory (order-based grouping) |
| `src/MCore_simulate/DeviceConstant.h` | Constant memory + `REGISTRY_DEVICE_CONSTANT` |
| `src/MCore_simulate/DeviceReference.h` | Cross-container reference |
| `src/MCore_system/SimulateToRenderConverter.h` | Converter base + `REGISTRY_SIMULATE_TO_RENDER` |
| `src/MCore_util/MCudaUtil.cuh` | `kernel_Loop`, CUDA utilities |
| `src/MCore_util/DeviceBuffer.h` | `DeviceBuffer<T>` template |
| `src/MCore_util/VKDeviceBuffer.h` | `VKDeviceBuffer<T>` Vulkan interop |
| `src/MCore_util/DeviceSingleArray.h` | `DeviceSingleArray<T>`, `VKDeviceSingleArray<T>` |
| `src/MPS_simulate/DeviceForceDynamicsContainer.h` | Container example |
| `src/MPS_simulate/DeviceMeshContainer.h` | Container example (VK interop) |
| `src/MPS_simulate/ApplyForceStep.h/.cu` | SimulateStep example |
| `src/MPS_simulate/DynamicsSolveStep.h/.cu` | CG solver example |
| `src/MPS_simulate/ClothDynamicsContributor.h/.cu` | IDynamicsContributor example |
| `src/MPS_simulate/DevicePhysicsConstant.h/.cu/.cuh` | DeviceConstant example |
| `src/MPS_system/MeshToRenderConverter.h/.cpp` | Converter example |
