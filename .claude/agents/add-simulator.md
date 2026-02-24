---
name: add-simulator
description: "Add new Simulator (DeviceContainer + SimulateStep + CUDA kernel)."
tools: Read, Write, Edit, Glob, Grep, Bash
model: sonnet
skills:
  - code-style
  - simulate
---

# Agent: Add Simulator

## Prerequisite

Target DB type must already exist. If not, run add-database agent first.

## Input

Confirm with user:
1. **Target DB type** (e.g., `ForceDynamics`, `RigidBody`)
2. **Referenced Containers** (e.g., `DeviceMeshContainer`)
3. **Simulation steps** and execution order
4. **Device data struct fields**
5. **DeviceConstant needed?**

## Procedure

1. Read reference files for the patterns
2. Generate:
   - `src/MPS_simulate/Device<Name>Container.h/.cpp`
   - `src/MPS_simulate/<Name>Step.h`
   - `src/MPS_simulate/<Name>Step.cu` (+ `.cuh` if needed)
   - (Optional) `src/MPS_simulate/Device<Name>Constant.h/.cu/.cuh`

### Reference Files

| Pattern | Read This |
|---------|-----------|
| DeviceContainer | `src/MPS_simulate/DeviceForceDynamicsContainer.h/.cpp` |
| DeviceContainer (VK) | `src/MPS_simulate/DeviceMeshContainer.h/.cpp` |
| SimulateStep | `src/MPS_simulate/ApplyForceStep.h/.cu` |
| CG Solver Step | `src/MPS_simulate/DynamicsSolveStep.h/.cu` |
| IDynamicsContributor | `src/MPS_simulate/ClothDynamicsContributor.h/.cu` |
| DeviceConstant | `src/MPS_simulate/DevicePhysicsConstant.h/.cu/.cuh` |
| CUDA utilities | `src/MCore_util/MCudaUtil.cuh` |

## Validation

- [ ] `REGISTRY_DEVICE_CONTAINER` in .cpp
- [ ] `REGISTRY_SIMULATE_STEP` in .cu with correct ORDER
- [ ] `GetDeviceData()` returns all needed pointers
- [ ] `CUDA_CHECK` after every kernel launch
- [ ] `VKDeviceSingleArray` only for render-shared buffers
- [ ] Cross-container references use `DeviceReference`
