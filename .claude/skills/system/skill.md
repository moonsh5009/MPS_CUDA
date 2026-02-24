---
description: "System domain architecture reference."
---

# System Domain

## Architecture

```
App (Main.cpp)
  └─ System (MCore_system)
       ├─ IDBSession     ← database::Session
       ├─ IRenderCore    ← render::RenderCore
       ├─ ISimulateManager ← PhysicsSimulateManager (injected)
       └─ ISystemController ← SystemController
```

### Manager Injection

Inject via `ISystem::SetSimulateManager()` **before** `Initialize()`. If not injected, default `SimulateManager` is created.

### Step Ordering

- Lower order runs first (100 → 200 → 300)
- Same order → concurrent on separate CUDA streams
- Maintain: Force (100) → Solver (200) → Integrate (300)
- Auto-registered via `REGISTRY_SIMULATE_STEP(StepClass, ORDER)`
- Cross-step discovery: `PostInitialize()` + `GetManager()->FindStep<T>()`

## Key Reference Files

| File | Description |
|------|-------------|
| `src/MCore_interface/ISystem.h` | System interface, `SetSimulateManager()` |
| `src/MCore_interface/ISimulateManager.h` | Manager interface |
| `src/MCore_system/System.h/.cpp` | System implementation |
| `src/MCore_simulate/SimulateManager.h/.cpp` | Default manager |
| `src/MPS_system/PhysicsSimulateManager.h/.cpp` | Custom manager |
| `src/App/Main.cpp` | Manager injection point |
