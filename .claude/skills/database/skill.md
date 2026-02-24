---
description: "DB domain architecture reference."
---

# Database Domain

## Architecture

```
Data (value object) → Pool (hash storage) → Session (transaction) → PoolFactory (DLL auto-register)
```

- **Data**: Serializable value object (`DATABASE_FIELD` macro for auto Clone/Stream)
- **Pool**: Hash-based storage (`Pool<T>` multi-instance / `SinglePool<T>` singleton)
- **Session**: Transaction management (Undo/Redo)
- **PoolFactory**: Auto-registration on DLL load (IMPLEMENT macros)
- **database::Ref<T>**: Weak reference to other data types (DBKey-based)

## DB Type ID Registry

```
// src/MPS_database/DBRegistery.h
REGISTRY_SINGLE_DATABASE(PHYSICS, 1)
REGISTRY_DATABASE(MESH, 10, 10000)
REGISTRY_DATABASE(KINETIC, 12, 10000)
REGISTRY_DATABASE(CLOTH, 20, 10000)
REGISTRY_DATABASE(FORCE_DYNAMICS, 30, 10000)
```

ID allocation rule: Assign with gaps between groups (1, 10, 12, 20, 30, ...)

## Key Reference Files

| File | Description |
|------|-------------|
| `src/MCore_database/Data.h` | `DATABASE_FIELD` macro, `Data` base |
| `src/MCore_database/DBMetaDataDef.h` | `REGISTRY_DATABASE` / `REGISTRY_SINGLE_DATABASE` macros |
| `src/MCore_database/Pool.h` | `Pool<T>`, `DECLARE/IMPLEMENT_DATAPOOL` |
| `src/MCore_database/SinglePool.h` | `SinglePool<T>`, `DECLARE/IMPLEMENT_SINGLE_DATAPOOL` |
| `src/MCore_database/PoolFactory.h` | Factory singleton |
| `src/MCore_database/Ref.h` | `database::Ref<T>` reference type |
| `src/MPS_database/DBRegistery.h` | All DB type ID registrations |
| `src/MPS_database/MeshData.h` | Data example (general) |
| `src/MPS_database/MeshPool.h/.cpp` | Pool example |
| `src/MPS_database/ForceDynamics.h` | Ref pattern example |
| `src/MPS_database/PhysicsPool.h/.cpp` | SinglePool example |
