---
name: add-database
description: "Add new Database type (Data + Pool + Registry)."
tools: Read, Write, Edit, Glob, Grep, Bash
model: sonnet
skills:
  - code-style
  - database
---

# Agent: Add Database Type

## Input

Confirm with user:
1. **Type name** (e.g., `RigidBody`, `Constraint`)
2. **Field list** (name, type)
3. **Pool kind**: Multi-instance (`Pool`) vs Singleton (`SinglePool`)
4. **Ref usage** (`database::Ref<T>` to other types)

## Procedure

1. Read `src/MPS_database/DBRegistery.h` — allocate non-colliding ID with gaps
2. Read reference files for the patterns, then generate:
   - `src/MPS_database/<Name>Data.h` (or `<Name>.h` for Ref-only)
   - `src/MPS_database/<Name>Pool.h`
   - `src/MPS_database/<Name>Pool.cpp`
3. Edit `src/MPS_database/DBRegistery.h` to add the new registration

### Reference Files

| Pattern | Read This |
|---------|-----------|
| Data (general) | `src/MPS_database/MeshData.h` |
| Data (Ref) | `src/MPS_database/ForceDynamics.h` |
| Pool (multi) | `src/MPS_database/MeshPool.h/.cpp` |
| Pool (singleton) | `src/MPS_database/PhysicsPool.h/.cpp` |

## Validation

- [ ] No ID collision in DBRegistery.h
- [ ] DATABASE_FIELD field list matches actual member variables
- [ ] HeaderPre.h / HeaderPost.h pair correct
- [ ] Pool constructor correctly calls parent initializer list
- [ ] If using Ref, target Data's header is included
