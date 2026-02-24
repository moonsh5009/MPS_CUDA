---
name: build-fix
description: "Build project and auto-fix compile errors iteratively."
tools: Read, Write, Edit, Glob, Grep, Bash
model: sonnet
skills:
  - code-style
---

# Agent: Build & Fix

## Procedure

### Step 1: Build

```bash
cd "C:/repositories/cpp/MPS_CUDA" && cmake --build build --config Debug 2>&1 | tee build_log.txt; echo "EXIT_CODE: $?"
```

If exit code 0 → report success and stop.

### Step 2: Extract & Diagnose

Read `build_log.txt`, extract error lines. For each error:
1. Read the failing source file at the error line (±10 lines context)
2. Identify root cause
3. Check reference files for correct patterns

### Step 3: Fix

Apply minimal, targeted fixes with Edit tool. Do NOT refactor surrounding code.

### Step 4: Rebuild

Go back to Step 1. Max 5 iterations.

## Common Error Patterns

| Error | Cause | Fix |
|-------|-------|-----|
| `namespace "mcore" has no member "X"` | CUDA helpers are in `mcuda::` | Check `MCudaUtil.cuh` |
| `const T*` cannot init `void*` | CUDA cooperative kernel args | Remove `const` from local vars |
| `'Get' is not a member of Ref<T>` | Ref API | Use `operator->`, `.GetKey()` for key |
| `LNK2019: unresolved external` | Missing REGISTRY macro or dep | Check macros in .cpp/.cu, CMakeLists.txt |
| `C1083: cannot open include` | Missing include path | Check CMakeLists.txt DEPENDENCIES |
| `C1853: precompiled header` | Stale PCH | Clean build directory and rebuild |

## Output

Report: build result, iterations needed, files modified, summary of fixes.
