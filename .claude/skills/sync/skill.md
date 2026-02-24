---
description: "Sync & optimize CLAUDE.md, skills, agents based on session work."
user-invocable: true
argument-hint: "[focus area: all|skills|agents|claude]"
---

# Sync: Documentation Sync & Optimization

Analyze work performed in the current session, sync project documentation with the codebase.

## Procedure

### Phase 1: Collect Current State

1. `git diff --stat HEAD~5` and `git log --oneline -10` for recent changes
2. `git diff --cached --stat` and `git status` for uncommitted changes
3. Identify new/modified/deleted patterns (DB types, Simulators, RenderModels, enums, etc.)

### Phase 2: Analyze Documentation Drift

Read each document and identify discrepancies with codebase:
- **CLAUDE.md**: Dependency graph, directory structure, key types, skills/agents lists
- **Skills**: Architecture diagrams, enum values, key reference file tables
- **Agents**: Referenced files exist? Validation checklists complete?

### Phase 3: Execute Updates

Principles:
1. Facts only — reflect what's confirmed in code
2. Concise — remove duplicates
3. Real examples — use actual filenames, not hypothetical
4. Accurate paths — remove references to non-existent files
5. Current values — update hardcoded enum/ID values

### Phase 4: Verify

1. Glob to confirm all referenced file paths exist
2. Grep to confirm hardcoded enum/ID values match actual code

### Phase 5: Report

Summarize: updated files, key changes, unresolved discrepancies.
