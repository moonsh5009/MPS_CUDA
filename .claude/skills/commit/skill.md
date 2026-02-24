---
description: "Project commit style guide. Invoke with /commit to generate convention-matching commits."
---

# Commit Guide

## Format

```
<scope>, <change summary>
```

Or for single topic: `<change summary>`

## Rules

1. **Korean** language (English technical terms kept as-is)
2. **Concise** - single line, max 72 chars
3. **Descriptive** - use forms like "~추가", "~수정", "~개선", "~처리", "~적용"
4. **No body** - title only

## Examples (from actual history)

```
bvh, render aabb 연결
streambuffer attribute optional 처리
force dynamics simulator
rendering engine 개편
vulkan buffer move 버그 수정
database 기반 개선
```

## Vocabulary

| Action | Expression |
|--------|-----------|
| New feature | `~추가`, `~& ~` |
| Modify | `~수정`, `~처리`, `~개편` |
| Bug fix | `~버그 수정` |
| Performance | `~개선`, `~적용`, `~refactoring` |
| Cleanup | `코드 정리` |
| Integration | `~연결`, `~기반 고도화` |

## Commit Unit

One logical change = One commit. Bundle related files (e.g., Data+Pool+Registry = 1 commit).

## Procedure

When `/commit` is invoked:
1. `git status` + `git diff` to analyze changes
2. `git log --oneline -5` to check recent style
3. Group by logical unit, draft message, confirm, execute
