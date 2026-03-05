# AGENTS.md

## Scope

This repository is a Vivid package repo. Keep it focused on package-owned operators, graphs, and tests.

## Package layout

- `src/`: operator source files compiled into package plugins
- `graphs/`: package demo graphs for smoke coverage
- `tests/`: package-owned tests (manifest + behavior tests)
- `vivid-package.json`: package manifest
- `CMakeLists.txt`: package build definition

## Rules

- Do not re-introduce moved operators back into `vivid-core`.
- Keep package tests runnable from package CI.
- Keep graph smoke ownership in this package repo, not in `vivid-core`.
- Prefer small, focused commits: operator move, build wiring, tests/graphs, docs.

## Validation

Before pushing changes:

1. Configure + build package operators.
2. Run package tests.
3. Run `vivid` link/rebuild/uninstall cycle against this package.
4. Run `test_demo_graphs` against this package's `graphs/` directory.
