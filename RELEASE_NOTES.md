# GhostCut V1 Stable

Frozen production baseline.

## Frozen components
- GhostCut AI 1.0.3
- Agent 1 1.3.0
- Agent 2 1.0.20 (frozen v1.0.19 compact planner core + Editorial Vision adapter)
- Agent 3 1.0.0

## Freeze policy
Do not change the V1 core unless a reproducible production regression is found. New experiments should happen on a new branch/version.

## Intentionally excluded
Source footage, voiceovers/project scripts, GhostCutProjects, indexes/databases, cached frames/embeddings, Ollama weights, virtual environments, `.env`, logs, and patch backups.
