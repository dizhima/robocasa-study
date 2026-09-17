# Agent Notes for the Study Fork

This repository is the project-specific RoboCasa fork used with the companion
system repository:

- System repo directory: `mujoco-skill-playground` (locate it in the current workspace)
- GitHub system repo: `https://github.com/dizhima/chi27robot`
- This fork: `https://github.com/dizhima/robocasa-study`

## Repository boundary

- This fork owns RoboCasa modifications, scene export and atomic replay tools,
  and source layout/object YAML under
  `robocasa/models/assets/scenes/custom_layouts/`.
- The system repo owns the frontend, backend, skill pipeline, handoff docs, and
  finalized runtime scenes and trajectories under `frontend/public/`.
- Source YAML lives here; a manually curated exported XML/MJB in the system repo
  is the authoritative runtime scene and should not be overwritten casually.

Run RoboCasa commands from this repository root with `uv run`. Run system
pipeline commands from the companion repository root.

## Cross-repository rules

- Keep committed object YAML portable. Use paths beginning with
  `robocasa/models/assets/objects/`; never commit `C:\Users\...` paths.
- Do not commit datasets, exported asset bundles, MJB files, or other large
  generated outputs here. Local `exported_scenes*` directories are ignored.
- When changing exporter inputs or output contracts, also check the companion
  repo's `docs/scene_export_pipeline.md` and dependent pipeline scripts.
- `origin` is the private study fork. `upstream` is the official RoboCasa repo
  and is fetch-only. Project work belongs on `study-scene-export`, not upstream
  `main`.
- Stop every process started for testing and do not leave preview or replay
  services running.
