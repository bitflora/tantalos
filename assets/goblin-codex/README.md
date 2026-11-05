# Goblin — Codex model

A real, articulated 3D interpretation of the supplied **goblin bot 2.webp** reference.
The unseen back is an original continuation of the reference's mechanical design.

## Deliverables

- `goblin.blend`: editable meshes, UVs, 16-bone mechanical rig, walk action, materials, studio lights and camera.
- `goblin.glb`: portable skinned model with its walk preview animation and embedded texture.
- `goblin_atlas.png`: deterministic 2048 × 2048 painted-metal texture atlas.
- `goblin-portrait.png`, `goblin-front.png`, `goblin-back.png`: renders of the actual mesh.
- `goblin-in-game.png`, `goblin-through-portal.png`: screenshots captured by Portal 2 itself.
- `goblin-codex.vpk`: compiled Portal 2 models and material. Replaces the community-chamber Bendy model when mounted; also contains a separately named preview model.
- `source/`: generated SMD meshes, animations, material definition and QC build instructions.
- `package-manifest.json`: file inventory and verification status.

## Build

From this worktree, with Python + Pillow + NumPy and Blender 4.5:

```powershell
python tools/goblin-codex/make_atlas.py
blender --background --threads 4 --python tools/goblin-codex/build_model.py
blender --background --threads 4 --python tools/goblin-codex/export_player.py
python tools/goblin-codex/package_mod.py
```

The last command requires **Portal 2 Authoring Tools**, Steam app 629. The compiler
and installed game's content are read from the default Steam location; use
`package_mod.py --game PATH` for a different installation.

The model build writes to `out/goblin-codex/game` in this worktree. The test launcher
uses `out/goblin-codex/packed-game`, which contains the deliverable as
`pak01_dir.vpk`, the inspection map, and its preview config. It mounts the installed
Portal 2 content beneath that package, with no loose-model fallback. No original game
archive or configuration is overwritten. Keep this mod separate from other
player-replacement mods: both would override the same community-player path.

## Run the isolated mod

Run `tools/goblin-codex/launch_goblin.ps1` from PowerShell. It opens the inspection
room with Goblin selected as the player, enables a third-person preview, and
refuses to interrupt an existing Portal 2 session. To rebuild that room:

```powershell
python tools/goblin-codex/build_showroom.py
```

The room's authoritative VMF is generated in this worktree. Valve's map utilities
require an SDK scratch directory, so they compile a copy in the separately named
`Portal 2/sdk_content/maps/goblin_codex_worktree` folder; the final BSP is copied
back into this worktree's isolated mod. No existing map is replaced.

For a manual launch, pass `-game` followed by this worktree's absolute
`out/goblin-codex/packed-game` directory to Portal 2. This is a local test mod.

For a deliberate third-person preview, load a chamber and execute
`goblin_codex_preview.cfg` in the developer console. It precaches the separately
named model, selects it for the player, and enables third person. It enables
`sv_cheats` for this preview session.

The replacement provides Portal 2's `portalgun_*` and `nogun_*` sequence names with
custom idle, walk, jump and crouch motions, plus an additive head aim matrix.
Standing and crouched movement blend eight directional steps with a centered idle.
The walk cycle alternates planted feet and lifted steps with a slight body bob. The
Source rig adds a seventeenth helper bone for the game's world portal gun. The
first-person portal gun and hands use the game's existing viewmodel. Model height
is approximately 69 Source units; the game's player collision hull and camera
remain controlled by Portal 2.

## Validation

The package contains both `.vtx` and `.dx90.vtx` model index names for current and
older Portal 2 builds. The package builder checks matching model-component
checksums and compiler return status, decodes the compressed texture
with Valve's tool, compares it with the source atlas, and roundtrips every VPK entry
byte-for-byte through a separate reader. See `VALIDATION.md` for actual in-game
checks and remaining limitations; a successful asset build alone does not prove
that every gameplay animation behaves correctly.

## Design and scope

The model includes a fused, sculpted face, recessed asymmetric optics, seven
separate purple lens shards, articulated ear linkages, collar hazard stripes,
wind-up key, exposed chest pistons and wiring, side turbines, back cooling panel,
joint bearings, claws and split feet. All geometry and texture generation are
reproducible from `tools/goblin-codex/`.

No Valve model geometry or animations are distributed in the package. The game
provides its own base content when the mod runs. The supplied reference remains
the user's design reference; it is not embedded as a fake 3D surface.
