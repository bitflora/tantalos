# Validation

- Blender procedural build completed: 248 component meshes, 34,696 vertices,
  68,160 triangles, 16 mechanical bones.
- Actual mesh was rendered and visually inspected in front, back and three-quarter views.
- Editable Blender project and skinned GLB exported successfully.
- Runtime mesh and custom Portal 2 player animation sequences exported.
- Both Source models compiled successfully with Valve's Portal 2 compiler.
  The compiler splits the model into three meshes to honor its vertex limits;
  this is the expected vertex-count warning, rather than lost geometry.
- Final package: 18,359,665 bytes; all 14 VPK entries read back byte-for-byte.
  MDL, VVD and every VTX variant have matching component checksums. Both current
  `.vtx` and legacy `.dx90.vtx` index filenames are included.
- Texture: DXT1 with a full 12-level mip pyramid. Valve's own decoder reads it
  correctly. Mean absolute RGB errors against the source atlas are 2.14, 1.66,
  and 2.10 on a 0–255 scale.
- Final VPK was tested in a clean mod directory with no loose-model fallback.
  Console/VScript confirmed the actual player's model was
  `models/info_character/info_character_player.mdl`, the community-player path.
- Model faces the correct direction. All component colors and reference green
  paint patches render. Walking, crouch and jump inputs completed without a
  crash or detached parts.
- A named `weapon_bone` and a simple raised-arm pose keep the world portal gun
  at hand level rather than above the character's head.
- The actual player model was visible through linked portals. Test props used
  for precaching were removed before capturing the clean portal proof.
- `goblin-in-game.png` and `goblin-through-portal.png` were captured by the game,
  converted from its TGA screenshots to PNG, and visually inspected.

## Limits

- Mechanical locomotion and the low gun-holding pose are first-pass animations.
  Full directional blending, elaborate aiming/hand poses, and first-person
  goblin hands are not implemented.
- Camera height and collision hull use the existing game's settings.
- No distance LODs yet; the mesh has approximately 68k triangles.
- The community-player asset path was tested directly. A separate Workshop map,
  campaign Chell replacement, and co-op Atlas/P-body replacements were not tested.

This file is updated with observed results; untested behavior is not marked passed.
