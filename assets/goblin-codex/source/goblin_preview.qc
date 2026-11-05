// Goblin: custom proportions and mechanical animation, Portal 2 player sequence names.
$modelname "goblin_codex/goblin_preview.mdl"
$body "goblin" "player_reference.smd"
$surfaceprop "metal"
$cdmaterials "models/goblin_codex/"
$eyeposition 0 0 54
$illumposition 0 0 36
$bbox -16 -16 0 16 16 72
$cbox -16 -16 0 16 16 72
$poseparameter move_x -1 1
$poseparameter move_y -1 1
$poseparameter body_pitch -90 90
$poseparameter body_yaw -90 90
$poseparameter aim_pitch -90 90
$poseparameter aim_yaw -90 90
$attachment "eyes" "head" 0 12 0 rotate 0 0 0
$attachment "forward" "head" 0 12 0 rotate 0 0 0
$bonemerge "weapon_bone"
$attachment "anim_attachment_RH" "weapon_bone" 0 0 0 rotate 0 0 0
$attachment "anim_attachment_LH" "hand.R" 0 0 0 rotate 0 0 0
$attachment "weapon_bone" "weapon_bone" 0 0 0 rotate 0 0 0
$attachment "portalgun" "weapon_bone" 0 0 0 rotate 0 0 0
$sequence "ref" "player_idle.smd" fps 24
$sequence "idle_raw" "player_idle.smd" fps 24
$animation "run_nw" "player_run_nw.smd" fps 30 LX LY loop
$animation "run_n" "player_run_n.smd" fps 30 LX LY loop
$animation "run_ne" "player_run_ne.smd" fps 30 LX LY loop
$animation "run_w" "player_run_w.smd" fps 30 LX LY loop
$animation "run_center" "player_idle.smd" fps 30
$animation "run_e" "player_run_e.smd" fps 30 LX LY loop
$animation "run_sw" "player_run_sw.smd" fps 30 LX LY loop
$animation "run_s" "player_run_s.smd" fps 30 LX LY loop
$animation "run_se" "player_run_se.smd" fps 30 LX LY loop
$animation "crouchwalk_nw" "player_crouchwalk_nw.smd" fps 30 LX LY loop
$animation "crouchwalk_n" "player_crouchwalk_n.smd" fps 30 LX LY loop
$animation "crouchwalk_ne" "player_crouchwalk_ne.smd" fps 30 LX LY loop
$animation "crouchwalk_w" "player_crouchwalk_w.smd" fps 30 LX LY loop
$animation "crouchwalk_center" "player_crouch.smd" fps 30
$animation "crouchwalk_e" "player_crouchwalk_e.smd" fps 30 LX LY loop
$animation "crouchwalk_sw" "player_crouchwalk_sw.smd" fps 30 LX LY loop
$animation "crouchwalk_s" "player_crouchwalk_s.smd" fps 30 LX LY loop
$animation "crouchwalk_se" "player_crouchwalk_se.smd" fps 30 LX LY loop
$sequence "portalgun_standing_idle" "player_idle.smd" fps 24 loop ACT_MP_STAND_PRIMARY 1
$sequence "portalgun_crouch_idle" "player_crouch.smd" fps 24 loop ACT_MP_CROUCH_PRIMARY 1
$sequence "portalgun_standing_jump" "player_jump.smd" fps 24
$sequence "portalgun_standing_jump_blue_paint" "player_jump.smd" fps 24
$sequence "portalgun_jump_float" "player_float.smd" fps 24 loop
$sequence "portalgun_jump_float_tbeam" "player_float.smd" fps 24 loop
$sequence "portalgun_airwalk" "player_float.smd" fps 24 loop
$sequence "portalgun_jump_land" "player_idle.smd" fps 24
$sequence "portalgun_tractorbeam_float" "player_float.smd" fps 24 loop
$sequence "portalgun_standing_fire" "player_idle.smd" fps 24
$sequence "portalgun_crouch_fire" "player_crouch.smd" fps 24
$sequence "portalgun_run" {
 "run_nw" "run_n" "run_ne" "run_w" "run_center" "run_e" "run_sw" "run_s" "run_se"
 blendwidth 3
 blend move_y -1 1
 blend move_x 1 -1
 loop
 ACT_MP_RUN_PRIMARY 1
}
$sequence "portalgun_crouchWalk" {
 "crouchwalk_nw" "crouchwalk_n" "crouchwalk_ne" "crouchwalk_w" "crouchwalk_center" "crouchwalk_e" "crouchwalk_sw" "crouchwalk_s" "crouchwalk_se"
 blendwidth 3
 blend move_y -1 1
 blend move_x 1 -1
 loop
 ACT_MP_CROUCHWALK_PRIMARY 1
}
$sequence "portalgun_standing_aimmatrix" {
 "player_aim_0_0.smd"
 "player_aim_0_1.smd"
 "player_aim_0_2.smd"
 "player_aim_1_0.smd"
 "player_aim_1_1.smd"
 "player_aim_1_2.smd"
 "player_aim_2_0.smd"
 "player_aim_2_1.smd"
 "player_aim_2_2.smd"
 blendwidth 3
 blend body_yaw -45 45
 blend body_pitch -35 35
 subtract "idle_raw" 0
 delta
}
$sequence "portalgun_crouch_aimmatrix" {
 "player_aim_0_0.smd"
 "player_aim_0_1.smd"
 "player_aim_0_2.smd"
 "player_aim_1_0.smd"
 "player_aim_1_1.smd"
 "player_aim_1_2.smd"
 "player_aim_2_0.smd"
 "player_aim_2_1.smd"
 "player_aim_2_2.smd"
 blendwidth 3
 blend body_yaw -45 45
 blend body_pitch -35 35
 subtract "idle_raw" 0
 delta
}
$sequence "nogun_standing_idle" "player_idle.smd" fps 24 loop ACT_MP_STAND_IDLE 1
$sequence "nogun_crouch_idle" "player_crouch.smd" fps 24 loop ACT_MP_CROUCH_IDLE 1
$sequence "nogun_standing_jump" "player_jump.smd" fps 24
$sequence "nogun_standing_jump_blue_paint" "player_jump.smd" fps 24
$sequence "nogun_jump_float" "player_float.smd" fps 24 loop
$sequence "nogun_jump_float_tbeam" "player_float.smd" fps 24 loop
$sequence "nogun_airwalk" "player_float.smd" fps 24 loop
$sequence "nogun_jump_land" "player_idle.smd" fps 24
$sequence "nogun_tractorbeam_float" "player_float.smd" fps 24 loop
$sequence "nogun_standing_fire" "player_idle.smd" fps 24
$sequence "nogun_crouch_fire" "player_crouch.smd" fps 24
$sequence "nogun_run" {
 "run_nw" "run_n" "run_ne" "run_w" "run_center" "run_e" "run_sw" "run_s" "run_se"
 blendwidth 3
 blend move_y -1 1
 blend move_x 1 -1
 loop
 ACT_MP_RUN 1
}
$sequence "nogun_crouchWalk" {
 "crouchwalk_nw" "crouchwalk_n" "crouchwalk_ne" "crouchwalk_w" "crouchwalk_center" "crouchwalk_e" "crouchwalk_sw" "crouchwalk_s" "crouchwalk_se"
 blendwidth 3
 blend move_y -1 1
 blend move_x 1 -1
 loop
 ACT_MP_CROUCHWALK 1
}
$sequence "nogun_standing_aimmatrix" {
 "player_aim_0_0.smd"
 "player_aim_0_1.smd"
 "player_aim_0_2.smd"
 "player_aim_1_0.smd"
 "player_aim_1_1.smd"
 "player_aim_1_2.smd"
 "player_aim_2_0.smd"
 "player_aim_2_1.smd"
 "player_aim_2_2.smd"
 blendwidth 3
 blend body_yaw -45 45
 blend body_pitch -35 35
 subtract "idle_raw" 0
 delta
}
$sequence "nogun_crouch_aimmatrix" {
 "player_aim_0_0.smd"
 "player_aim_0_1.smd"
 "player_aim_0_2.smd"
 "player_aim_1_0.smd"
 "player_aim_1_1.smd"
 "player_aim_1_2.smd"
 "player_aim_2_0.smd"
 "player_aim_2_1.smd"
 "player_aim_2_2.smd"
 blendwidth 3
 blend body_yaw -45 45
 blend body_pitch -35 35
 subtract "idle_raw" 0
 delta
}
$sequence "idle" "player_idle.smd" fps 24 loop ACT_IDLE 1
$sequence "walk" "run_n" fps 24 loop ACT_WALK 1
$sequence "run" "run_n" fps 30 loop ACT_RUN 1
