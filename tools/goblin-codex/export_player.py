"""Export Goblin's articulated rig with Portal 2's Bendy sequence interface.

Run with Blender. Unlike borrowing human animations, these motions retain the
reference's short limbs and oversized head. Source units and +X facing are explicit.
"""
import bpy
import math
import json
from pathlib import Path
from mathutils import Matrix, Vector

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'assets/goblin-codex'
SOURCE=OUT/'source'
bpy.ops.wm.open_mainfile(filepath=str(OUT/'goblin.blend'))
rig=bpy.data.objects['Goblin mechanical rig']
rig.animation_data_clear()
# Portal 2 merges its world portal gun onto a bone named weapon_bone.
# Keep the gun steady at the raised right palm rather than inheriting the
# stock character's chest/eye-height placement inside our oversized head.
for p in rig.pose.bones:
    p.rotation_mode='XYZ';p.rotation_euler=(0,0,0);p.location=(0,0,0)
rig.pose.bones['upperarm.L'].rotation_euler.x=-.35
rig.pose.bones['forearm.L'].rotation_euler.x=-.80
bpy.context.view_layer.update()
palm=Vector((-.95,-.13,1.65))
held_palm=rig.pose.bones['hand.L'].matrix@rig.data.bones['hand.L'].matrix_local.inverted()@palm
for p in rig.pose.bones:p.rotation_euler=(0,0,0)
bpy.context.view_layer.update()
bpy.context.view_layer.objects.active=rig
rig.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')
weapon=rig.data.edit_bones.new('weapon_bone')
weapon.parent=rig.data.edit_bones['body']
weapon.matrix=Matrix(((0,-1,0,held_palm.x),(0,0,1,held_palm.y),(-1,0,0,held_palm.z),(0,0,0,1)))
weapon.length=.2
bpy.ops.object.mode_set(mode='OBJECT')
meshes=[o for o in bpy.context.scene.objects if o.type=='MESH' and 'part_bone' in o]
bones=list(rig.data.bones)
indices={b.name:i for i,b in enumerate(bones)}
# StudioMDL's default sequence import adds the Source +X-facing rotation.
# Keep Blender's -Y-facing mesh and reference skeleton in the same input basis.
ROT=Matrix.Identity(4)
SCALE=12


def pose(kind,frame,total,pitch=0,yaw=0):
    for p in rig.pose.bones:
        p.rotation_mode='XYZ';p.rotation_euler=(0,0,0);p.location=(0,0,0);p.scale=(1,1,1)
    phase=frame/total*math.tau
    if kind in ('walk','crouch_walk'):
        for label,sign in [('L',1),('R',-1)]:
            for part,amp in [('upperarm',.22),('forearm',.10),('thigh',.38),('shin',.24),('foot',.12)]:
                rig.pose.bones[f'{part}.{label}'].rotation_euler.x=math.sin(phase)*amp*sign
        # Small engine-body vibration, rather than soft organic breathing.
        rig.pose.bones['body'].location.y=.018*math.cos(phase*2)
    if kind.startswith('crouch'):
        rig.pose.bones['body'].location.y=-.28
        rig.pose.bones['head'].rotation_euler.x=.14
        for label in ('L','R'):
            rig.pose.bones[f'thigh.{label}'].rotation_euler.x+=.28
            rig.pose.bones[f'shin.{label}'].rotation_euler.x-=.48
    if kind in ('jump','float'):
        for label in ('L','R'):
            rig.pose.bones[f'thigh.{label}'].rotation_euler.x=.20
            rig.pose.bones[f'shin.{label}'].rotation_euler.x=-.35
            rig.pose.bones[f'upperarm.{label}'].rotation_euler.x=-.18
    if kind!='reference':
        rig.pose.bones['upperarm.L'].rotation_euler.x=-.35
        rig.pose.bones['forearm.L'].rotation_euler.x=-.80
    rig.pose.bones['head'].rotation_euler.x+=pitch
    rig.pose.bones['head'].rotation_euler.z+=yaw
    bpy.context.view_layer.update()


def smd(name,kind='idle',frames=1,triangles=False,pitch=0,yaw=0):
    with (SOURCE/name).open('w',newline='\n') as f:
        f.write('version 1\nnodes\n')
        for b in bones:
            f.write(f'{indices[b.name]} "{b.name}" {indices[b.parent.name] if b.parent else -1}\n')
        f.write('end\nskeleton\n')
        for frame in range(frames):
            pose(kind,frame,max(1,frames-1),pitch,yaw)
            f.write(f'time {frame}\n')
            for b in bones:
                m=rig.pose.bones[b.name].matrix.copy()
                if b.parent:m=rig.pose.bones[b.parent.name].matrix.inverted()@m
                else:m=ROT@m
                t=m.to_translation()*SCALE;r=m.to_euler('XYZ')
                f.write(f'{indices[b.name]} '+' '.join(f'{v:.6f}' for v in (*t,*r))+'\n')
        f.write('end\n')
        if triangles:
            f.write('triangles\n')
            for o in meshes:
                mesh=o.data;mesh.calc_loop_triangles()
                normal_matrix=(ROT@o.matrix_world).to_3x3().inverted().transposed()
                bi=indices[o['part_bone']]
                for tri in mesh.loop_triangles:
                    f.write('goblin_atlas\n')
                    for li in tri.loops:
                        loop=mesh.loops[li];v=ROT@o.matrix_world@mesh.vertices[loop.vertex_index].co*SCALE
                        n=(normal_matrix@loop.normal).normalized();uv=mesh.uv_layers.active.data[li].uv
                        f.write(f'{bi} '+' '.join(f'{v:.6f}' for v in (*v,*n,*uv))+f' 1 {bi} 1.0\n')
            f.write('end\n')


smd('player_reference.smd',kind='reference',triangles=True)
for name,kind,frames in [('idle','idle',2),('run','walk',25),('crouch','crouch',2),('crouchwalk','crouch_walk',25),('jump','jump',8),('float','float',2)]:
    smd('player_'+name+'.smd',kind,frames)
for p in range(3):
    for y in range(3):smd(f'player_aim_{p}_{y}.smd',pitch=math.radians((p-1)*35),yaw=math.radians((y-1)*45))

qc='''// Goblin: custom proportions and mechanical animation, Portal 2 player sequence names.
$modelname "info_character/info_character_player.mdl"
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
'''
qc+='$sequence "ref" "player_idle.smd" fps 24\n'
qc+='$sequence "idle_raw" "player_idle.smd" fps 24\n'
for prefix in ('portalgun','nogun'):
    for name,file,extra in [
        ('standing_idle','idle','loop'),('crouch_idle','crouch','loop'),
        ('standing_jump','jump',''),('standing_jump_blue_paint','jump',''),
        ('jump_float','float','loop'),('jump_float_tbeam','float','loop'),
        ('airwalk','float','loop'),('jump_land','idle',''),
        ('tractorbeam_float','float','loop'),('standing_fire','idle',''),
        ('crouch_fire','crouch',''),('crouchWalk','crouchwalk','loop'),
        ('run','run','loop'),
    ]:
        qc+=f'$sequence "{prefix}_{name}" "player_{file}.smd" fps 24{(" "+extra) if extra else ""}\n'
    for stance in ('standing','crouch'):
        qc+=f'$sequence "{prefix}_{stance}_aimmatrix" {{\n'
        for p in range(3):
            for y in range(3):qc+=f' "player_aim_{p}_{y}.smd"\n'
        qc+=' blendwidth 3\n blend body_yaw -45 45\n blend body_pitch -35 35\n subtract "idle_raw" 0\n delta\n}\n'
qc+='$sequence "idle" "player_idle.smd" fps 24 loop ACT_IDLE 1\n'
qc+='$sequence "walk" "player_run.smd" fps 24 loop ACT_WALK 1\n'
qc+='$sequence "run" "player_run.smd" fps 30 loop ACT_RUN 1\n'
(SOURCE/'goblin_player.qc').write_text(qc)
(SOURCE/'goblin_preview.qc').write_text(qc.replace('info_character/info_character_player.mdl','goblin_codex/goblin_preview.mdl'))
pose('idle',0,1)
print('Exported player mesh, mechanical animations, and Portal 2 sequence interface')
