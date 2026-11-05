"""Export Goblin's articulated rig with Portal 2's Bendy sequence interface.

Run with Blender. Unlike borrowing human animations, these motions retain the
reference's short limbs and oversized head. Source units and +X facing are explicit.
"""
import bpy
import math
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
LOCOMOTION_FPS=30
LOCOMOTION_FRAMES=19


def aim_bone(name, start, end):
    rest=rig.data.bones[name]
    rotation=(rest.tail_local-rest.head_local).rotation_difference(end-start)
    matrix=rotation.to_matrix().to_4x4()@rest.matrix_local
    matrix.translation=start
    rig.pose.bones[name].matrix=matrix
    bpy.context.view_layer.update()


def gait(phase,direction,crouch):
    reach=.38 if crouch else .51
    # Shorten sideways steps to keep the wide split toes from crossing.
    reach=min(reach,.27/max(.01,abs(direction.x)))
    bob=.015*(1-math.cos(phase*2))
    hip_drop=(.28 if crouch else .16)-bob
    rig.pose.bones['body'].location.y=-hip_drop
    # The knees point forward while the feet follow the movement direction.
    # Solve each leg in armature space so the sole stays level on contact.
    for label,offset in [('L',0),('R',.5)]:
        thigh,shin,foot=(f'{part}.{label}' for part in ('thigh','shin','foot'))
        hip=rig.data.bones[thigh].head_local-Vector((0,0,hip_drop))
        ankle=rig.data.bones[foot].head_local.copy()
        step=(phase/math.tau+offset)%1
        if step<.5:
            travel=1-4*step
        else:
            swing=(step-.5)*2
            travel=-math.cos(math.pi*swing)
            ankle.z+=(.16 if crouch else .23)*math.sin(math.pi*swing)**2
        ankle+=direction*(reach*travel)
        axis=(ankle-hip).normalized()
        forward=Vector((0,-1,0))
        knee_axis=(forward-axis*forward.dot(axis)).normalized()
        upper,lower=(rig.data.bones[name].length for name in (thigh,shin))
        distance=min((ankle-hip).length,upper+lower-.001)
        along=(upper**2-lower**2+distance**2)/(2*distance)
        knee=hip+axis*along+knee_axis*math.sqrt(max(0,upper**2-along**2))
        ankle=hip+axis*distance
        aim_bone(thigh,hip,knee)
        aim_bone(shin,knee,ankle)
        sole=rig.data.bones[foot].matrix_local.copy()
        sole.translation=ankle
        rig.pose.bones[foot].matrix=sole
        bpy.context.view_layer.update()
    rig.pose.bones['upperarm.R'].rotation_euler.x=.23*math.cos(phase)*-direction.y
    rig.pose.bones['forearm.R'].rotation_euler.x=-.12-.08*math.cos(phase)
    return reach*4


def pose(kind,frame,total,pitch=0,yaw=0,direction=None):
    for p in rig.pose.bones:
        p.rotation_mode='XYZ';p.rotation_euler=(0,0,0);p.location=(0,0,0);p.scale=(1,1,1)
    phase=frame/total*math.tau
    if kind=='crouch':
        rig.pose.bones['body'].location.y=-.28
        rig.pose.bones['head'].rotation_euler.x=.14
        if direction is None:
            bpy.context.view_layer.update()
            gait(0,Vector((0,0,0)),True)
    if direction is not None:
        bpy.context.view_layer.update()
        distance=gait(phase,direction,kind=='crouch')
        # StudioMDL extracts this travel for speed matching and removes it
        # from the rendered pose, leaving movement under player control.
        root=rig.pose.bones['root']
        root.location=rig.data.bones['root'].matrix_local.to_3x3().inverted()@(
            direction*(distance*frame/total))
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


def smd(name,kind='idle',frames=1,triangles=False,pitch=0,yaw=0,direction=None):
    with (SOURCE/name).open('w',newline='\n') as f:
        f.write('version 1\nnodes\n')
        for b in bones:
            f.write(f'{indices[b.name]} "{b.name}" {indices[b.parent.name] if b.parent else -1}\n')
        f.write('end\nskeleton\n')
        for frame in range(frames):
            pose(kind,frame,max(1,frames-1),pitch,yaw,direction)
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
for name,frames in [('idle',2),('crouch',2),('jump',8),('float',2)]:
    smd('player_'+name+'.smd',name,frames)

# Match Bendy's 3x3 move_y/move_x grid from forward-left to back-right.
directions=('nw','n','ne','w','center','e','sw','s','se')
for name,stance in [('run','idle'),('crouchwalk','crouch')]:
    for index,direction in enumerate(directions):
        if direction=='center':
            continue
        vector=Vector((1-index%3,index//3-1,0)).normalized()
        smd(f'player_{name}_{direction}.smd',stance,LOCOMOTION_FRAMES,direction=vector)
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
for name,stance in [('run','idle'),('crouchwalk','crouch')]:
    for direction in directions:
        if direction=='center':
            qc+=f'$animation "{name}_center" "player_{stance}.smd" fps {LOCOMOTION_FPS}\n'
        else:
            qc+=f'$animation "{name}_{direction}" "player_{name}_{direction}.smd" fps {LOCOMOTION_FPS} LX LY loop\n'
for prefix in ('portalgun','nogun'):
    armed=prefix=='portalgun'
    idle_activity='ACT_MP_STAND_PRIMARY' if armed else 'ACT_MP_STAND_IDLE'
    crouch_activity='ACT_MP_CROUCH_PRIMARY' if armed else 'ACT_MP_CROUCH_IDLE'
    qc+=f'$sequence "{prefix}_standing_idle" "player_idle.smd" fps 24 loop {idle_activity} 1\n'
    qc+=f'$sequence "{prefix}_crouch_idle" "player_crouch.smd" fps 24 loop {crouch_activity} 1\n'
    for name,file,extra in [
        ('standing_jump','jump',''),('standing_jump_blue_paint','jump',''),
        ('jump_float','float','loop'),('jump_float_tbeam','float','loop'),
        ('airwalk','float','loop'),('jump_land','idle',''),
        ('tractorbeam_float','float','loop'),('standing_fire','idle',''),
        ('crouch_fire','crouch',''),
    ]:
        qc+=f'$sequence "{prefix}_{name}" "player_{file}.smd" fps 24{(" "+extra) if extra else ""}\n'
    for sequence,name,activity in [('run','run','ACT_MP_RUN'),('crouchWalk','crouchwalk','ACT_MP_CROUCHWALK')]:
        qc+=f'$sequence "{prefix}_{sequence}" {{\n'
        qc+=' '+' '.join(f'"{name}_{direction}"' for direction in directions)+'\n'
        qc+=' blendwidth 3\n blend move_y -1 1\n blend move_x 1 -1\n loop\n'
        qc+=f' {activity}{"_PRIMARY" if armed else ""} 1\n}}\n'
    for stance in ('standing','crouch'):
        qc+=f'$sequence "{prefix}_{stance}_aimmatrix" {{\n'
        for p in range(3):
            for y in range(3):qc+=f' "player_aim_{p}_{y}.smd"\n'
        qc+=' blendwidth 3\n blend body_yaw -45 45\n blend body_pitch -35 35\n subtract "idle_raw" 0\n delta\n}\n'
qc+='$sequence "idle" "player_idle.smd" fps 24 loop ACT_IDLE 1\n'
qc+='$sequence "walk" "run_n" fps 24 loop ACT_WALK 1\n'
qc+='$sequence "run" "run_n" fps 30 loop ACT_RUN 1\n'
(SOURCE/'goblin_player.qc').write_text(qc)
(SOURCE/'goblin_preview.qc').write_text(qc.replace('info_character/info_character_player.mdl','goblin_codex/goblin_preview.mdl'))
pose('idle',0,1)
print('Exported player mesh, mechanical animations, and Portal 2 sequence interface')
