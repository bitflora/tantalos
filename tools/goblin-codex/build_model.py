"""Build Goblin from geometry, not a billboard. Run with Blender --background --python.

Outputs an editable .blend, skinned .glb, Source SMD, QC, and studio renders.
All components, materials, UVs and animation are reproducible from this file.
"""
import bpy
import math
import json
import random
import sys
from pathlib import Path
from mathutils import Vector, Matrix

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets/goblin-codex"
OUT.mkdir(parents=True, exist_ok=True)
random.seed(620)
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
MESHES = []
PARTS = {}
MAT_NAMES = ['Ivory ceramic','Gunmetal','Machined steel','Oxide green','Warning ochre',
             'Broken amethyst','Yellow optic','Recess black','Copper','Cable rubber',
             'Teal cable','Brass','Edge metal','Graphite','Purple highlights','Warm ivory']
ATLAS = bpy.data.images.load(str(OUT / 'goblin_atlas.png'))
MATS = []
for i, name in enumerate(MAT_NAMES):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    image = mat.node_tree.nodes.new('ShaderNodeTexImage')
    image.image = ATLAS
    mat.node_tree.links.new(image.outputs['Color'],bsdf.inputs['Base Color'])
    bsdf.inputs['Metallic'].default_value = .7 if i in (1,2,8,11,12,13) else .24
    bsdf.inputs['Roughness'].default_value = .31 if i in (2,5,6,14) else .63
    noise = mat.node_tree.nodes.new('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value=145
    bump=mat.node_tree.nodes.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value=.19
    bump.inputs['Distance'].default_value=.025
    mat.node_tree.links.new(noise.outputs['Fac'],bump.inputs['Height'])
    mat.node_tree.links.new(bump.outputs['Normal'],bsdf.inputs['Normal'])
    if i==6:
        mat.node_tree.links.new(image.outputs['Color'],bsdf.inputs['Emission Color'])
        bsdf.inputs['Emission Strength'].default_value=.75
    MATS.append(mat)


def finish(obj, name, mat=0, bone='body', smooth=True, bevel=0):
    obj.name = name
    obj.data.materials.clear()
    obj.data.materials.append(MATS[mat])
    if bevel:
        mod=obj.modifiers.new('Machined edge radii','BEVEL')
        mod.width=bevel
        mod.segments=2
        bpy.context.view_layer.objects.active=obj
        bpy.ops.object.modifier_apply(modifier=mod.name)
    for p in obj.data.polygons:
        p.use_smooth=smooth
    MESHES.append(obj)
    PARTS[obj.name] = bone
    obj['part_bone']=bone
    obj['atlas_tile']=mat
    return obj


def ell(name, p, scale, mat=0, bone='head', seg=32, rings=20):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=seg,ring_count=rings,location=p)
    o=bpy.context.object
    o.scale=scale
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    return finish(o,name,mat,bone)


def box(name,p,scale,mat=0,bone='body',bevel=.035):
    bpy.ops.mesh.primitive_cube_add(size=1,location=p)
    o=bpy.context.object
    o.scale=scale
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    return finish(o,name,mat,bone,False,bevel)


def rod(name,a,b,r,mat=2,bone='body',r2=None,verts=16):
    d=Vector(b)-Vector(a)
    bpy.ops.mesh.primitive_cone_add(vertices=verts,radius1=r,radius2=r if r2 is None else r2,depth=d.length,location=(Vector(a)+Vector(b))/2)
    o=bpy.context.object
    o.rotation_mode='QUATERNION'
    o.rotation_quaternion=d.to_track_quat('Z','Y')
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    return finish(o,name,mat,bone)


def tube(name,pts,r,mat=1,bone='head'):
    c=bpy.data.curves.new(name,'CURVE')
    c.dimensions='3D'
    c.resolution_u=12
    c.bevel_depth=r
    c.bevel_resolution=2
    s=c.splines.new('BEZIER')
    s.bezier_points.add(len(pts)-1)
    for bp,p in zip(s.bezier_points,pts):
        bp.co=p
        bp.handle_left_type='AUTO'
        bp.handle_right_type='AUTO'
    o=bpy.data.objects.new(name,c)
    bpy.context.collection.objects.link(o)
    bpy.context.view_layer.objects.active=o
    o.select_set(True)
    bpy.ops.object.convert(target='MESH')
    o.select_set(False)
    return finish(o,name,mat,bone)


def ring(name,p,major,minor,mat=1,bone='body',rotation=None):
    bpy.ops.mesh.primitive_torus_add(major_segments=40,minor_segments=8,location=p,major_radius=major,minor_radius=minor)
    o=bpy.context.object
    if rotation:o.rotation_euler=rotation
    return finish(o,name,mat,bone)


def bolt(name,p,r=.055,bone='head',axis='Y'):
    d=Vector((0,1,0) if axis=='Y' else (1,0,0) if axis=='X' else (0,0,1))
    p=Vector(p)
    rod(name+' washer',p-d*.008,p+d*.018,r*1.45,1,bone,verts=16)
    rod(name,p-d*.016,p-d*.035,r,2,bone,verts=6)
    # Recess in the screw head.
    o=box(name+' slot',p-d*.038,(r*1.2,.008,r*.20) if axis=='Y' else (.008,r*1.2,r*.20),7,bone,.002)
    return o


def panel(name,front,depth,mat=0,bone='head'):
    # Closed extruded irregular panel. Front points run counterclockwise as seen externally.
    n=len(front)
    vertices=list(front)+[(x,y+depth,z) for x,y,z in front]
    faces=[tuple(range(n-1,-1,-1)),tuple(range(n,2*n))]
    faces += [(i,(i+1)%n,(i+1)%n+n,i+n) for i in range(n)]
    mesh=bpy.data.meshes.new(name)
    mesh.from_pydata(vertices,[],faces)
    mesh.update()
    o=bpy.data.objects.new(name,mesh)
    bpy.context.collection.objects.link(o)
    return finish(o,name,mat,bone,False,.025)


# Face: fused volume, with the sockets cut through the surface.
forms=[]
forms.append(ell('Cranial shell',(0,.04,4.43),(.91,.67,1.02)))
forms.append(ell('Lower face',(0,-.27,3.93),(.75,.55,.59)))
for side in (-1,1):
    forms.append(ell('Jowl',(side*.49,-.48,3.93),(.33,.39,.51)))
    forms.append(ell('Sagging cheek',(side*.53,-.56,4.17),(.29,.30,.43)))
    forms.append(ell('Muzzle fold',(side*.25,-.77,3.88),(.29,.21,.27)))
    brow=ell('Heavy brow',(side*.38,-.58,4.65),(.46,.28,.19))
    brow.rotation_euler[1]=side*.16
    forms.append(brow)
forms.append(ell('Nose bridge',(0,-.71,4.38),(.18,.24,.43)))
forms.append(ell('Nose bulb',(0,-.96,4.15),(.24,.27,.17)))
forms.append(ell('Proud chin',(0,-.54,3.58),(.49,.40,.25)))
bpy.ops.object.select_all(action='DESELECT')
for o in forms:o.select_set(True)
bpy.context.view_layer.objects.active=forms[0]
bpy.ops.object.join()
head=bpy.context.object
for o in forms[1:]:
    if o in MESHES:MESHES.remove(o)
remesh=head.modifiers.new('Fused sculpt volumes','REMESH')
remesh.mode='VOXEL'
remesh.voxel_size=.025
bpy.ops.object.modifier_apply(modifier=remesh.name)
smooth=head.modifiers.new('Polished cast shell','SMOOTH')
smooth.factor=1.0
smooth.iterations=5
bpy.ops.object.modifier_apply(modifier=smooth.name)
for side in (-1,1):
    cut=ell('Socket tool',(side*.39,-.66,4.44),(.30,.30,.245),7)
    mod=head.modifiers.new('Recessed eye socket','BOOLEAN')
    mod.operation='DIFFERENCE';mod.object=cut
    bpy.context.view_layer.objects.active=head
    bpy.ops.object.modifier_apply(modifier=mod.name)
    MESHES.remove(cut)
    bpy.data.objects.remove(cut,do_unlink=True)
dec=head.modifiers.new('Game surface reduction','DECIMATE')
dec.ratio=.42
bpy.context.view_layer.objects.active=head
bpy.ops.object.modifier_apply(modifier=dec.name)
head.name='Sculpted goblin face and shell'
PARTS[head.name]='head';head['part_bone']='head';head['atlas_tile']=0
for p in head.data.polygons:p.use_smooth=True
head.data.materials.append(MATS[3])
for p in head.data.polygons:
    c=head.matrix_world@p.center
    cheek=((c.x-.62)/.20)**2+((c.z-4.02)/.24)**2
    temple=((c.x+.58)/.17)**2+((c.z-4.92)/.18)**2
    ragged=.20*math.sin(c.x*91)*math.sin(c.z*83)+.12*math.sin(c.z*157)
    if c.y<-.40 and min(cheek,temple)<1+ragged:p.material_index=1

# Optics and broken purple lens. Camera faces negative Y.
for side in (-1,1):
    x=side*.39
    ell('Deep socket',(x,-.61,4.44),(.275,.13,.224),7)
    ring('Optic bezel',(x,-.755,4.43),.175,.038,2,'head',(math.pi/2,0,0))
    if side==-1:
        ell('Yellow glass optic',(x,-.795,4.43),(.142,.055,.145),6)
        ell('Optic inner pupil',(x+.006,-.846,4.43),(.055,.016,.095),7)
        ell('Optic reflected glint',(x-.042,-.859,4.48),(.025,.009,.035),15,seg=12,rings=8)
    else:
        # Irregular tessellated lens; gaps stay physically open.
        pts=[(-.17,.08),(-.09,.18),(.075,.17),(.18,.055),(.15,-.10),(.015,-.17),(-.14,-.105),(0,.025)]
        faces=[(0,1,7),(1,2,7),(2,3,7),(3,4,7),(4,5,7),(5,6,7),(6,0,7)]
        for j,inds in enumerate(faces):
            coords=[pts[k] for k in inds]
            cx=sum(p[0] for p in coords)/3;cz=sum(p[1] for p in coords)/3
            front=[(x+cx+(px-cx)*.9,-.812+random.uniform(-.018,.018),4.44+cz+(pz-cz)*.9) for px,pz in coords]
            panel('Cracked purple lens shard %02d'%j,front,.028,5 if j%3 else 14)
    tube('Drooping upper eyelid',[(x-.24,-.78,4.48),(x,-.87,4.54),(x+.24,-.78,4.49)],.062,0)
    tube('Lower eye rim',[(x-.23,-.73,4.39),(x,-.80,4.28),(x+.22,-.73,4.38)],.032,12)
    tube('Under-eye crease',[(x-.22,-.76,4.23),(x,-.79,4.18),(x+.16,-.73,4.22)],.012,7)

# Frown, nostrils, deep expression lines.
mouth=[(-.46,-.805,3.64),(-.29,-.93,3.79),(0,-.989,3.86),(.29,-.93,3.79),(.46,-.805,3.64)]
tube('Downturned mouth recess',mouth,.041,7)
tube('Heavy lower lip',[(x,y-.008,z-.075) for x,y,z in mouth],.047,15)
for side in (-1,1):
    ell('Nostril',(side*.155,-1.137,4.085),(.07,.027,.042),7,seg=16,rings=10)
    tube('Nose wing crease',[(side*.19,-1.105,4.19),(side*.26,-.99,4.08),(side*.37,-.88,3.99),(side*.46,-.82,3.80)],.014,7)
    tube('Forehead worry line',[(side*.07,-.67,4.99),(side*.13,-.76,4.84),(side*.10,-.82,4.73)],.012,7)
    tube('Helmet brow seam',[(side*.02,-.608,5.03),(side*.36,-.615,5.08),(side*.68,-.47,5.04),(side*.80,-.29,5.13)],.016,1)
    tube('Side cranial panel seam',[(side*.71,-.31,5.12),(side*.85,-.35,4.89),(side*.87,-.33,4.51),(side*.78,-.40,4.18),(side*.70,-.30,3.94)],.014,1)
    for z in (4.0,4.72):bolt('Temple captive screw',(side*.77,-.42,z),.037)
bolt('Forehead service screw',(.0,-.568,5.22),.033)

# Pointed, concave mechanical ears, with open linkages in front of the backing.
for s in (-1,1):
    panel('Pointed ear shell',[(s*.77,.015,4.38),(s*1.34,.015,4.30),(s*1.36,.015,4.60),(s*1.19,.015,5.70),(s*1.02,.015,4.99)],.17,0)
    panel('Ear dark recessed interior',[(s*.88,-.025,4.48),(s*1.23,-.025,4.43),(s*1.25,-.025,4.64),(s*1.16,-.025,5.42),(s*1.07,-.025,4.92)],.025,1)
    rod('Ear piston shaft',(s*.94,-.095,4.55),(s*1.16,-.095,5.10),.034,2,'head')
    rod('Ear piston sleeve',(s*.94,-.095,4.55),(s*1.05,-.095,4.84),.065,13,'head')
    rod('Ear articulated strut',(s*1.22,-.10,4.52),(s*1.17,-.10,5.07),.038,2,'head')
    for x,z,r in [(.94,4.55,.105),(1.17,5.07,.082),(1.22,4.52,.065)]:
        rod('Ear circular pivot',(s*x,-.06,z),(s*x,-.17,z),r,2,'head',verts=24)
        rod('Ear hub cap',(s*x,-.175,z),(s*x,-.19,z),r*.55,1,'head',verts=20)
    rod('Ear tip ferrule',(s*1.19,.08,5.58),(s*1.19,.08,5.77),.063,2,'head',r2=.012)

# Collar, hazard band and wind-up key.
rod('Black neck collar',(0,0,3.02),(0,0,3.40),.72,1,'neck',verts=64)
ring('Collar upper rolled edge',(0,0,3.39),.715,.04,2,'neck')
ring('Collar lower rolled edge',(0,0,3.02),.715,.035,2,'neck')
rod('Hazard band',(0,0,3.035),(0,0,3.15),.728,4,'neck',verts=64)
for j in range(22):
    a=2*math.pi*j/22
    verts=[]
    for da,z in [(0,3.035),(.10,3.035),(.19,3.15),(.09,3.15)]:
        verts.append((.731*math.cos(a+da),.731*math.sin(a+da),z))
    panel('Diagonal hazard stripe',verts,.004,1,'neck')
for a in [-2.7,-2.15,-1.57,-.95,-.4,.5,1.4,2.3]:
    x,y=.727*math.cos(a),.727*math.sin(a)
    ell('Collar rivet',(x,y,3.27),(.034,.034,.034),12,'neck',seg=12,rings=8)
rod('Key spindle',(.69,.15,3.27),(1.03,.15,3.27),.061,2,'neck')
box('Wind-up key crossbar',(1.08,.15,3.27),(.09,.10,.40),1,'neck',.025)
box('Wind-up key upper tab',(1.08,.15,3.50),(.12,.13,.14),2,'neck',.026)
box('Wind-up key lower tab',(1.08,.15,3.04),(.12,.13,.14),2,'neck',.026)

# Open torso: back armor and side shells wrap around a visible engine bay.
ell('Torso back carapace',(0,.20,2.28),(.66,.42,.72),0,'body')
belly=ell('Belly bowl',(0,-.025,1.89),(.63,.49,.44),0,'body')
belly.data.materials.append(MATS[3])
for p in belly.data.polygons:
    c=belly.matrix_world@p.center
    patch=((c.x+.31)/.20)**2+((c.z-1.88)/.17)**2
    if c.y<-.32 and patch<1+.25*math.sin(c.z*80):p.material_index=1
panel('Left engine-bay side',[(-.63,-.23,2.13),(-.46,-.45,2.01),(-.48,-.45,2.67),(-.59,-.22,2.85)],.21,0,'body')
panel('Right engine-bay side',[(.46,-.45,2.01),(.63,-.23,2.13),(.59,-.22,2.85),(.48,-.45,2.67)],.21,0,'body')
panel('Shoulder mantle',[(-.68,-.27,2.87),(-.54,-.60,2.82),(-.21,-.59,2.97),(.21,-.59,2.97),(.54,-.60,2.82),(.68,-.27,2.87),(.43,-.24,3.07),(-.43,-.24,3.07)],.25,0,'body')
box('Engine bay darkness',(0,-.24,2.44),(.85,.10,.72),7,'body',.08)
for s in (-1,1):
    rod('Chest piston outer',(s*.28,-.43,2.18),(s*.40,-.43,2.70),.10,1,'body')
    rod('Chest polished piston',(s*.20,-.43,2.10),(s*.28,-.43,2.34),.054,2,'body')
    ring('Piston compression ring',(s*.335,-.43,2.51),.103,.014,2,'body')
    bolt('Mantle bolt',(s*.48,-.565,2.86),.044,'body')
    tube('Copper chest conduit',[(s*.46,-.38,2.17),(s*.51,-.35,2.40),(s*.49,-.32,2.64)],.022,8,'body')
box('Central engine spine',(0,-.39,2.43),(.17,.17,.64),2,'body',.018)
for z in (2.22,2.47,2.68):
    box('Engine spine block',(0,-.48,z),(.23,.09,.15),11,'body',.012)
    rod('Spine black socket',(0,-.525,z),(0,-.54,z),.052,7,'body')
tube('Teal engine wire',[(-.4,-.40,2.13),(-.25,-.52,2.10),(.07,-.54,2.17),(.45,-.40,2.32)],.019,10,'body')
tube('Amber engine wire',[(-.41,-.38,2.35),(-.2,-.50,2.16),(.2,-.51,2.12),(.46,-.35,2.25)],.017,4,'body')
ell('Belly service hole',(0,-.494,1.83),(.16,.025,.10),7,'body')
rod('Belly service pin',(-.06,-.52,1.83),(.09,-.52,1.86),.031,2,'body')
for s in (-1,1):
    # Side turbine, shallow cylinder along X.
    rod('Hip turbine casing',(s*.55,.05,1.97),(s*.73,.05,1.97),.205,1,'body',verts=32)
    rod('Hip turbine hub',(s*.73,.05,1.97),(s*.77,.05,1.97),.12,2,'body',verts=24)
    for j in range(10):
        a=j*math.tau/10
        box('Turbine radial tooth',(s*.76,.05+.165*math.cos(a),1.97+.165*math.sin(a)),(.045,.043,.055),13,'body',.008)

# Limbs, layered joint bearings, articulated claw fingers and split feet.
for s,label in [(-1,'L'),(1,'R')]:
    upper=f'upperarm.{label}';lower=f'forearm.{label}';hand=f'hand.{label}'
    thigh=f'thigh.{label}';shin=f'shin.{label}';foot=f'foot.{label}'
    shoulder=(s*.71,.015,2.80);elbow=(s*.91,-.035,2.28);wrist=(s*.95,-.12,1.80)
    rod('Shoulder axle',(s*.60,.015,2.80),(s*.83,.015,2.80),.16,2,upper,verts=24)
    rod('Upper arm steel link',shoulder,elbow,.072,2,upper)
    rod('Upper arm hydraulic housing',(s*.75,-.02,2.69),(s*.85,-.045,2.45),.095,1,upper)
    tube('Arm hydraulic line',[(s*.78,.07,2.76),(s*.94,.08,2.48),(s*.99,.0,2.29)],.024,9,upper)
    rod('Elbow bearing',(s*.85,-.03,2.28),(s*1.01,-.03,2.28),.123,2,lower,verts=24)
    rod('Elbow center cap',(s*1.01,-.03,2.28),(s*1.035,-.03,2.28),.067,1,lower)
    rod('Forearm internal rod',elbow,wrist,.052,2,lower)
    box('Forearm ivory guard',(s*.96,-.07,2.02),(.19,.19,.38),0,lower,.027)
    box('Forearm black channel',(s*.96,-.175,2.02),(.074,.02,.24),7,lower,.008)
    rod('Wrist swivel',(s*.88,-.12,1.79),(s*1.02,-.12,1.79),.08,2,hand)
    box('Palm plate',(s*.95,-.13,1.65),(.19,.16,.20),0,hand,.026)
    for j in range(3):
        x=s*(.865+j*.083)
        a=(x,-.17,1.55);b=(x,-.22,1.40);c=(x,-.31,1.33)
        rod('Claw proximal finger',a,b,.032,2,hand,verts=8)
        rod('Claw distal finger',b,c,.031,12,hand,verts=8)
        ell('Finger knuckle',b,(.044,.044,.044),1,hand,seg=12,rings=8)
    rod('Opposing thumb',(s*.85,-.12,1.67),(s*.78,-.23,1.52),.039,2,hand,verts=8)
    hip=(s*.39,.10,1.65);knee=(s*.43,-.005,.93);ankle=(s*.46,-.01,.27)
    rod('Hip ball axle',(s*.28,.10,1.59),(s*.50,.10,1.59),.13,1,thigh,verts=24)
    rod('Thigh structural link',hip,knee,.082,2,thigh)
    armor=box('Thigh shell',(s*.43,.04,1.31),(.28,.25,.40),0,thigh,.06)
    armor.rotation_euler[1]=s*.14
    box('Thigh front slit',(s*.43,-.096,1.34),(.12,.015,.20),7,thigh,.01)
    rod('Knee joint axle',(s*.33,-.005,.93),(s*.56,-.005,.93),.143,1,shin,verts=24)
    rod('Knee silver bearing',(s*.55,-.005,.93),(s*.60,-.005,.93),.104,2,shin,verts=24)
    rod('Knee dark cap',(s*.60,-.005,.93),(s*.615,-.005,.93),.057,1,shin)
    for dx in (-.069,.069):
        rod('Shin parallel strut',(s*.43+dx,-.005,.90),(s*.46+dx,-.01,.31),.040,2,shin)
    rod('Shin hydraulic sleeve',(s*.43,.06,.82),(s*.45,.06,.56),.065,1,shin)
    rod('Ankle hinge',(s*.35,-.01,.27),(s*.57,-.01,.27),.095,2,foot)
    for dx in (-.095,.095):
        box('Split toe',(s*.46+dx,-.16,.115),(.15,.48,.16),12,foot,.028)
        box('Toe dark sole',(s*.46+dx,-.16,.035),(.152,.46,.035),1,foot,.008)
    rod('Foot instep brace',(s*.46,.02,.28),(s*.46,-.28,.14),.045,1,foot)

# Back access details are designed to match the visible front reference.
panel('Rear spine access panel',[(-.36,.58,2.12),(.36,.58,2.12),(.40,.58,2.65),(-.40,.58,2.65)],-.06,3,'body')
for z in (2.24,2.34,2.44,2.54):box('Back cooling louvre',(0,.62,z),(.50,.035,.035),1,'body',.006)
for x in (-.29,.29):
    for z in (2.18,2.59):ell('Back panel rivet',(x,.63,z),(.035,.025,.035),2,'body',seg=12,rings=8)

# Apply shared UV tile coordinates. UVs stay inside tile gutters in all exports.
MESHES=[o for o in bpy.context.scene.objects if o.type=='MESH']
bpy.ops.object.select_all(action='DESELECT')
for o in MESHES:
    bpy.context.view_layer.objects.active=o
    o.select_set(True)
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=1.15,island_margin=.025)
    bpy.ops.object.mode_set(mode='OBJECT')
    for polygon in o.data.polygons:
        tile=3 if polygon.material_index==1 and o in (head,belly) else int(o.get('atlas_tile',0))
        tx=tile%4;ty=3-tile//4
        for li in polygon.loop_indices:
            uv=o.data.uv_layers.active.data[li]
            uv.uv.x=(tx+.02+uv.uv.x*.96)/4
            uv.uv.y=(ty+.02+uv.uv.y*.96)/4
    o.select_set(False)

# Editable rigid mechanical rig; each component rotates around its actual joint.
bpy.ops.object.armature_add(location=(0,0,0))
rig=bpy.context.object;rig.name='Goblin mechanical rig'
bpy.ops.object.mode_set(mode='EDIT')
rig.data.edit_bones.remove(rig.data.edit_bones[0])
DEFS={
 'root':((0,0,0),(0,0,.35),None),
 'body':((0,.1,1.65),(0,0,2.9),'root'),
 'neck':((0,0,2.98),(0,0,3.35),'body'),
 'head':((0,0,3.35),(0,0,4.8),'neck'),
}
for s,label in [(-1,'L'),(1,'R')]:
    DEFS.update({
      f'upperarm.{label}':((s*.71,.015,2.8),(s*.91,-.035,2.28),'body'),
      f'forearm.{label}':((s*.91,-.035,2.28),(s*.95,-.12,1.8),f'upperarm.{label}'),
      f'hand.{label}':((s*.95,-.12,1.8),(s*.95,-.2,1.45),f'forearm.{label}'),
      f'thigh.{label}':((s*.39,.1,1.65),(s*.43,-.005,.93),'root'),
      f'shin.{label}':((s*.43,-.005,.93),(s*.46,-.01,.27),f'thigh.{label}'),
      f'foot.{label}':((s*.46,-.01,.27),(s*.46,-.3,.1),f'shin.{label}'),
    })
for name,(h,t,parent) in DEFS.items():
    bone=rig.data.edit_bones.new(name);bone.head=h;bone.tail=t
    if parent:bone.parent=rig.data.edit_bones[parent]
bpy.ops.object.mode_set(mode='OBJECT')
for o in MESHES:
    bone=o.get('part_bone','body')
    vg=o.vertex_groups.new(name=bone)
    vg.add(list(range(len(o.data.vertices))),1,'REPLACE')
    mod=o.modifiers.new('Rigid mechanical articulation','ARMATURE');mod.object=rig
    o.parent=rig
rig.show_in_front=True

# A short actual rig animation, useful for verifying pivots outside the game.
scene=bpy.context.scene
scene.frame_start=1;scene.frame_end=48;scene.render.fps=24
for frame in (1,13,25,37,49):
    scene.frame_set(frame)
    phase=(frame-1)/48*math.tau
    for label,sign in [('L',1),('R',-1)]:
        for part,amp in [('upperarm',.24),('forearm',.10),('thigh',.26),('shin',.16),('foot',.08)]:
            p=rig.pose.bones[f'{part}.{label}'];p.rotation_mode='XYZ'
            p.rotation_euler.x=math.sin(phase)*amp*sign
            p.keyframe_insert(data_path='rotation_euler',frame=frame,group=part)
    p=rig.pose.bones['head'];p.rotation_mode='XYZ'
    p.rotation_euler.z=math.sin(phase)*.035
    p.keyframe_insert(data_path='rotation_euler',frame=frame,group='head')
if rig.animation_data and rig.animation_data.action:rig.animation_data.action.name='Goblin mechanical walk preview'
scene.frame_set(1)

# Export the editable model and portable skinned model before adding studio objects.
bpy.ops.object.select_all(action='DESELECT')
for o in MESHES:o.select_set(True)
rig.select_set(True);bpy.context.view_layer.objects.active=rig
bpy.ops.export_scene.gltf(filepath=str(OUT/'goblin.glb'),use_selection=True,export_format='GLB',export_animations=True,export_apply=False)

# Native Source reference mesh and local bone transforms. Units: 12 Source units/modeled unit.
SCALE=12.0
bones=list(rig.data.bones)
index={b.name:i for i,b in enumerate(bones)}
def write_smd(path,triangles=False,animated=False):
    with path.open('w',newline='\n') as f:
        f.write('version 1\nnodes\n')
        for b in bones:f.write(f'{index[b.name]} "{b.name}" {index[b.parent.name] if b.parent else -1}\n')
        f.write('end\nskeleton\n')
        for frame in (range(1,49) if animated else [1]):
            scene.frame_set(frame)
            f.write(f'time {frame-1}\n')
            for b in bones:
                m=rig.pose.bones[b.name].matrix.copy() if animated else b.matrix_local.copy()
                if b.parent:
                    pm=rig.pose.bones[b.parent.name].matrix if animated else b.parent.matrix_local
                    m=pm.inverted()@m
                t=m.to_translation()*SCALE;r=m.to_euler('XYZ')
                f.write(f'{index[b.name]} '+' '.join(f'{v:.6f}' for v in (*t,*r))+'\n')
        f.write('end\n')
        if triangles:
            scene.frame_set(1);f.write('triangles\n')
            for o in MESHES:
                mesh=o.data;mesh.calc_loop_triangles()
                normal_matrix=o.matrix_world.to_3x3().inverted().transposed()
                bi=index[o.get('part_bone','body')]
                for tri in mesh.loop_triangles:
                    f.write('goblin_atlas\n')
                    for li in tri.loops:
                        loop=mesh.loops[li];v=o.matrix_world@mesh.vertices[loop.vertex_index].co*SCALE
                        n=(normal_matrix@loop.normal).normalized();uv=mesh.uv_layers.active.data[li].uv
                        f.write(f'{bi} '+' '.join(f'{v:.6f}' for v in (*v,*n,*uv))+f' 1 {bi} 1.0\n')
            f.write('end\n')
source=OUT/'source';source.mkdir(exist_ok=True)
write_smd(source/'goblin_reference.smd',triangles=True)
write_smd(source/'goblin_idle.smd')
write_smd(source/'goblin_walk.smd',animated=True)
scene.frame_set(1)
qc='''// Preview model; player integration uses a separate native skeleton export.
$modelname "goblin_codex/goblin_preview.mdl"
$body "body" "goblin_reference.smd"
$surfaceprop "metal"
$cdmaterials "models/goblin_codex/"
$sequence idle "goblin_idle.smd" fps 24 loop ACT_IDLE 1
$sequence walk "goblin_walk.smd" fps 24 loop ACT_WALK 1
$sequence run "goblin_walk.smd" fps 32 loop ACT_RUN 1
$attachment "eyes" "head" 0 0 12 rotate 0 0 0
$attachment "anim_attachment_RH" "hand.R" 0 0 0
$eyeposition 0 -8 54
'''
(source/'goblin_preview.qc').write_text(qc)
(source/'goblin_atlas.vmt').write_text('''"VertexLitGeneric"
{
 "$basetexture" "models/goblin_codex/goblin_atlas"
 "$model" "1"
 "$phong" "1"
 "$phongexponent" "16"
 "$phongboost" "0.35"
}
''')

# Studio lights, ground and three useful views of the real mesh.
def plain_mat(name,color,metal=0,rough=.5):
    m=bpy.data.materials.new(name);m.diffuse_color=(*color,1);m.use_nodes=True
    p=m.node_tree.nodes.get('Principled BSDF');p.inputs['Base Color'].default_value=(*color,1)
    p.inputs['Metallic'].default_value=metal;p.inputs['Roughness'].default_value=rough
    return m
bpy.ops.mesh.primitive_plane_add(size=200,location=(0,0,-.015))
floor=bpy.context.object;floor.name='STUDIO ground';floor.data.materials.append(plain_mat('Studio slate',(.032,.046,.050),.1,.48))
def aim(o,p):o.rotation_euler=(Vector(p)-o.location).to_track_quat('-Z','Y').to_euler()
def light(name,p,power,size,color):
    d=bpy.data.lights.new(name,'AREA');d.energy=power;d.shape='DISK';d.size=size;d.color=color
    o=bpy.data.objects.new(name,d);scene.collection.objects.link(o);o.location=p;aim(o,(0,0,3))
light('Key softbox',(-4,-6,8),1050,5,(.82,.91,1))
light('Warm fill',(4,-4,5),650,4,(1,.82,.62))
light('Cool rim',(2,3,7),1500,3,(.57,.79,1))
light('Face card',(-.5,-5,3.6),110,2,(1,1,.88))
world=bpy.data.worlds.new('Goblin studio world');world.use_nodes=True
world.node_tree.nodes['Background'].inputs[0].default_value=(.10,.14,.17,1)
world.node_tree.nodes['Background'].inputs[1].default_value=.35;scene.world=world
bpy.ops.object.camera_add(location=(8,-13,7.0))
cam=bpy.context.object;cam.name='Goblin portrait';cam.data.type='ORTHO';cam.data.ortho_scale=6.8;aim(cam,(0,0,2.87));scene.camera=cam
scene.render.engine='CYCLES';scene.cycles.samples=32;scene.cycles.use_denoising=True
scene.render.resolution_x=1000;scene.render.resolution_y=1200;scene.render.resolution_percentage=100
scene.view_settings.view_transform='AgX'
ATLAS.pack()
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'goblin.blend'))
stats={'mesh_objects':len(MESHES),'vertices':sum(len(o.data.vertices) for o in MESHES),'triangles':sum(len(o.data.loop_triangles) for o in MESHES),'bones':len(bones),'height_source_units':5.77*SCALE,'reference':'User-supplied goblin bot 2.webp','status':'Articulated model complete; see VALIDATION.md for runtime checks'}
(OUT/'model-stats.json').write_text(json.dumps(stats,indent=2))
for name,p,target,scale in [
 ('goblin-portrait.png',(8,-13,7),(0,0,2.87),6.8),
 ('goblin-front.png',(0,-14,5.5),(0,0,2.86),6.65),
 ('goblin-back.png',(-8,13,6.5),(0,0,2.85),6.8),
]:
    cam.location=p;aim(cam,target);cam.data.ortho_scale=scale
    scene.render.filepath=str(OUT/name);bpy.ops.render.render(write_still=True)
print('GOBLIN_BUILD_COMPLETE',json.dumps(stats))
