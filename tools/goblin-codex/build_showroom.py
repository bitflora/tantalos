"""Build a small sealed Portal 2 room for repeatable model inspection."""
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'out/goblin-codex'
GAME=Path(r'C:\Program Files (x86)\Steam\steamapps\common\Portal 2')
MOD=OUT/'game'
counter=1


def solid(lo,hi,material):
    global counter
    x,y,z=lo;X,Y,Z=hi
    planes=[[(x,y,z),(x,Y,z),(X,Y,z)],[(x,y,Z),(X,y,Z),(X,Y,Z)],
            [(x,y,z),(X,y,z),(X,y,Z)],[(x,Y,z),(x,Y,Z),(X,Y,Z)],
            [(x,y,z),(x,y,Z),(x,Y,Z)],[(X,y,z),(X,Y,z),(X,Y,Z)]]
    counter+=1;s=f'solid\n{{\n"id" "{counter}"\n'
    for face,pts in enumerate(planes):
        counter+=1;plane=' '.join('('+ ' '.join(map(str,p))+')' for p in reversed(pts))
        u='[0 1 0 0]' if face in (4,5) else '[1 0 0 0]'
        v='[0 -1 0 0]' if face in (0,1) else '[0 0 -1 0]'
        s+=f'side\n{{\n"id" "{counter}"\n"plane" "{plane}"\n"material" "{material}"\n"uaxis" "{u} 0.25"\n"vaxis" "{v} 0.25"\n"rotation" "0"\n"lightmapscale" "16"\n"smoothing_groups" "0"\n}}\n'
    return s+'}\n'


def entity(**fields):
    global counter
    counter+=1
    return 'entity\n{\n'+f'"id" "{counter}"\n'+''.join(f'"{k}" "{v}"\n' for k,v in fields.items())+'}\n'


def main():
    vmf=OUT/'goblin_showroom.vmf'
    text='versioninfo\n{\n"editorversion" "400"\n"editorbuild" "0"\n"mapversion" "1"\n"formatversion" "100"\n"prefab" "0"\n}\nworld\n{\n"id" "1"\n"classname" "worldspawn"\n"skyname" "sky_black_nofog"\n'
    for lo,hi in [((-384,-384,-16),(384,384,0)),((-384,-384,256),(384,384,272)),((-400,-400,-16),(-384,400,272)),((384,-400,-16),(400,400,272)),((-384,-400,-16),(384,-384,272)),((-384,384,-16),(384,400,272))]:
        material='tile/white_floor_tile002a' if hi[2]<=0 else 'tile/white_wall_tile003a'
        text+=solid(lo,hi,material)
    text+='}\n'
    text+=entity(classname='info_player_start',origin='-100 0 1',angles='0 0 0')
    text+=entity(classname='prop_dynamic',targetname='goblin_display',model='models/info_character/info_character_player.mdl',origin='80 0 0',angles='0 180 0',DefaultAnim='idle',solid='0')
    for x,y in [(-150,-150),(-150,150),(150,-150),(150,150)]:text+=entity(classname='light',origin=f'{x} {y} 200',_light='220 233 255 75',_lightHDR='-1 -1 -1 1')
    vmf.write_text(text)
    # The SDK map compilers use their game installation's content workspace.
    # Authoritative source and the final BSP remain in this Git worktree.
    scratch=GAME/'sdk_content/maps/goblin_codex_worktree'
    scratch.mkdir(parents=True,exist_ok=True)
    compiled_vmf=scratch/vmf.name
    shutil.copy2(vmf,compiled_vmf)
    for tool,args in [('vbsp',[]),('vvis',['-fast']),('vrad',['-fast','-ldr'])]:
        result=subprocess.run([str(GAME/'bin'/f'{tool}.exe'),'-threads','4','-game',str(GAME/'portal2'),*args,str(compiled_vmf)],cwd=scratch,capture_output=True,text=True)
        (OUT/f'showroom-{tool}.log').write_text(result.stdout+result.stderr)
        print(tool,result.returncode,result.stdout[-1000:])
        if result.returncode:raise SystemExit(result.returncode)
    target=MOD/'maps/goblin_codex/showroom.bsp';target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(compiled_vmf.with_suffix('.bsp'),target)
    if (OUT/'packed-game/gameinfo.txt').exists():
        packed=OUT/'packed-game/maps/goblin_codex/showroom.bsp';packed.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(target,packed)
    print('SHOWROOM_READY',target)


if __name__=='__main__':main()
