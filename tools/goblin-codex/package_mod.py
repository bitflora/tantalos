"""Compile and package Goblin in an isolated Portal 2 mod directory.

Never writes into the installed game. Uses the installed Valve compiler and
mounted game assets, and creates a self-contained VPK containing our own assets.
"""
import argparse
import json
import shutil
import struct
import subprocess
import zlib
import io
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
ASSETS=ROOT/'assets/goblin-codex'
OUT=ROOT/'out/goblin-codex'
GAME=Path(r'C:\Program Files (x86)\Steam\steamapps\common\Portal 2')


def write_vtf(png, dest):
    image=Image.open(png).convert('RGBA')
    w,h=image.size
    levels=[]
    current=image
    while True:
        encoded=io.BytesIO()
        # A BC1 block is at least 4x4, including the final tiny mip levels.
        block_image=current.resize((max(4,current.width),max(4,current.height)),Image.Resampling.LANCZOS)
        block_image.save(encoded,format='DDS',pixel_format='DXT1')
        payload=encoded.getvalue()[128:]
        expected=max(1,(current.width+3)//4)*max(1,(current.height+3)//4)*8
        assert len(payload)==expected,(current.size,len(payload),expected)
        levels.append(payload)
        if current.size==(1,1):break
        current=current.resize((max(1,current.width//2),max(1,current.height//2)),Image.Resampling.LANCZOS)
    header=bytearray(80)
    struct.pack_into('<4sIIIHHIHH',header,0,b'VTF\0',7,2,80,w,h,0x2,1,0)
    struct.pack_into('<3f',header,32,.3,.3,.3)
    struct.pack_into('<f',header,48,1.0)
    struct.pack_into('<IBIBBH',header,52,13,len(levels),0xFFFFFFFF,0,0,1)
    dest.parent.mkdir(parents=True,exist_ok=True)
    # VTF stores the complete mip pyramid from smallest to largest.
    dest.write_bytes(header+b''.join(reversed(levels)))


def vpk(files,dest):
    groups={}
    for name,data in sorted(files.items()):
        p=Path(name)
        groups.setdefault(p.suffix[1:],{}).setdefault(p.parent.as_posix(),[]).append((p.stem,data))
    tree=bytearray();blob=bytearray()
    for ext,folders in groups.items():
        tree+=ext.encode()+b'\0'
        for folder,items in folders.items():
            tree+=folder.encode()+b'\0'
            for name,data in items:
                tree+=name.encode()+b'\0'
                tree+=struct.pack('<IHHIIH',zlib.crc32(data),0,0x7FFF,len(blob),len(data),0xFFFF)
                blob+=data
            tree+=b'\0'
        tree+=b'\0'
    tree+=b'\0'
    dest.write_bytes(struct.pack('<3I',0x55AA1234,1,len(tree))+tree+blob)


def packed_runtime():
    """Mount the actual deliverable VPK ahead of loose compiler outputs."""
    mod=OUT/'game'
    packed=OUT/'packed-game';packed.mkdir(parents=True,exist_ok=True)
    info=(mod/'gameinfo.txt').read_text()
    old=f' Game "{mod.as_posix()}"\n'
    new=f' Game "{packed.as_posix()}"\n'
    assert old in info
    (packed/'gameinfo.txt').write_text(info.replace(old,new,1))
    # Portal 2 automatically mounts pak01_dir.vpk in a Game search directory.
    # There is deliberately no loose-model search path in this runtime.
    shutil.copy2(ASSETS/'goblin-codex.vpk',packed/'pak01_dir.vpk')
    for relative in ('maps/goblin_codex/showroom.bsp','cfg/goblin_codex_preview.cfg'):
        src=mod/relative
        if src.exists():
            target=packed/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)
    print('PACKED_RUNTIME',packed)
    return packed


def main():
    p=argparse.ArgumentParser();p.add_argument('--game',type=Path,default=GAME);p.add_argument('--runtime-only',action='store_true');p.add_argument('--repack-only',action='store_true');args=p.parse_args()
    if args.runtime_only:packed_runtime();return
    game=args.game
    mod=OUT/'game';mod.mkdir(parents=True,exist_ok=True)
    paths=[mod,game/'update',game/'portal2_dlc2',game/'portal2_dlc1',game/'portal2']
    info='"GameInfo"\n{\n game "Goblin Codex"\n title "GOBLIN"\n GameData "portal2.fgd"\n FileSystem\n {\n SteamAppId 620\n ToolsAppId 629\n SearchPaths\n {\n'
    info+=f' GameBin "{(game/"portal2/bin").as_posix()}"\n'
    info+=''.join(f' Game "{path.as_posix()}"\n' for path in paths)
    info+=' }\n }\n}\n'
    (mod/'gameinfo.txt').write_text(info)
    materials=mod/'materials/models/goblin_codex';materials.mkdir(parents=True,exist_ok=True)
    write_vtf(ASSETS/'goblin_atlas.png',materials/'goblin_atlas.vtf')
    shutil.copy2(ASSETS/'source/goblin_atlas.vmt',materials/'goblin_atlas.vmt')
    # Validate with Valve's decoder, not only our own header assumptions.
    decoded=OUT/'atlas-valve-decoded.tga'
    decode=subprocess.run([str(game/'bin/vtf2tga.exe'),'-i',str(materials/'goblin_atlas.vtf'),'-o',str(decoded)],capture_output=True,text=True)
    if decode.returncode or not decoded.exists():raise SystemExit('Valve VTF decode failed')
    import numpy as np
    expected=np.array(Image.open(ASSETS/'goblin_atlas.png').convert('RGB')).astype('float32')
    actual=np.array(Image.open(decoded).convert('RGB')).astype('float32')
    assert actual.shape==expected.shape
    error=np.abs(actual-expected).mean(axis=(0,1)).tolist()
    assert max(error)<8,error
    compiler=game/'bin/studiomdl.exe'
    if not compiler.exists():raise SystemExit('Install Portal 2 Authoring Tools (Steam app 629).')
    for qc in (() if args.repack_only else ('goblin_player.qc','goblin_preview.qc')):
        result=subprocess.run([str(compiler),'-nop4','-game',str(mod),str(ASSETS/'source'/qc)],cwd=ASSETS/'source',capture_output=True,text=True)
        (OUT/(qc+'.log')).write_text(result.stdout+result.stderr)
        print(result.stdout[-2800:]);print(result.stderr)
        if result.returncode or 'ERROR:' in result.stdout:raise SystemExit('Compilation failed; see '+str(OUT/(qc+'.log')))
    # Updated Portal 2 installations prefer .vtx; older branches use .dx90.vtx.
    # Without the alias, a replacement can accidentally pair our MDL with the
    # stock model's index file, causing an engine checksum mismatch.
    for index_file in (mod/'models').rglob('*.dx90.vtx'):
        shutil.copy2(index_file,index_file.with_name(index_file.name.replace('.dx90.vtx','.vtx')))
    files={p.relative_to(mod).as_posix():p.read_bytes() for folder in ('models','materials') for p in (mod/folder).rglob('*') if p.is_file()}
    for name,data in files.items():
        if not name.endswith('.mdl'):continue
        checksum=struct.unpack_from('<I',data,8)[0]
        stem=name[:-4]
        assert struct.unpack_from('<I',files[stem+'.vvd'],8)[0]==checksum
        for suffix in ('.vtx','.dx90.vtx','.dx80.vtx','.sw.vtx'):
            assert struct.unpack_from('<I',files[stem+suffix],16)[0]==checksum,(name,suffix)
    package=ASSETS/'goblin-codex.vpk';vpk(files,package)
    # Roundtrip every packed byte and CRC through our independent VPK reader.
    from source_assets import VPK
    packed=VPK(package)
    assert set(packed.entries)==set(files)
    for name,data in files.items():assert packed.read(name)==data,name
    cfg=mod/'cfg';cfg.mkdir(exist_ok=True)
    (cfg/'goblin_codex_preview.cfg').write_text('''sv_cheats 1
prop_dynamic_create goblin_codex/goblin_preview
setmodel goblin_codex/goblin_preview
thirdperson
cam_idealdist 110
cam_idealyaw 160
cam_idealpitch 8
''')
    manifest={'package':str(package),'mod_directory':str(mod),'files':{k:len(v) for k,v in files.items()},'base_game_assets_untouched':True,'vpk_roundtrip_verified':True,'model_component_checksums_verified':True,'valve_texture_decode_verified':True,'texture_mean_abs_error_rgb':error,'texture_format':'DXT1, full 12-level mip pyramid'}
    manifest['runtime_directory']=str(packed_runtime())
    (ASSETS/'package-manifest.json').write_text(json.dumps(manifest,indent=2))
    print('PACKAGED',package,package.stat().st_size)
    print('ISOLATED_GAME',mod)


if __name__=='__main__':main()
