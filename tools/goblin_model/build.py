"""Builds the Goblin bot player model and installs it into Portal 2 in place of Chell.

    python tools/goblin_model/build.py build [--skip-blender]   # out/goblin_model/, nothing installed
    python tools/goblin_model/build.py install                  # build's result -> Portal 2/portal2_dlc3
    python tools/goblin_model/build.py uninstall [--sol]        # Chell again (--sol: only Sol's)
    python tools/goblin_model/build.py shoot [--sol]            # screenshots of it in the game
    python tools/goblin_model/build.py install-sol [VPK]        # Sol's goblin too -> portal2_dlc4

Sol (another agent, ~/.codex/worktrees/goblin-model-codex) built a goblin that replaces
models/info_character/info_character_player.mdl instead, with its own rig. install-sol puts its
pack beside ours, so one game has both: in the console, exec goblin_sol (Sol's) or exec goblin_mine
(ours). A bare setmodel to Sol's model crashes the game unless the map has precached it
("UTIL_SetModel: not precached"), so goblin_sol spawns and removes a prop of it first.

The model is made by goblin_blender.py (portable Blender 4.2, BLENDER or
%LOCALAPPDATA%/Programs/Blender/blender-4.2.3-windows-x64), rigidly skinned to Chell's skeleton so
her animations (models/player_animations.mdl, $includemodel) drive it, and compiled with the
Portal 2 Authoring Tools' studiomdl and vtex. The base game is never written to: the model goes
into its own folder, portal2_dlc3/pak01_dir.vpk, which the game mounts after its own DLC
folders; uninstall deletes that folder.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "train"))
import mdl_info  # noqa: E402
import vpk  # noqa: E402

GAME = Path(r"C:\Program Files (x86)\Steam\steamapps\common\Portal 2")
BIN = GAME / "bin"
WORK = ROOT / "out" / "goblin_model"
STAGE = WORK / "game"           # studiomdl/vtex game dir: mounts portal2 behind it
MOD = GAME / "portal2_dlc3"
MARK = "glados-goblin"          # in the VPK, so uninstall only removes our folder
SOL_MOD = GAME / "portal2_dlc4"  # the game mounts dlcN folders in sequence, so it needs dlc3
SOL_MARK = "glados-goblin-sol"
SOL_VPK = Path.home() / ".codex/worktrees/goblin-model-codex/glados/assets/goblin-codex/goblin-codex.vpk"
SOL_MODEL = "info_character/info_character_player"
BLENDER = Path(os.environ.get("BLENDER", Path(os.environ["LOCALAPPDATA"]) / "Programs" / "Blender" /
                              "blender-4.2.3-windows-x64" / "blender.exe"))
CHELL = "models/player/chell/player.mdl"
MAT_DIR = "models/goblin"


def run(cmd, **kw):
    print(">", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, errors="replace", **kw)
    if r.returncode != 0:
        print(r.stdout[-4000:], r.stderr[-4000:])
        raise SystemExit(f"{Path(str(cmd[0])).name} failed ({r.returncode})")
    return r.stdout


def chell_skeleton():
    src = WORK / "chell"
    mdl = src / CHELL
    if not mdl.exists():
        d = str(GAME / "portal2" / "pak01_dir.vpk")
        entries = vpk.read_dir(d)
        mdl.parent.mkdir(parents=True, exist_ok=True)
        mdl.write_bytes(vpk.read_file(d, entries, CHELL))
    info = mdl_info.parse(str(mdl))
    with open(WORK / "chell.json", "w") as f:
        json.dump(info, f)
    return info


def textures(blend_out):
    src = STAGE / "materialsrc" / MAT_DIR
    src.mkdir(parents=True, exist_ok=True)
    color = np.asarray(Image.open(blend_out / "goblin_color.png").convert("RGB"))
    emit = np.asarray(Image.open(blend_out / "goblin_emit.png").convert("L")).astype(np.float32)
    alpha = np.clip(emit * 2.2, 0, 255).astype(np.uint8)    # $selfillum mask: the eyes
    Image.fromarray(np.dstack([color, alpha]), "RGBA").save(src / "goblin_color.tga")
    Image.open(blend_out / "goblin_normal.png").convert("RGB").save(src / "goblin_normal.tga")
    (src / "goblin_normal.txt").write_text("normal 1\n")
    for name in ("goblin_color", "goblin_normal"):
        run([BIN / "vtex.exe", "-nopause", "-quiet", "-game", STAGE, src / f"{name}.tga"])
    out = STAGE / "materials" / MAT_DIR
    (out / "goblin.vmt").write_text(f'''"VertexLitGeneric"
{{
	"$basetexture" "{MAT_DIR}/goblin_color"
	"$bumpmap" "{MAT_DIR}/goblin_normal"
	"$selfillum" "1"
	"$phong" "1"
	"$phongexponent" "16"
	"$phongboost" "1.2"
	"$phongfresnelranges" "[0.25 0.6 1]"
	"$surfaceprop" "metal"
}}
''')


def used_bones(info, smd):
    """Bones the mesh is skinned to, and their ancestors: the ones studiomdl keeps."""
    used = set()
    with open(smd) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 12:
                used.update(int(parts[i]) for i in range(10, len(parts), 2))
    for i in list(used):
        while i >= 0:
            used.add(i)
            i = info["bones"][i]["parent"]
    return used


def qc(info, used):
    bones = info["bones"]
    lines = [f'$modelname "{CHELL[len("models/"):]}"',
             f'$cdmaterials "{MAT_DIR}/"',
             '$body "goblin" "goblin.smd"',
             '$surfaceprop "metal"',
             '$eyeposition 0 0 70',
             '$illumposition 0 0 37',
             '$maxverts 65000',
             '$includemodel "player_animations.mdl"',
             '$bonemerge "weapon_bone"']
    for a in info["attachments"]:
        m = a["matrix"]
        pitch = math.degrees(math.atan2(-m[2][0], math.hypot(m[0][0], m[1][0])))
        yaw = math.degrees(math.atan2(m[1][0], m[0][0]))
        roll = math.degrees(math.atan2(m[2][1], m[2][2]))
        lines.append(f'$attachment "{a["name"]}" "{bones[a["bone"]]["name"]}" {m[0][3]:.4f} {m[1][3]:.4f} '
                     f'{m[2][3]:.4f} rotate {pitch:.4f} {yaw:.4f} {roll:.4f}')
    for s in info["hitboxsets"]:
        lines.append(f'$hboxset "{s["name"]}"')
        for h in s["boxes"]:
            if h["bone"] not in used:
                continue
            lines.append(f'$hbox {h["group"]} "{bones[h["bone"]]["name"]}" ' +
                         " ".join(f"{x:.4f}" for x in h["min"] + h["max"]))
    return "\n".join(lines) + "\n"


def compare(info, built):
    """Bind pose of each bone in the compiled model against Chell's (inches, max abs)."""
    want = {b["name"]: np.linalg.inv(np.array(b["pose_to_bone"] + [[0, 0, 0, 1]])) for b in info["bones"]}
    worst = 0.0
    for b in built["bones"]:
        got = np.linalg.inv(np.array(b["pose_to_bone"] + [[0, 0, 0, 1]]))
        worst = max(worst, float(np.abs(got - want[b["name"]]).max()))
    return worst


def build(skip_blender):
    WORK.mkdir(parents=True, exist_ok=True)
    info = chell_skeleton()
    blend_out = WORK / "blender"
    if not skip_blender:
        run([BLENDER, "-b", "--factory-startup", "-P", HERE / "goblin_blender.py", "--", WORK / "chell.json", blend_out])
    STAGE.mkdir(parents=True, exist_ok=True)
    (STAGE / "gameinfo.txt").write_text(f'''"GameInfo"
{{
	game "glados goblin staging"
	FileSystem
	{{
		SteamAppId 620
		ToolsAppId 211
		SearchPaths
		{{
			Game |gameinfo_path|.
			Game "{(GAME / "portal2").as_posix()}"
		}}
	}}
}}
''')
    textures(blend_out)
    src = WORK / "src"
    src.mkdir(exist_ok=True)
    shutil.copy(blend_out / "goblin.smd", src / "goblin.smd")
    (src / "goblin.qc").write_text(qc(info, used_bones(info, src / "goblin.smd")))
    out = run([BIN / "studiomdl.exe", "-game", STAGE, "-nop4", "-nox360", src / "goblin.qc"])
    (WORK / "studiomdl.log").write_text(out)
    for line in out.splitlines():
        if "warning" in line.lower() or "error" in line.lower():
            print("  studiomdl:", line)
    built = mdl_info.parse(str(STAGE / CHELL))
    worst = compare(info, built)
    print(f"bones {len(built['bones'])}, attachments {len(built['attachments'])}, "
          f"include {built['includemodels']}, bind pose off Chell's by {worst:.4f} at most")
    if worst > 0.01:
        raise SystemExit("the compiled skeleton doesn't match Chell's: her animations would break it")
    pack = WORK / "pack"
    if pack.exists():
        shutil.rmtree(pack)
    for rel in ("models/player/chell", f"materials/{MAT_DIR}"):
        dst = pack / rel
        dst.mkdir(parents=True)
        for f in (STAGE / rel).iterdir():
            if f.name.startswith("player.") or rel.startswith("materials"):
                shutil.copy(f, dst / f.name)
    # Portal 2 also reads plain .vtx, and the base game has Chell's: ours must shadow it
    shutil.copy(pack / "models/player/chell/player.dx90.vtx", pack / "models/player/chell/player.vtx")
    (pack / "scripts").mkdir()
    (pack / "scripts" / f"{MARK}.txt").write_text("Goblin bot player model, from glados tools/goblin_model\n")
    print(vpk.pack(str(pack), str(WORK / "pak01_dir.vpk")), "files in", WORK / "pak01_dir.vpk")
    # the game reads VPKs before any loose folder, so the staging game (shoot) needs it as one too
    shutil.copy(WORK / "pak01_dir.vpk", STAGE / "pak01_dir.vpk")


def ours(folder, mark=MARK):
    """Our VPK and nothing else, except the sound cache the game writes into every search folder."""
    vp = folder / "pak01_dir.vpk"
    others = [p for p in folder.rglob("*") if p.is_file() and p != vp
              and p.relative_to(folder).as_posix() != "maps/soundcache/_master.cache"]
    return vp.exists() and f"scripts/{mark}.txt" in vpk.read_dir(str(vp)) and not others


def install():
    if not (WORK / "pak01_dir.vpk").exists():
        raise SystemExit("build first")
    if MOD.exists() and not ours(MOD):
        raise SystemExit(f"{MOD} exists and isn't ours; not touching it")
    MOD.mkdir(exist_ok=True)
    shutil.copy(WORK / "pak01_dir.vpk", MOD / "pak01_dir.vpk")
    print("installed", MOD / "pak01_dir.vpk", "(takes effect the next time Portal 2 starts)")


def install_sol(src):
    """Sol's pack, unchanged but for our marker, into portal2_dlc4 (after ours, which it needs)."""
    if not (MOD.exists() and ours(MOD)):
        raise SystemExit(f"install ours first: the game only mounts {SOL_MOD.name} after {MOD.name}")
    if SOL_MOD.exists() and not ours(SOL_MOD, SOL_MARK):
        raise SystemExit(f"{SOL_MOD} exists and isn't ours; not touching it")
    entries = vpk.read_dir(str(src))
    if f"models/{SOL_MODEL}.mdl" not in entries:
        raise SystemExit(f"{src} has no models/{SOL_MODEL}.mdl")
    pack = WORK / "sol-pack"
    shutil.rmtree(pack, ignore_errors=True)
    for name in entries:
        (pack / name).parent.mkdir(parents=True, exist_ok=True)
        (pack / name).write_bytes(vpk.read_file(str(src), entries, name))
    (pack / "scripts").mkdir(exist_ok=True)
    (pack / "scripts" / f"{SOL_MARK}.txt").write_text(f"Sol's goblin player model, from {src}\n")
    (pack / "cfg").mkdir(exist_ok=True)
    (pack / "cfg" / "goblin_sol.cfg").write_text(
        "sv_cheats 1\n"
        f"prop_dynamic_create {SOL_MODEL}\n"   # precaches it (VScript has no PrecacheModel here)
        f'script Entities.FindByModel(null, "models/{SOL_MODEL}.mdl").Destroy()\n'
        f"setmodel {SOL_MODEL}\n")
    (pack / "cfg" / "goblin_mine.cfg").write_text("sv_cheats 1\nsetmodel player/chell/player\n")
    print(vpk.pack(str(pack), str(WORK / "sol-pak01_dir.vpk")), "files")
    SOL_MOD.mkdir(exist_ok=True)
    shutil.copy(WORK / "sol-pak01_dir.vpk", SOL_MOD / "pak01_dir.vpk")
    print("installed", SOL_MOD / "pak01_dir.vpk")


def uninstall(sol_only):
    for folder, mark in [(SOL_MOD, SOL_MARK)] + ([] if sol_only else [(MOD, MARK)]):
        if not folder.exists():
            print("not installed:", folder)
        elif not ours(folder, mark):
            raise SystemExit(f"{folder} isn't ours; not touching it")
        else:
            shutil.rmtree(folder)
            print("removed", folder)


def shoot(map_name, installed, sol=False):
    """Starts Portal 2 on a stock map, shows the player in third person and screenshots it: from
    the staging folder (-game, nothing installed), or the installed game with --installed; --sol
    switches to Sol's model first (install-sol), to check that switching works."""
    import portal2
    import secrets
    if portal2.game_process() is not None:
        raise SystemExit("Portal 2 is already running; not touching it")
    port, password = portal2.free_port(), secrets.token_hex(8)
    game_dir = "portal2" if installed else str(STAGE)
    subprocess.Popen([str(portal2.EXE), "-game", game_dir, "-steam", "-insecure", "-novid", "-windowed", "-w", "1280",
                      "-h", "800", "-netconport", str(port), "-netconpassword", password], cwd=GAME)
    con = portal2.Console(port, password, 180)
    pid = portal2.game_process()[0]
    shots = (GAME / "portal2" if installed else STAGE) / "screenshots"
    before = set(shots.glob("*")) if shots.exists() else set()
    try:
        end = time.time() + 180
        while True:
            con.send(f"map {map_name}")
            if con.until(lambda l: f"Host_NewGame on map {map_name}" in l, 10) is not None:
                break
            if time.time() > end:
                raise SystemExit("the map didn't start loading")
        con.until(lambda l: "END loading assets" in l, 180)
        time.sleep(8)
        log = con.ask(["sv_cheats 1", "developer 1", "mat_queue_mode 0"], 10) or ""
        tag = "sol-" if sol else ""
        if sol:
            log += con.ask(["exec goblin_sol"], 10) or ""
            time.sleep(2)
        views = [("front", 180, 0), ("front34", 150, 10), ("side", 90, 5), ("back34", 30, 10)]
        con.ask(["thirdperson", "cam_snapto 1", "cam_collision 0", "cam_idealdist 110", "cam_idealpitch 8",
                 "hidehud 4", "r_drawviewmodel 0", "crosshair 0"], 10)
        time.sleep(2)
        for name, yaw, pitch in views:
            con.ask([f"cam_idealyaw {yaw}", f"cam_idealpitch {pitch}"], 5)
            time.sleep(2.5)
            con.ask([f"jpeg goblin-{tag}{name} 95"], 10)
            time.sleep(1.5)
        con.ask(["+forward"], 5)
        time.sleep(0.6)
        con.ask(["cam_idealyaw 140", f"jpeg goblin-{tag}walk 95"], 10)
        time.sleep(1.0)
        con.ask(["-forward", "+jump"], 5)
        time.sleep(0.25)
        con.ask([f"jpeg goblin-{tag}jump 95", "-jump"], 10)
        time.sleep(2)
        con.ask(["+duck"], 5)
        time.sleep(1.2)
        con.ask(["cam_idealyaw 160", f"jpeg goblin-{tag}crouch 95", "-duck"], 10)
        time.sleep(2)
        print("\n".join(l for l in log.splitlines() if any(k in l.lower() for k in ("goblin", "player.mdl", "info_character", "precache"))))
    finally:
        try:
            con.send("quit")
        except portal2.GameUnavailable:
            print("the game went away (crashed?)")
        end = time.time() + 60
        while time.time() < end and portal2.alive(pid):
            time.sleep(1)
        if portal2.alive(pid):
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
        con.close()
    dest = WORK / "shots"
    dest.mkdir(parents=True, exist_ok=True)
    for f in sorted(set(shots.glob("goblin-*")) - before):
        shutil.move(str(f), dest / f.name)
        print(dest / f.name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--skip-blender", action="store_true", help="reuse out/goblin_model/blender")
    sub.add_parser("install")
    i = sub.add_parser("install-sol")
    i.add_argument("vpk", nargs="?", type=Path, default=SOL_VPK)
    u = sub.add_parser("uninstall")
    u.add_argument("--sol", action="store_true", help="only Sol's model; ours stays")
    s = sub.add_parser("shoot")
    s.add_argument("--map", default="sp_a1_intro3")
    s.add_argument("--installed", action="store_true", help="the installed game, not the staging folder")
    s.add_argument("--sol", action="store_true", help="setmodel to Sol's model first")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.skip_blender)
    elif a.cmd == "install":
        install()
    elif a.cmd == "install-sol":
        install_sol(a.vpk)
    elif a.cmd == "uninstall":
        uninstall(a.sol)
    else:
        shoot(a.map, a.installed, a.sol)


if __name__ == "__main__":
    main()
