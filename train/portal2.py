"""Ask Portal 2's own Puzzle Maker whether .p2c files are valid.

The editor refuses to export a puzzle with misplaced items and logs each one as
`INVALID ITEM:  <TargetName> ( x , y , z )` (x y z = the item's VoxelPos). There is no console
command that loads a file in the retail game: `puzzlemaker_load_dev` calls a stub. `patch`
rewrites one byte in the running game so it calls the loader the My Chambers list uses, and
sets the editor's active flag, which `puzzlemaker_export` checks. Commands reach the game
through `portal2.exe -hijack`, and its console output through `con_logfile`.

Memory is only patched in a game started with -insecure (VAC off for that session), and the
patch refuses any client.dll that doesn't match the build it was written for (Jan 2025).

    python train/portal2.py check out/ingame/sample/p2c --out results.jsonl --launch
    python train/portal2.py patch [--undo]       # a game you started with -insecure

`--launch` starts the game (refusing if it's already running), opens the editor, patches it,
checks, quits, and puts back the editor autosaves it overwrote. Exits 3 if the game couldn't
be used, so callers can carry on without it.
"""
import argparse, ctypes, ctypes.wintypes as wt, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

GAME = Path(r"C:\Program Files (x86)\Steam\steamapps\common\Portal 2")
EXE = GAME / "portal2.exe"
LOGNAME = "glados_console.log"
LOG = GAME / "portal2" / LOGNAME
CFG = GAME / "portal2" / "cfg" / "glados_step.cfg"
STAGE = GAME / "portal2" / "puzzles" / "glados_check"
VMF = GAME / "sdk_content" / "maps" / "glados_check.vmf"
PREVIEW = GAME / "portal2" / "maps" / "puzzlemaker" / "preview.bsp"
INVALID = re.compile(r"INVALID ITEM:\s+(\S+)\s+\(\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")
UNAVAILABLE = 3


class GameUnavailable(Exception):
    pass


# --- the process ---------------------------------------------------------------------------

def game_process():
    """(pid, command line) of the running portal2.exe, or None."""
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process -Filter \"Name='portal2.exe'\" | "
         "ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }"],
        capture_output=True, text=True).stdout.strip().splitlines()
    if not out:
        return None
    if len(out) > 1:
        raise GameUnavailable("more than one portal2.exe running")
    pid, cmd = out[0].split("\t", 1)
    return int(pid), cmd


def alive(pid):
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = wt.HANDLE
    k32.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
    h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = wt.DWORD()
    k32.GetExitCodeProcess(wt.HANDLE(h), ctypes.byref(code))
    k32.CloseHandle(wt.HANDLE(h))
    return code.value == 259  # STILL_ACTIVE


def send(cmds, pid=None):
    """Run console commands in the running game (via a cfg, so paths need no escaping). With
    no game window, `-hijack` would start a new (VAC-secured) game, so `pid` must be alive."""
    if pid is not None and not alive(pid):
        raise GameUnavailable("the game exited")
    CFG.write_text("\n".join(cmds) + "\n", encoding="ascii")
    subprocess.run([str(EXE), "-hijack", "+exec", CFG.stem],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


class Log:
    """Reads the console log forward from where it was when created."""

    def __init__(self):
        self.pos = LOG.stat().st_size if LOG.exists() else 0

    def until(self, marker, timeout, poke=None):
        """Text logged since the last read, up to `marker`, or None on timeout. `poke` is
        called every few seconds while waiting (to resend a command the game missed)."""
        end = time.time() + timeout
        next_poke = time.time()
        while time.time() < end:
            if poke and time.time() >= next_poke:
                poke()
                next_poke = time.time() + 5
            if LOG.exists():
                with open(LOG, "rb") as f:
                    f.seek(self.pos)
                    text = f.read().decode("utf-8", "replace")
                i = text.find(marker)
                if i >= 0:
                    self.pos += len(text[: i + len(marker)].encode("utf-8"))
                    return text[:i]
            time.sleep(0.05)
        return None


# --- the patch -----------------------------------------------------------------------------

# In the puzzlemaker_load_dev handler: mov ecx,[g_pPuzzleMaker]; mov edx,[ecx]; push eax;
# mov eax,[edx+0xbc]; call eax; pop ebp; ret. Slot 0xbc is the stub; 0x4c(const char *path)
# is the loader. Bytes 2..5 are the relocated address of g_pPuzzleMaker.
HANDLER_RVA = 0x3597A1
PATTERN = bytes.fromhex("8b0d00000000" "8b11" "50" "8b82bc000000" "ffd0" "5dc3")
SLOT_AT = 11
PUZZLEMAKER_PTR_RVA = 0x94EAE8
ACTIVE_OFFSET = 5  # CPuzzleMaker::SetActive (slot 0x60) sets this byte; export checks it


def _win():
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    k32.OpenProcess.restype = wt.HANDLE
    k32.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
    rw = [wt.HANDLE, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
    k32.ReadProcessMemory.argtypes = k32.WriteProcessMemory.argtypes = rw
    k32.VirtualProtectEx.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.c_size_t, wt.DWORD, ctypes.POINTER(wt.DWORD)]
    k32.FlushInstructionCache.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.c_size_t]
    k32.CloseHandle.argtypes = [wt.HANDLE]
    psapi.EnumProcessModulesEx.argtypes = [wt.HANDLE, ctypes.POINTER(ctypes.c_void_p), wt.DWORD, ctypes.POINTER(wt.DWORD), wt.DWORD]
    psapi.GetModuleBaseNameW.argtypes = [wt.HANDLE, ctypes.c_void_p, wt.LPWSTR, wt.DWORD]
    return k32, psapi


def patch(undo=False):
    proc = game_process()
    if proc is None:
        raise GameUnavailable("Portal 2 isn't running")
    pid, cmd = proc
    if "-insecure" not in cmd.split():
        raise GameUnavailable("Portal 2 wasn't started with -insecure; not touching a VAC-secured game")
    k32, psapi = _win()
    h = k32.OpenProcess(0x0008 | 0x0010 | 0x0020 | 0x0400, False, pid)
    if not h:
        raise GameUnavailable(f"OpenProcess failed: {ctypes.get_last_error()}")
    try:
        mods = (ctypes.c_void_p * 1024)()
        need = wt.DWORD()
        psapi.EnumProcessModulesEx(h, mods, ctypes.sizeof(mods), ctypes.byref(need), 0x03)
        name = ctypes.create_unicode_buffer(260)
        base = None
        for m in mods[: need.value // ctypes.sizeof(ctypes.c_void_p)]:
            psapi.GetModuleBaseNameW(h, m, name, 260)
            if name.value.lower() == "client.dll":
                base = m
        if base is None:
            raise GameUnavailable("client.dll isn't loaded yet")
        n = ctypes.c_size_t()

        def read(addr, size):
            buf = (ctypes.c_ubyte * size)()
            if not k32.ReadProcessMemory(h, addr, buf, size, ctypes.byref(n)):
                raise GameUnavailable(f"ReadProcessMemory failed: {ctypes.get_last_error()}")
            return bytes(buf)

        def write(addr, value, code=False):
            old = wt.DWORD()
            if code:
                k32.VirtualProtectEx(h, addr, 1, 0x40, ctypes.byref(old))
            ok = k32.WriteProcessMemory(h, addr, ctypes.byref(ctypes.c_ubyte(value)), 1, ctypes.byref(n))
            if code:
                k32.VirtualProtectEx(h, addr, 1, old.value, ctypes.byref(old))
                k32.FlushInstructionCache(h, addr, 1)
            if not ok:
                raise GameUnavailable(f"WriteProcessMemory failed: {ctypes.get_last_error()}")

        addr = base + HANDLER_RVA
        got = read(addr, len(PATTERN))
        same = all(got[i] == PATTERN[i] for i in range(len(PATTERN)) if i not in (2, 3, 4, 5, SLOT_AT))
        global_ok = int.from_bytes(got[2:6], "little") == base + PUZZLEMAKER_PTR_RVA
        if not same or not global_ok or got[SLOT_AT] not in (0xBC, 0x4C):
            raise GameUnavailable(f"client.dll isn't the build this patch knows ({got.hex(' ')})")
        want = 0xBC if undo else 0x4C
        if got[SLOT_AT] != want:
            write(addr + SLOT_AT, want, code=True)
        if not undo:
            obj = int.from_bytes(read(base + PUZZLEMAKER_PTR_RVA, 4), "little")
            if not obj:
                raise GameUnavailable("the puzzle maker hasn't been created yet")
            write(obj + ACTIVE_OFFSET, 1)
    finally:
        k32.CloseHandle(h)


# --- a session -----------------------------------------------------------------------------

def autosaves():
    return [p for p in (GAME / "portal2" / "puzzles").rglob("autosave.*") if STAGE not in p.parents]


def launch(log, saved, timeout=180):
    """Start the game, open the editor and patch it. Returns the game's pid."""
    if game_process() is not None:
        raise GameUnavailable("Portal 2 is already running (maybe being played); not interrupting it")
    if not PREVIEW.exists():
        raise GameUnavailable(f"{PREVIEW} is missing: build any puzzle in the editor once")
    subprocess.Popen([str(EXE), "-game", "portal2", "-steam", "-insecure", "-novid", "-windowed",
                      "-w", "1280", "-h", "800", "+con_logfile", LOGNAME], cwd=GAME)
    pid = None
    try:
        # Before the window exists, -hijack would start a second game instead of forwarding.
        end = time.time() + timeout
        while time.time() < end and not _window_up():
            time.sleep(1)
        proc = game_process()
        if proc is None or "-insecure" not in proc[1].split():
            raise GameUnavailable("the game didn't start (or not with -insecure)")
        pid = proc[0]
        tag = f"GLADOS_{os.getpid()}_UP"
        if log.until(tag, timeout, poke=lambda: send([f"echo {tag}"], pid)) is None:
            raise GameUnavailable("the game didn't answer console commands")
        # A game still starting up can answer the echo and then drop these, so resend them
        # until the map starts loading.
        open_editor = ["puzzlemaker_show 1", "puzzlemaker_switch_session puzzlemaker/preview sp"]
        if log.until("Host_NewGame on map puzzlemaker/preview", timeout,
                     poke=lambda: send(open_editor, pid)) is None:
            raise GameUnavailable("the editor didn't start loading")
        if log.until("END loading assets", timeout) is None:
            raise GameUnavailable("the editor didn't open")
        time.sleep(3)
        patch()
    except BaseException:
        if pid is not None:
            quit_game(saved, pid)
        raise
    return pid


def _window_up():
    proc = game_process()
    if proc is None:
        return False
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          f"(Get-Process -Id {proc[0]}).MainWindowHandle"], capture_output=True, text=True)
    return out.stdout.strip() not in ("", "0")


def quit_game(saved, pid, timeout=60):
    if alive(pid):
        send(["quit"], pid)
        end = time.time() + timeout
        while time.time() < end and alive(pid):
            time.sleep(1)
    for p in autosaves():
        if p not in saved:
            p.unlink()
    for p, data in saved.items():
        p.write_bytes(data)
    VMF.unlink(missing_ok=True)
    CFG.unlink(missing_ok=True)
    shutil.rmtree(STAGE, ignore_errors=True)


# --- checking ------------------------------------------------------------------------------

def check_one(path, n, log, pid, timeout=20):
    STAGE.mkdir(parents=True, exist_ok=True)
    staged = STAGE / f"check{n % 2}.p2c"  # alternate names so a stale load can't pass as fresh
    shutil.copyfile(path, staged)
    VMF.unlink(missing_ok=True)
    tag = f"GLADOS_{os.getpid()}_{n}"
    for stage, cmd in (("load", f'puzzlemaker_load_dev "{staged.as_posix()}"'),
                       ("export", "puzzlemaker_export glados_check")):
        send([cmd, f"echo {tag}_{stage}"], pid)
        text = log.until(f"{tag}_{stage}", timeout)
        if text is None:
            return {"file": str(path), "verdict": "crash" if not alive(pid) else "timeout", "stage": stage}
        if stage == "load":
            loaded = text
        else:
            exported = text
    items = [{"name": m[0], "pos": [int(m[1]), int(m[2]), int(m[3])]} for m in INVALID.findall(exported)]
    size = re.search(r"chamber size: \((\d+),(\d+),(\d+)\)", exported)
    pm = lambda t: [l.split("]", 1)[1].strip() for l in t.splitlines() if "PUZZLEMAKER]" in l]
    if "Loaded puzzle" not in loaded:
        verdict = "load-failed"
    elif items or "invalid state" in exported:
        verdict = "invalid"
    elif size and VMF.exists():
        verdict = "valid"
    else:
        verdict = "error"
    return {"file": str(path), "verdict": verdict, "invalid_items": items,
            "chamber_size": [int(x) for x in size.groups()] if size else None,
            "load_messages": pm(loaded), "export_messages": pm(exported)}


def check(files, out_path, launch_game=False, timeout=20, progress=print, restart_every=500,
          max_crashes=10):
    """Check .p2c files, appending one JSON line each to out_path (resumable). With
    launch_game, the game is restarted every `restart_every` levels (it slows down as it runs)
    and after a crash; a level that crashed it is recorded as "crash". Raises GameUnavailable
    if the game can't be used."""
    out_path = Path(out_path)
    done = set()
    if out_path.exists():
        done = {json.loads(l)["file"] for l in out_path.read_text().splitlines() if l.strip()}
    todo = [f for f in files if str(f) not in done]
    progress(f"{len(todo)} to check ({len(done)} already in {out_path})")
    if not todo:
        return
    saved = {p: p.read_bytes() for p in autosaves()}
    start = time.time()
    i = crashes = 0
    with open(out_path, "a", encoding="utf-8") as out:
        while i < len(todo):
            log = Log()
            if launch_game:
                pid = launch(log, saved)
            else:
                proc = game_process()
                if proc is None:
                    raise GameUnavailable("Portal 2 isn't running")
                pid = proc[0]
                patch()
            try:
                first = i
                while i < len(todo) and not (launch_game and i - first >= restart_every):
                    r = check_one(todo[i], i + 1, log, pid, timeout)
                    if r["verdict"] == "timeout":  # alive but silent: give it one more go
                        r = check_one(todo[i], i + 1, log, pid, timeout)
                    if r["verdict"] == "timeout":
                        raise GameUnavailable(f"the game stopped answering at {todo[i]}")
                    out.write(json.dumps(r) + "\n")
                    out.flush()
                    i += 1
                    if i % 100 == 0 or i == len(todo):
                        progress(f"[{i}/{len(todo)} {time.time() - start:6.0f}s]")
                    if r["verdict"] == "crash":
                        crashes += 1
                        progress(f"the game crashed on {todo[i - 1]} ({crashes} so far)")
                        if not launch_game or crashes > max_crashes:
                            raise GameUnavailable("the game crashed")
                        break
            finally:
                if launch_game:
                    quit_game(saved, pid)
                else:
                    VMF.unlink(missing_ok=True)


def p2c_files(paths):
    return [g for f in map(Path, paths) for g in (sorted(f.glob("*.p2c")) if f.is_dir() else [f])]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="check .p2c files (or directories of them)")
    c.add_argument("files", nargs="+")
    c.add_argument("--out", required=True)
    c.add_argument("--launch", action="store_true", help="start and quit the game too")
    c.add_argument("--timeout", type=float, default=20)
    p = sub.add_parser("patch", help="patch a game already running with -insecure")
    p.add_argument("--undo", action="store_true")
    a = ap.parse_args()
    try:
        if a.cmd == "check":
            check(p2c_files(a.files), a.out, a.launch, a.timeout, progress=lambda s: print(s, flush=True))
        else:
            patch(a.undo)
            print("unpatched" if a.undo else "patched")
    except GameUnavailable as e:
        print(f"Portal 2 unavailable: {e}", file=sys.stderr)
        sys.exit(UNAVAILABLE)


if __name__ == "__main__":
    main()
