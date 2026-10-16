"""Ask Portal 2's own Puzzle Maker whether .p2c files are valid.

The editor refuses to export a puzzle with misplaced items and logs each one as
`INVALID ITEM:  <TargetName> ( x , y , z )` (x y z = the item's VoxelPos). There is no console
command that loads a file in the retail game: `puzzlemaker_load_dev` calls a stub. `patch`
rewrites one byte in the running game so it calls the loader the My Chambers list uses, and
sets the editor's active flag, which `puzzlemaker_export` checks. Commands go in and console
output comes back over the game's TCP console (-netconport). (`portal2.exe -hijack` per
command was slow, and popped up "did not accept the request to load a savegame" whenever the
game was too busy to answer its window message in time, stalling until it was dismissed.)

Memory is only patched in a game started with -insecure (VAC off for that session), and the
patch refuses any client.dll that doesn't match the build it was written for (Jan 2025).

    python train/portal2.py check out/ingame/sample/p2c --out results.jsonl --launch
    python train/portal2.py check DIR --out results.jsonl --port PORT --password PW
    python train/portal2.py patch [--undo]       # a game you started with -insecure

A game you start yourself needs `-insecure -netconport PORT -netconpassword PW` in its launch
options. `--launch` starts the game (refusing if it's already running), opens the editor, patches it,
checks, quits, and puts back the editor autosaves it overwrote. Exits 3 if the game couldn't
be used, so callers can carry on without it.
"""
import argparse, collections, ctypes, ctypes.wintypes as wt, itertools, json, re, secrets, shutil, socket, subprocess, sys, time
from pathlib import Path

GAME = Path(r"C:\Program Files (x86)\Steam\steamapps\common\Portal 2")
EXE = GAME / "portal2.exe"
STAGE = GAME / "portal2" / "puzzles" / "glados_check"
VMF = GAME / "sdk_content" / "maps" / "glados_check.vmf"
PREVIEW = GAME / "portal2" / "maps" / "puzzlemaker" / "preview.bsp"
CANARY = Path(__file__).resolve().parent.parent / "fixtures" / "ingame" / "canary_invalid.p2c"
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


class Console:
    """The game's TCP console (-netconport): command lines in, everything it prints out."""

    acks = itertools.count(1)

    def __init__(self, port, password, timeout=60):
        end = time.time() + timeout
        while True:
            try:
                self.sock = socket.create_connection(("127.0.0.1", port), timeout=2)
                break
            except OSError:
                if time.time() > end:
                    raise GameUnavailable(f"nothing listening on console port {port}")
                time.sleep(1)
        self.sock.settimeout(0.02)
        self.buf = ""
        self.lines = []
        if password:
            self.send(f"PASS {password}")
        # A game still starting up can take a while to answer, or drop what it was sent.
        while self.ask(["echo glados_hello"], 5) is None:
            if time.time() > end:
                raise GameUnavailable("the console didn't answer (wrong password?)")

    def send(self, *cmds):
        assert not any("\n" in c for c in cmds)
        try:
            self.sock.sendall(("\n".join(cmds) + "\n").encode("utf-8"))
        except OSError as e:
            raise GameUnavailable(f"the console went away ({e})")

    def _read(self):
        try:
            data = self.sock.recv(1 << 16)
        except socket.timeout:
            return
        except OSError as e:
            raise GameUnavailable(f"the console went away ({e})")
        if not data:
            raise GameUnavailable("the game closed its console")
        self.buf += data.decode("utf-8", "replace")
        *done, self.buf = self.buf.split("\n")
        self.lines += [l.rstrip("\r") for l in done]

    def until(self, pred, timeout):
        """Lines printed up to the first one `pred` accepts: (lines before it, that line), or
        None on timeout. Lines are consumed either way."""
        end = time.time() + timeout
        seen = []
        while True:
            while self.lines:
                line = self.lines.pop(0)
                if pred(line):
                    return seen, line
                seen.append(line)
            if time.time() > end:
                return None
            self._read()

    def ask(self, cmds, timeout=5):
        """Runs commands and returns what they printed, or None on timeout."""
        tag = f"glados_ack_{next(Console.acks)}"
        self.send(*cmds, f"echo {tag}")
        got = self.until(lambda l: l.strip() == tag, timeout)
        return None if got is None else "\n".join(got[0])

    def drain(self):
        self._read()
        lines, self.lines = self.lines, []
        return lines

    def close(self):
        self.sock.close()


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


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


def launch(saved, timeout=180, extra=()):
    """Start the game with a TCP console on a free port and a random password, open the
    editor and patch it. Returns (pid, Console). `extra`: more command-line options."""
    if game_process() is not None:
        raise GameUnavailable("Portal 2 is already running (maybe being played); not interrupting it")
    if not PREVIEW.exists():
        raise GameUnavailable(f"{PREVIEW} is missing: build any puzzle in the editor once")
    port, password = free_port(), secrets.token_hex(8)
    subprocess.Popen([str(EXE), "-game", "portal2", "-steam", "-insecure", "-novid", "-windowed",
                      "-w", "1280", "-h", "800", "-netconport", str(port), "-netconpassword", password,
                      *extra], cwd=GAME)
    pid = con = None
    try:
        con = Console(port, password, timeout)
        proc = game_process()
        if proc is None or "-insecure" not in proc[1].split():
            raise GameUnavailable("the game didn't start (or not with -insecure)")
        pid = proc[0]
        open_editor(con, timeout)
    except BaseException:
        if pid is None:
            pid = (game_process() or (None,))[0]
        if pid is not None:
            quit_game(saved, pid, con)
        raise
    return pid, con


def open_editor(con, timeout=180):
    """Opens the editor (from the main menu or a level) and patches it for loading files."""
    end = time.time() + timeout
    while True:  # resent until the map starts loading: a game still starting up can drop them
        con.send("puzzlemaker_show 1", "puzzlemaker_switch_session puzzlemaker/preview sp")
        if con.until(lambda l: "Host_NewGame on map puzzlemaker/preview" in l, 10) is not None:
            break
        if time.time() > end:
            raise GameUnavailable("the editor didn't start loading")
    if con.until(lambda l: "END loading assets" in l, timeout) is None:
        raise GameUnavailable("the editor didn't open")
    time.sleep(3)
    patch()


def quit_game(saved, pid, con=None, timeout=60):
    """Quits a game this started (over `con`, else by ending the process) and tidies up."""
    if alive(pid):
        try:
            if con is None:
                raise GameUnavailable("no console")
            con.send("quit")
        except (GameUnavailable, OSError):
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
        end = time.time() + timeout
        while time.time() < end and alive(pid):
            time.sleep(1)
        if alive(pid):
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
    if con is not None:
        con.close()
    for p in autosaves():
        if p not in saved:
            p.unlink()
    for p, data in saved.items():
        p.write_bytes(data)
    VMF.unlink(missing_ok=True)
    shutil.rmtree(STAGE, ignore_errors=True)


# --- checking ------------------------------------------------------------------------------

def check_one(path, n, con, pid, timeout=20, settle=0.1):
    STAGE.mkdir(parents=True, exist_ok=True)
    staged = STAGE / f"check{n % 2}.p2c"  # alternate names so a stale load can't pass as fresh
    shutil.copyfile(path, staged)
    VMF.unlink(missing_ok=True)
    # A tag unique to this attempt marks the end of each step's output, so a late answer to an
    # attempt that timed out can't pass for this one's. The editor marks misplaced items while
    # it draws, so the export waits `settle` seconds after the load: straight after it (even a
    # few `wait` frames after), the export passes every level as valid. 50 ms was enough in
    # tests, 100 ms with the game behind other windows; check() watches for it not being enough.
    tag = f"glados_check_{next(Console.acks)}"
    texts = {}
    for stage, cmd in (("load", f'puzzlemaker_load_dev "{staged.as_posix()}"'),
                       ("export", "puzzlemaker_export glados_check")):
        if stage == "export":
            time.sleep(settle)
        try:
            con.send(cmd, f"echo {tag}_{stage}")
            got = con.until(lambda l: l.strip() == f"{tag}_{stage}", timeout)
        except GameUnavailable:  # the console closed: the game is going down
            end = time.time() + 10
            while time.time() < end and alive(pid):
                time.sleep(0.5)
            got = None
        if got is None:
            return {"file": str(path), "verdict": "crash" if not alive(pid) else "timeout", "stage": stage}
        texts[stage] = "\n".join(got[0])
    loaded, exported = texts["load"], texts["export"]
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


def check(files, out_path, launch_game=False, timeout=20, progress=print, restart_every=100,
          max_crashes=None, port=None, password="", canary_every=50, max_bad=3):
    """Check .p2c files, appending one JSON line each to out_path (resumable). With
    launch_game, the game is restarted every `restart_every` levels and after a crash (exports
    slow from ~0.15 s to ~0.6 s over the first 150 levels of a session, and neither
    puzzlemaker_new_chamber nor reopening the editor helps; a restart takes ~17 s); a level that
    crashed it is recorded as "crash", and it gives up after `max_crashes` crashes (default: 10,
    or 5% of the levels to check if more; sampled levels crash the editor ~2.6% of the time).
    Otherwise it uses the
    game already running, on console `port`, with the editor open. Raises GameUnavailable if
    the game can't be used.

    The editor only marks misplaced items as it draws, so exported too soon after loading (see
    check_one) every level passes as valid, and a session can get stuck that way. So CANARY
    (a level the editor must call invalid) is checked first and then every `canary_every`
    levels. Results are only written
    once a canary after them has passed; if it fails, the game is restarted and those levels
    are checked again."""
    out_path = Path(out_path)
    done = set()
    if out_path.exists():
        done = {json.loads(l)["file"] for l in out_path.read_text().splitlines() if l.strip()}
    todo = collections.deque(f for f in files if str(f) not in done)
    total = len(todo)
    progress(f"{total} to check ({len(done)} already in {out_path})")
    if max_crashes is None:
        max_crashes = max(10, total // 20)
    if not todo:
        return
    saved = {p: p.read_bytes() for p in autosaves()}
    start = time.time()
    written = crashes = bad = 0
    seq = itertools.count(1)  # numbers every check_one, so the staged name always alternates
    with open(out_path, "a", encoding="utf-8") as out:

        def write(rows):
            nonlocal written
            for r in rows:
                out.write(json.dumps(r) + "\n")
                written += 1
                if written % 100 == 0 or written == total:
                    progress(f"[{written}/{total} {time.time() - start:6.0f}s]")
            out.flush()

        while todo:
            if launch_game:
                pid, con = launch(saved)
            else:
                proc = game_process()
                if proc is None:
                    raise GameUnavailable("Portal 2 isn't running")
                if not port:
                    raise GameUnavailable("pass the console port (and password) of the game you started")
                pid = proc[0]
                patch()
                con = Console(port, password)
            try:
                def one(path):
                    r = check_one(path, next(seq), con, pid, timeout)
                    if r["verdict"] == "timeout":  # alive but silent: give it one more go
                        r = check_one(path, next(seq), con, pid, timeout)
                    if r["verdict"] == "timeout":
                        raise GameUnavailable(f"the game stopped answering at {path}")
                    return r

                def canary_flagged():
                    return one(CANARY)["verdict"] == "invalid"

                checked = 0
                pending = []
                healthy = canary_flagged()
                while healthy and todo and not (launch_game and checked >= restart_every):
                    path = todo.popleft()
                    r = one(path)
                    checked += 1
                    if r["verdict"] == "crash":
                        crashes += 1
                        progress(f"the game crashed on {path} ({crashes} so far)")
                        write([r])  # the levels before it go round again: no canary after them
                        todo.extendleft(reversed([Path(x["file"]) for x in pending]))
                        pending = []
                        if not launch_game or crashes > max_crashes:
                            raise GameUnavailable("the game crashed")
                        break
                    pending.append(r)
                    if len(pending) >= canary_every or not todo or (launch_game and checked >= restart_every):
                        healthy = canary_flagged()
                        if healthy:
                            write(pending)
                            pending = []
                if not healthy:
                    todo.extendleft(reversed([Path(x["file"]) for x in pending]))
                    bad += 1
                    progress(f"the editor stopped flagging invalid items; restarting ({bad} so far)")
                    if not launch_game or bad > max_bad:
                        raise GameUnavailable("the editor isn't flagging invalid items (the canary passed as valid)")
            finally:
                if launch_game:
                    quit_game(saved, pid, con)
                else:
                    con.close()
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
    c.add_argument("--port", type=int, help="console port (-netconport) of a game you started")
    c.add_argument("--password", default="", help="its -netconpassword")
    p = sub.add_parser("patch", help="patch a game already running with -insecure")
    p.add_argument("--undo", action="store_true")
    a = ap.parse_args()
    try:
        if a.cmd == "check":
            check(p2c_files(a.files), a.out, a.launch, a.timeout, progress=lambda s: print(s, flush=True),
                  port=a.port, password=a.password)
        else:
            patch(a.undo)
            print("unpatched" if a.undo else "patched")
    except GameUnavailable as e:
        print(f"Portal 2 unavailable: {e}", file=sys.stderr)
        sys.exit(UNAVAILABLE)


if __name__ == "__main__":
    main()
