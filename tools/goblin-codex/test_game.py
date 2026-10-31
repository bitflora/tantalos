"""Local console test for our isolated mod; never attaches to another game session."""
import argparse
import json
import secrets
import socket
import subprocess
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'out/goblin-codex'
GAME=Path(r'C:\Program Files (x86)\Steam\steamapps\common\Portal 2')
SESSION=OUT/'test-session.json'


def running():
    r=subprocess.run(['tasklist','/FI','IMAGENAME eq portal2.exe','/FO','CSV','/NH'],capture_output=True,text=True)
    return 'portal2.exe' in r.stdout.lower()


def connect(port,timeout=30):
    until=time.monotonic()+timeout
    while True:
        try:
            s=socket.create_connection(('127.0.0.1',port),timeout=2);s.settimeout(.5);return s
        except OSError:
            if time.monotonic()>until:raise
            time.sleep(.5)


def main():
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');p.add_argument('--allowmultiple',action='store_true');p.add_argument('--map',default='goblin_codex/showroom');p.add_argument('--command',action='append',default=[]);p.add_argument('--seconds',type=float,default=3)
    args=p.parse_args()
    if args.launch:
        if running() and not args.allowmultiple:raise SystemExit('Portal 2 is already running. Leaving it untouched.')
        with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
        password=secrets.token_hex(12)
        runtime=OUT/'packed-game' if (OUT/'packed-game/gameinfo.txt').exists() else OUT/'game'
        proc=subprocess.Popen([str(GAME/'portal2.exe'),*(['-allowmultiple'] if args.allowmultiple else []),'-game',str(runtime),'-steam','-insecure','-novid','-windowed','-w','1280','-h','800','-netconport',str(port),'-netconpassword',password,'-condebug','+map',args.map],cwd=GAME)
        SESSION.write_text(json.dumps(dict(pid=proc.pid,port=port,password=password)))
        print('Started isolated Goblin test; pid',proc.pid,flush=True)
    session=json.loads(SESSION.read_text())
    with connect(session['port']) as s:
        s.sendall(('PASS '+session['password']+'\n').encode())
        for cmd in args.command:s.sendall((cmd+'\n').encode())
        end=time.monotonic()+args.seconds;data=bytearray()
        while time.monotonic()<end:
            try:
                chunk=s.recv(65536)
                if not chunk:break
                data+=chunk
            except socket.timeout:pass
    log=data.decode('utf-8',errors='replace')
    with (OUT/'game-test.log').open('a') as f:f.write(log)
    print(log[-16000:])


if __name__=='__main__':main()
