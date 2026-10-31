"""Deterministic painted metal atlas, shared by Blender, glTF and Source."""
from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets/goblin-codex"
PALETTE = [
    (203, 204, 185), (48, 54, 53), (118, 126, 121), (82, 107, 81),
    (196, 151, 37), (87, 51, 121), (215, 224, 71), (35, 26, 23),
    (112, 57, 33), (27, 38, 41), (37, 97, 102), (146, 130, 93),
    (177, 184, 174), (24, 26, 28), (118, 83, 135), (219, 205, 168),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(620)
    noise = np.random.default_rng(620)
    size = 512
    atlas = Image.new("RGB", (size * 4, size * 4))
    for i, color in enumerate(PALETTE):
        a = np.clip(np.array(color)[None, None, :] + noise.normal(0, 3.4, (size, size, 1)), 0, 255).astype("uint8")
        tile = Image.fromarray(a)
        draw = ImageDraw.Draw(tile)
        if i in (0, 3, 4, 12, 15):
            # Islands of worn paint with a dark oxidized edge and exposed undercoat.
            for _ in range(300):
                x, y = rng.randrange(size), rng.randrange(size)
                r = rng.choices([1, 3, 7, 14], [55, 28, 13, 4])[0]
                pts = [(x + rng.randrange(-r, r + 1), y + rng.randrange(-r, r + 1)) for _ in range(9)]
                pts.sort(key=lambda p: __import__('math').atan2(p[1]-y, p[0]-x))
                draw.polygon(pts, fill=(70, 73, 63))
                if r > 3:
                    draw.polygon([(int(x+(px-x)*.65),int(y+(py-y)*.65)) for px,py in pts], fill=(113,119,105))
            for _ in range(100):
                x, y = rng.randrange(size), rng.randrange(size)
                draw.line((x, y, x + rng.randrange(-15, 30), y+rng.randrange(3, 17)), fill=(91,94,81), width=1)
            for _ in range(900):
                x,y=rng.randrange(size),rng.randrange(size)
                draw.point((x,y), fill=rng.choice([(125,127,112),(231,230,208),(79,81,69)]))
        elif i in (1, 2, 8, 9, 13):
            for _ in range(110):
                x,y=rng.randrange(size),rng.randrange(size)
                c=tuple(min(255,v+rng.randrange(10,35)) for v in color)
                draw.line((x,y,x+rng.randrange(8,45),y+1),fill=c,width=1)
        elif i in (5, 14):
            # Cracked enamel; individual eye shards are also separate geometry.
            for _ in range(9):
                x,y=rng.randrange(size),rng.randrange(size)
                draw.line([(x,y),(x+25,y+40),(x+7,y+76),(x+46,y+110)],fill=(29,19,31),width=2)
        atlas.paste(tile, ((i % 4)*size, (i//4)*size))
    atlas.save(OUT / "goblin_atlas.png")
    print(OUT / "goblin_atlas.png")


if __name__ == "__main__":
    main()
