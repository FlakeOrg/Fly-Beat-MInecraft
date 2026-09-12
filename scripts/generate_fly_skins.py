"""Generates Minecraft Java Edition player skins (64x64 PNG, standard UV
layout) with a fly/insect theme: dark iridescent body, red compound eyes,
translucent wing patches on the back overlay layer. Several color variants
so the 20 parallel bot instances don't all look identical.

Not connected to the connectome/training pipeline at all - purely cosmetic,
run once to produce files under assets/fly_skins/ (tracked in git, since
they need to be reachable at a public raw.githubusercontent.com URL for
Mojang/MineSkin's skin-signing pipeline - see scripts/apply_fly_skins.py),
then applied to each bot via SkinsRestorer.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fly_skins"

# (body_color, wing_color, eye_color) per variant
VARIANTS = {
    "housefly": ((30, 28, 35), (210, 220, 230, 90), (170, 20, 20)),
    "bluebottle": ((25, 40, 70), (200, 210, 240, 90), (200, 30, 30)),
    "greenbottle": ((25, 55, 35), (210, 235, 210, 90), (180, 40, 20)),
    "fruitfly": ((60, 35, 20), (230, 210, 190, 90), (150, 10, 10)),
    "firefly": ((20, 20, 25), (230, 210, 120, 100), (220, 180, 40)),
}


def _fill(px, x0, y0, x1, y1, color):
    for x in range(x0, x1):
        for y in range(y0, y1):
            px[x, y] = color


def build_skin(body: tuple[int, int, int], wing: tuple[int, int, int, int], eye: tuple[int, int, int]) -> Image.Image:
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    px = img.load()
    opaque = (*body, 255)

    # --- base layer: head, torso, arms, legs (standard 64x64 UV layout) ---
    # head
    _fill(px, 8, 0, 24, 8, opaque)  # top+bottom
    _fill(px, 0, 8, 32, 16, opaque)  # right/front/left/back
    # torso
    _fill(px, 16, 16, 40, 20, opaque)
    _fill(px, 16, 20, 40, 32, opaque)
    # right arm
    _fill(px, 40, 16, 56, 20, opaque)
    _fill(px, 40, 20, 56, 32, opaque)
    # left arm (new-format region)
    _fill(px, 32, 48, 48, 52, opaque)
    _fill(px, 32, 52, 48, 64, opaque)
    # right leg
    _fill(px, 0, 16, 16, 20, opaque)
    _fill(px, 0, 20, 16, 32, opaque)
    # left leg (new-format region)
    _fill(px, 16, 48, 32, 52, opaque)
    _fill(px, 16, 52, 32, 64, opaque)

    # compound eyes on the head's front face (front face is x8-16, y8-16)
    _fill(px, 9, 10, 12, 13, (*eye, 255))
    _fill(px, 12, 10, 15, 13, (*eye, 255))

    # wings as a semi-transparent patch on the torso's back-overlay
    # (jacket layer sits on the body's back face, 32,36 - 36,48)
    _fill(px, 32, 36, 40, 46, wing)
    _fill(px, 16, 36, 24, 46, wing)

    return img


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, (body, wing, eye) in VARIANTS.items():
        img = build_skin(body, wing, eye)
        path = OUT_DIR / f"{name}.png"
        img.save(path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
