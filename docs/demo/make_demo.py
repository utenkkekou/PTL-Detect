"""Create a synthetic manual-annotation fixture, not a model prediction.

Run from the repository root: python docs/demo/make_demo.py
Only Pillow is required (included in the base package dependencies).
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def font(size: int):
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def make_image() -> Path:
    image = Image.new("RGB", (1200, 680), "#111c2b")
    draw = ImageDraw.Draw(image)
    draw.text((40, 30), "FOUR-CLASS ANNOTATION WORKBENCH", font=font(30), fill="#eaf3fa")
    draw.text((40, 78), "SYNTHETIC DEMO / MANUAL BOXES / NOT MODEL OUTPUT", font=font(18), fill="#81dfd3")
    draw.line((40, 120, 1160, 120), fill="#2d4257", width=2)
    labels = ["PEDESTRIAN RED", "PEDESTRIAN GREEN", "STATE UNKNOWN", "VEHICLE SIGNAL"]
    colors = ["#ff606a", "#62e8a4", "#a5b2c8", "#ffce75"]
    for index, (name, color) in enumerate(zip(labels, colors)):
        x = 65 + index * 290
        draw.rounded_rectangle((x, 155, x + 220, 555), radius=14, fill="#192b3e", outline="#304b63", width=2)
        draw.text((x + 15, 176), f"0{index + 1}", font=font(20), fill=color)
        bx = x + 55
        draw.rectangle((bx + 49, 450, bx + 61, 530), fill="#496377")
        draw.rounded_rectangle((bx, 230, bx + 110, 455), radius=18, fill="#07101a", outline="#597389", width=3)
        if index < 3:
            draw.rounded_rectangle((bx + 12, 246, bx + 98, 438), radius=10, fill="#10202b")
            lit = color if index < 2 else "#536475"
            draw.ellipse((bx + 44, 282, bx + 66, 304), fill=lit)
            draw.line((bx + 55, 312, bx + 55, 364), fill=lit, width=10)
            draw.line((bx + 55, 320, bx + 29, 343), fill=lit, width=8)
            draw.line((bx + 55, 320, bx + 80, 340), fill=lit, width=8)
            draw.line((bx + 55, 362, bx + 33, 402), fill=lit, width=9)
            draw.line((bx + 55, 362, bx + 79, 399), fill=lit, width=9)
            if index == 2:
                draw.rectangle((bx + 12, 332, bx + 98, 355), fill="#213341")
        else:
            for cy, c in [(277, "#553235"), (343, "#5b4e32"), (409, "#5ee9a2")]:
                draw.ellipse((bx + 31, cy - 24, bx + 79, cy + 24), fill=c)
        draw.text((x + 10, 575), name, font=font(17), fill=color)
    draw.text((40, 634), "Practice labeling only. This fixture does not represent real-world detection performance.", font=font(18), fill="#91a9bc")
    target = Path(__file__).resolve().parent / "signals-demo.png"
    image.save(target, optimize=True)
    return target


if __name__ == "__main__":
    print(make_image().name)
