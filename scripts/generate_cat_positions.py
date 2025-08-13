#!/usr/bin/env python3

import os
import math
from typing import Tuple

from PIL import Image, ImageDraw


FINAL_SIZE = 1024  # output image size (square)
SCALE = 4          # supersampling factor for smoother strokes
SIZE = FINAL_SIZE * SCALE
BACKGROUND = (255, 255, 255)
FOREGROUND = (0, 0, 0)
# Choose a visually pleasant thick outline in final image
FINAL_STROKE = 14
STROKE = FINAL_STROKE * SCALE

OUTPUT_DIR = "/workspace/cat_positions"


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_canvas() -> Tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (SIZE, SIZE), BACKGROUND)
    draw = ImageDraw.Draw(img)
    return img, draw


def downscale_and_save(img: Image.Image, filename: str) -> None:
    out = img.resize((FINAL_SIZE, FINAL_SIZE), Image.Resampling.LANCZOS)
    out.save(os.path.join(OUTPUT_DIR, filename), format="PNG")


# ---------- Basic shape helpers ----------

def ellipse_bbox(cx: float, cy: float, rx: float, ry: float) -> Tuple[int, int, int, int]:
    return (
        int(cx - rx),
        int(cy - ry),
        int(cx + rx),
        int(cy + ry),
    )


def draw_circle_outline(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float) -> None:
    draw.ellipse(ellipse_bbox(cx, cy, r, r), outline=FOREGROUND, width=STROKE)


def draw_ellipse_outline(draw: ImageDraw.ImageDraw, cx: float, cy: float, rx: float, ry: float) -> None:
    draw.ellipse(ellipse_bbox(cx, cy, rx, ry), outline=FOREGROUND, width=STROKE)


def draw_arc_outline(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    start_deg: float,
    end_deg: float,
) -> None:
    draw.arc(ellipse_bbox(cx, cy, rx, ry), start=start_deg, end=end_deg, fill=FOREGROUND, width=STROKE)


def draw_line(draw: ImageDraw.ImageDraw, p1: Tuple[float, float], p2: Tuple[float, float]) -> None:
    draw.line((int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])), fill=FOREGROUND, width=STROKE)


def draw_triangle_outline(draw: ImageDraw.ImageDraw, p1, p2, p3) -> None:
    draw_line(draw, p1, p2)
    draw_line(draw, p2, p3)
    draw_line(draw, p3, p1)


# ---------- Cat feature helpers (still minimalist) ----------

def draw_ears(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float) -> None:
    # Two simple triangles as ears, placed above the head circle
    ear_height = r * 0.95
    ear_base = r * 0.9

    # Left ear
    left_apex = (cx - r * 0.55, cy - r * 1.35)
    left_base_left = (cx - r * 1.05, cy - r * 0.25)
    left_base_right = (cx - r * 0.25, cy - r * 0.25)
    draw_triangle_outline(draw, left_apex, left_base_left, left_base_right)

    # Right ear
    right_apex = (cx + r * 0.55, cy - r * 1.35)
    right_base_left = (cx + r * 0.25, cy - r * 0.25)
    right_base_right = (cx + r * 1.05, cy - r * 0.25)
    draw_triangle_outline(draw, right_apex, right_base_left, right_base_right)


def draw_head(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float) -> None:
    # No facial features per requirements; only circle + ears
    draw_circle_outline(draw, cx, cy, r)
    draw_ears(draw, cx, cy, r)


# ---------- Pose drawings ----------

def pose_standing_all_fours() -> str:
    img, draw = create_canvas()

    # Body (horizontal oval)
    body_cx, body_cy = SIZE * 0.58, SIZE * 0.58
    body_rx, body_ry = SIZE * 0.30, SIZE * 0.20
    draw_ellipse_outline(draw, body_cx, body_cy, body_rx, body_ry)

    # Head
    head_cx, head_cy, head_r = SIZE * 0.28, SIZE * 0.50, SIZE * 0.12
    draw_head(draw, head_cx, head_cy, head_r)

    # Legs (four short vertical lines under body)
    leg_top_y = body_cy + body_ry * 0.65
    leg_len = SIZE * 0.14
    leg_xs = [
        body_cx - body_rx * 0.55,
        body_cx - body_rx * 0.15,
        body_cx + body_rx * 0.15,
        body_cx + body_rx * 0.55,
    ]
    for x in leg_xs:
        draw_line(draw, (x, leg_top_y), (x, leg_top_y + leg_len))

    # Tail (upward arc behind body)
    tail_cx, tail_cy = body_cx + body_rx * 0.95, body_cy - body_ry * 0.3
    tail_rx, tail_ry = body_rx * 0.55, body_ry * 1.1
    draw_arc_outline(draw, tail_cx, tail_cy, tail_rx, tail_ry, start_deg=200, end_deg=360)

    downscale_and_save(img, "standing_on_all_fours.png")
    return "standing_on_all_fours.png"


def pose_standing_hind_legs() -> str:
    img, draw = create_canvas()

    # Body (vertical oval)
    body_cx, body_cy = SIZE * 0.52, SIZE * 0.58
    body_rx, body_ry = SIZE * 0.20, SIZE * 0.30
    draw_ellipse_outline(draw, body_cx, body_cy, body_rx, body_ry)

    # Head
    head_cx, head_cy, head_r = SIZE * 0.52, SIZE * 0.27, SIZE * 0.12
    draw_head(draw, head_cx, head_cy, head_r)

    # Front legs (raised/forward lines)
    left_front_start = (body_cx - body_rx * 0.6, body_cy - body_ry * 0.2)
    left_front_end = (left_front_start[0] - SIZE * 0.06, left_front_start[1] - SIZE * 0.16)
    right_front_start = (body_cx + body_rx * 0.6, body_cy - body_ry * 0.2)
    right_front_end = (right_front_start[0] + SIZE * 0.06, right_front_start[1] - SIZE * 0.16)
    draw_line(draw, left_front_start, left_front_end)
    draw_line(draw, right_front_start, right_front_end)

    # Hind legs (short vertical lines at bottom)
    hind_y_top = body_cy + body_ry * 0.55
    hind_len = SIZE * 0.16
    draw_line(draw, (body_cx - body_rx * 0.45, hind_y_top), (body_cx - body_rx * 0.45, hind_y_top + hind_len))
    draw_line(draw, (body_cx + body_rx * 0.45, hind_y_top), (body_cx + body_rx * 0.45, hind_y_top + hind_len))

    # Tail (curved to the side)
    tail_cx, tail_cy = body_cx + body_rx * 0.9, body_cy
    tail_rx, tail_ry = body_rx * 0.65, body_ry * 0.9
    draw_arc_outline(draw, tail_cx, tail_cy, tail_rx, tail_ry, start_deg=300, end_deg=120)

    downscale_and_save(img, "standing_up_on_hind_legs.png")
    return "standing_up_on_hind_legs.png"


def pose_loaf() -> str:
    img, draw = create_canvas()

    # Body (rounded loaf)
    body_cx, body_cy = SIZE * 0.54, SIZE * 0.62
    body_rx, body_ry = SIZE * 0.34, SIZE * 0.22
    draw_ellipse_outline(draw, body_cx, body_cy, body_rx, body_ry)

    # Head (tucked close)
    head_cx, head_cy, head_r = SIZE * 0.30, SIZE * 0.58, SIZE * 0.12
    draw_head(draw, head_cx, head_cy, head_r)

    # Tail (wrapped along side)
    tail_cx, tail_cy = body_cx + body_rx * 0.65, body_cy + body_ry * 0.1
    tail_rx, tail_ry = body_rx * 0.55, body_ry * 0.7
    draw_arc_outline(draw, tail_cx, tail_cy, tail_rx, tail_ry, start_deg=90, end_deg=250)

    downscale_and_save(img, "loaf.png")
    return "loaf.png"


def pose_curled_up() -> str:
    img, draw = create_canvas()

    # Body (nearly circular curled shape)
    body_cx, body_cy = SIZE * 0.54, SIZE * 0.58
    body_rx, body_ry = SIZE * 0.34, SIZE * 0.32
    draw_ellipse_outline(draw, body_cx, body_cy, body_rx, body_ry)

    # Tail (wrap around outer curve)
    tail_cx, tail_cy = body_cx + body_rx * 0.1, body_cy + body_ry * 0.05
    tail_rx, tail_ry = body_rx * 0.75, body_ry * 0.9
    draw_arc_outline(draw, tail_cx, tail_cy, tail_rx, tail_ry, start_deg=200, end_deg=30)

    # Head (nestled along edge)
    head_cx, head_cy, head_r = body_cx - body_rx * 0.35, body_cy - body_ry * 0.15, SIZE * 0.11
    draw_head(draw, head_cx, head_cy, head_r)

    downscale_and_save(img, "curled_up.png")
    return "curled_up.png"


def pose_between_legs() -> str:
    img, draw = create_canvas()

    # Interpret as cat sitting with front paws between splayed hind legs
    # Torso (vertical oval)
    body_cx, body_cy = SIZE * 0.52, SIZE * 0.58
    body_rx, body_ry = SIZE * 0.22, SIZE * 0.30
    draw_ellipse_outline(draw, body_cx, body_cy, body_rx, body_ry)

    # Head
    head_cx, head_cy, head_r = SIZE * 0.52, SIZE * 0.30, SIZE * 0.115
    draw_head(draw, head_cx, head_cy, head_r)

    # Hind legs (rounded outlines on both sides)
    hind_rx, hind_ry = body_rx * 0.85, body_ry * 0.55
    left_hind_cx, left_hind_cy = body_cx - body_rx * 0.95, body_cy + body_ry * 0.3
    right_hind_cx, right_hind_cy = body_cx + body_rx * 0.95, body_cy + body_ry * 0.3
    draw_ellipse_outline(draw, left_hind_cx, left_hind_cy, hind_rx, hind_ry)
    draw_ellipse_outline(draw, right_hind_cx, right_hind_cy, hind_rx, hind_ry)

    # Front paws (two short vertical lines centered)
    paw_y_top = body_cy + body_ry * 0.35
    paw_len = SIZE * 0.14
    draw_line(draw, (body_cx - SIZE * 0.03, paw_y_top), (body_cx - SIZE * 0.03, paw_y_top + paw_len))
    draw_line(draw, (body_cx + SIZE * 0.03, paw_y_top), (body_cx + SIZE * 0.03, paw_y_top + paw_len))

    # Tail (curving forward between legs)
    tail_cx, tail_cy = body_cx, body_cy + body_ry * 0.15
    tail_rx, tail_ry = body_rx * 0.9, body_ry * 0.8
    draw_arc_outline(draw, tail_cx, tail_cy, tail_rx, tail_ry, start_deg=230, end_deg=330)

    downscale_and_save(img, "between_legs.png")
    return "between_legs.png"


def pose_belly_up() -> str:
    img, draw = create_canvas()

    # Body (on the back)
    body_cx, body_cy = SIZE * 0.52, SIZE * 0.56
    body_rx, body_ry = SIZE * 0.26, SIZE * 0.34
    draw_ellipse_outline(draw, body_cx, body_cy, body_rx, body_ry)

    # Head (slightly offset)
    head_cx, head_cy, head_r = body_cx - body_rx * 0.2, body_cy - body_ry * 0.65, SIZE * 0.11
    draw_head(draw, head_cx, head_cy, head_r)

    # Legs (four upward/outward short lines)
    top_y = body_cy - body_ry * 0.65
    leg_len = SIZE * 0.16
    offsets = [-0.55, -0.15, 0.15, 0.55]
    for ox in offsets:
        x = body_cx + body_rx * ox
        draw_line(draw, (x, top_y), (x, top_y - leg_len))

    # Tail (to the side)
    tail_cx, tail_cy = body_cx + body_rx * 0.9, body_cy + body_ry * 0.1
    tail_rx, tail_ry = body_rx * 0.7, body_ry * 0.7
    draw_arc_outline(draw, tail_cx, tail_cy, tail_rx, tail_ry, start_deg=20, end_deg=180)

    downscale_and_save(img, "belly_up.png")
    return "belly_up.png"


def pose_pretzel() -> str:
    img, draw = create_canvas()

    # Two overlapping ovals to imply twisted body
    body1_cx, body1_cy = SIZE * 0.50, SIZE * 0.58
    body1_rx, body1_ry = SIZE * 0.30, SIZE * 0.20
    body2_cx, body2_cy = SIZE * 0.62, SIZE * 0.54
    body2_rx, body2_ry = SIZE * 0.26, SIZE * 0.26
    draw_ellipse_outline(draw, body1_cx, body1_cy, body1_rx, body1_ry)
    draw_ellipse_outline(draw, body2_cx, body2_cy, body2_rx, body2_ry)

    # Head positioned within the overlap
    head_cx, head_cy, head_r = SIZE * 0.44, SIZE * 0.50, SIZE * 0.11
    draw_head(draw, head_cx, head_cy, head_r)

    # Tail looping around like a pretzel
    tail_cx, tail_cy = SIZE * 0.70, SIZE * 0.64
    tail_rx, tail_ry = SIZE * 0.22, SIZE * 0.18
    draw_arc_outline(draw, tail_cx, tail_cy, tail_rx, tail_ry, start_deg=200, end_deg=30)

    downscale_and_save(img, "pretzel.png")
    return "pretzel.png"


def pose_laying_on_side() -> str:
    img, draw = create_canvas()

    # Body (horizontal oval)
    body_cx, body_cy = SIZE * 0.56, SIZE * 0.62
    body_rx, body_ry = SIZE * 0.34, SIZE * 0.22
    draw_ellipse_outline(draw, body_cx, body_cy, body_rx, body_ry)

    # Head on one end
    head_cx, head_cy, head_r = SIZE * 0.26, SIZE * 0.58, SIZE * 0.12
    draw_head(draw, head_cx, head_cy, head_r)

    # Legs (two short lines along lower edge to indicate side)
    leg_y = body_cy + body_ry * 0.2
    draw_line(draw, (body_cx - body_rx * 0.2, leg_y), (body_cx - body_rx * 0.2, leg_y + SIZE * 0.12))
    draw_line(draw, (body_cx + body_rx * 0.05, leg_y), (body_cx + body_rx * 0.05, leg_y + SIZE * 0.10))

    # Tail extended backward
    tail_cx, tail_cy = body_cx + body_rx * 0.9, body_cy + body_ry * 0.05
    tail_rx, tail_ry = body_rx * 0.7, body_ry * 0.6
    draw_arc_outline(draw, tail_cx, tail_cy, tail_rx, tail_ry, start_deg=330, end_deg=120)

    downscale_and_save(img, "laying_on_side.png")
    return "laying_on_side.png"


POSES = [
    pose_standing_all_fours,
    pose_standing_hind_legs,
    pose_loaf,
    pose_curled_up,
    pose_between_legs,
    pose_belly_up,
    pose_pretzel,
    pose_laying_on_side,
]


def main() -> None:
    ensure_output_dir()
    created = []
    for fn in POSES:
        created.append(fn())
    print("Created:")
    for name in created:
        print(" -", name)


if __name__ == "__main__":
    main()