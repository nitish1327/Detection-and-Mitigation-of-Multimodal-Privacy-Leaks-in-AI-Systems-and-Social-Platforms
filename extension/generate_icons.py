"""Generate Privacy Sandbox extension icons."""

from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:
    raise SystemExit("Pillow required: pip install pillow")

OUT = Path(__file__).resolve().parent / "icons"
OUT.mkdir(exist_ok=True)

SIZES = (16, 48, 128)


def draw_icon(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (15, 20, 25, 255))
    draw = ImageDraw.Draw(img)
    margin = max(2, size // 8)
    cx, cy = size // 2, size // 2 + size // 16
    w = size - margin * 2
    h = int(w * 1.1)
    top = margin
    shield = [
        (cx, top),
        (cx + w // 2, top + h // 5),
        (cx + w // 2, top + h * 3 // 5),
        (cx, top + h),
        (cx - w // 2, top + h * 3 // 5),
        (cx - w // 2, top + h // 5),
    ]
    draw.polygon(shield, fill=(59, 130, 246, 255), outline=(37, 99, 235, 255))
    lock_h = max(4, size // 5)
    lock_w = max(3, size // 6)
    lx = cx - lock_w // 2
    ly = cy - lock_h // 4
    draw.rounded_rectangle(
        [lx, ly, lx + lock_w, ly + lock_h],
        radius=max(1, size // 16),
        fill=(255, 255, 255, 230),
    )
    arch_h = max(2, size // 8)
    draw.arc(
        [lx - 1, ly - arch_h, lx + lock_w + 1, ly + arch_h],
        start=180,
        end=0,
        fill=(255, 255, 255, 230),
        width=max(1, size // 12),
    )
    return img


def main() -> None:
    for size in SIZES:
        path = OUT / f"icon{size}.png"
        draw_icon(size).save(path, format="PNG")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
