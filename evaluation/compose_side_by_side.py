"""Put rendered videos side by side under a title bar (for the README teaser).

    python evaluation/compose_side_by_side.py \\
        --videos renders/screwing_kuka/screwing_kuka_sharpa.mp4 renders/beam_assembly/beam_assembly_flexiv.mp4 \\
        --titles "Kuka+Sharpa" "Flexiv+Wuji" --out renders/teaser.mp4

Each video is scaled to a ``--tile`` square, the shorter ones hold their last
frame until the longest ends, and a title bar with one caption per tile goes on
top. Needs Pillow and imageio-ffmpeg.
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

# Pass --font (e.g. Blender's <blender>/<ver>/datafiles/fonts/Inter.woff2); else DejaVu.
FONT_CANDIDATES = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf"]


def duration(path: Path) -> float:
    out = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(path)],
                         capture_output=True, text=True).stderr
    hms = out.split("Duration: ")[1].split(",")[0]
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def load_font(size: int, path: str | None):
    for cand in ([path] if path else []) + FONT_CANDIDATES:
        try:
            font = ImageFont.truetype(cand, size)
            if "Inter" in cand:
                try:
                    font.set_variation_by_name("SemiBold")
                except Exception:
                    pass
            return font
        except Exception:
            continue
    return ImageFont.load_default(size)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--titles", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=960)
    ap.add_argument("--bar", type=int, default=110, help="title bar height (px)")
    ap.add_argument("--font", default=None)
    ap.add_argument("--font-size", type=int, default=50)
    ap.add_argument("--bg", default="#2b2d31")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--crf", type=int, default=18)
    args = ap.parse_args()
    if len(args.titles) != len(args.videos):
        raise SystemExit("one title per video")

    n, t = len(args.videos), args.tile
    bar = Image.new("RGB", (n * t, args.bar), args.bg)
    draw = ImageDraw.Draw(bar)
    font = load_font(args.font_size, args.font)
    for i, title in enumerate(args.titles):
        x0, y0, x1, y1 = draw.textbbox((0, 0), title, font=font)
        draw.text((i * t + (t - (x1 - x0)) / 2 - x0, (args.bar - (y1 - y0)) / 2 - y0),
                  title, font=font, fill="#f2f2f2")

    longest = max(duration(Path(v)) for v in args.videos)
    with tempfile.TemporaryDirectory() as tmp:
        bar_png = Path(tmp) / "bar.png"
        bar.save(bar_png)
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error"]
        for v in args.videos:
            cmd += ["-i", v]
        cmd += ["-loop", "1", "-i", str(bar_png)]
        chains = []
        for i, v in enumerate(args.videos):
            pad = longest - duration(Path(v)) + 0.05
            chains.append(f"[{i}:v]fps={args.fps},scale={t}:{t}:flags=lanczos,setsar=1,"
                          f"tpad=stop_mode=clone:stop_duration={pad:.3f}[v{i}]")
        tiles = "".join(f"[v{i}]" for i in range(n))
        chains.append(f"{tiles}hstack=inputs={n}[row]" if n > 1 else "[v0]copy[row]")
        chains.append(f"[{n}:v][row]vstack=inputs=2,format=yuv420p[out]")
        cmd += ["-filter_complex", ";".join(chains), "-map", "[out]", "-t", f"{longest:.3f}",
                "-r", str(args.fps), "-c:v", "libx264", "-crf", str(args.crf), "-preset", "slow",
                "-movflags", "+faststart", args.out]
        subprocess.run(cmd, check=True)
    print(f"wrote {args.out} ({n * t}x{t + args.bar}, {longest:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
