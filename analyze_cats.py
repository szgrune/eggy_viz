#!/usr/bin/env python3
import argparse
import csv
import os
import sys
import math
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Optional PIL imports
try:
    from PIL import Image, ImageOps, ExifTags  # type: ignore
except Exception:
    Image = None  # type: ignore
    ImageOps = None  # type: ignore
    ExifTags = None  # type: ignore

# Optional imports for extra EXIF parsing
try:
    import exifread  # type: ignore
except Exception:
    exifread = None  # type: ignore

try:
    import piexif  # type: ignore
except Exception:
    piexif = None  # type: ignore

# Optional: register HEIF/HEIC support
try:
    import pillow_heif  # type: ignore

    if Image is not None:
        pillow_heif.register_heif_opener()
except Exception:
    pillow_heif = None  # type: ignore

# Optional ML imports (lazy)
_CLIP_PIPELINE = None

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".bmp", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a photoset of cat images and produce a CSV with date, avg color, selfie flag, and body position classification.",
    )
    parser.add_argument("input_dir", help="Path to directory containing images")
    parser.add_argument("--output", default="cat_analysis.csv", help="Output CSV path")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    parser.add_argument("--workers", type=int, default=0, help="Number of workers (0 = single-threaded)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for ML classification")
    parser.add_argument(
        "--use-clip", action="store_true", help="Enable CLIP zero-shot classification for body position (requires transformers+torch)",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for ML (default cpu)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images processed (for quick tests)")
    parser.add_argument("--skip-avg-color", action="store_true", help="Skip computing average color")
    parser.add_argument("--skip-selfie", action="store_true", help="Skip selfie/front-camera heuristic")
    parser.add_argument("--skip-date", action="store_true", help="Skip date extraction")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging output")
    return parser.parse_args()


def find_images(input_dir: str, recursive: bool) -> List[str]:
    image_paths: List[str] = []
    if recursive:
        for root, _dirs, files in os.walk(input_dir):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    image_paths.append(os.path.join(root, f))
    else:
        for f in os.listdir(input_dir):
            full = os.path.join(input_dir, f)
            if not os.path.isfile(full):
                continue
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                image_paths.append(full)
    image_paths.sort()
    return image_paths


def open_image_for_processing(path: str) -> Optional["Image.Image"]:
    if Image is None or ImageOps is None:
        return None
    try:
        img = Image.open(path)
        # Convert to RGB, ignore alpha for average color consistency
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        return img
    except Exception:
        return None


def format_hex_color(r: float, g: float, b: float) -> str:
    r_i = max(0, min(255, int(round(r))))
    g_i = max(0, min(255, int(round(g))))
    b_i = max(0, min(255, int(round(b))))
    return f"#{r_i:02X}{g_i:02X}{b_i:02X}"


def compute_average_color(img: "Image.Image") -> str:
    # Resize to small size to speed up mean calculation while keeping representativeness
    try:
        small = img.resize((64, 64), resample=Image.BILINEAR)
    except Exception:
        small = img
    pixels = list(small.getdata())
    total_r = 0
    total_g = 0
    total_b = 0
    count = 0
    for p in pixels:
        if isinstance(p, tuple) and len(p) >= 3:
            r, g, b = p[:3]
        else:
            # grayscale
            r = g = b = int(p)
        total_r += int(r)
        total_g += int(g)
        total_b += int(b)
        count += 1
    if count == 0:
        return ""
    mean_r = total_r / count
    mean_g = total_g / count
    mean_b = total_b / count
    return format_hex_color(mean_r, mean_g, mean_b)


def _to_iso8601_from_exif_str(exif_datetime: str) -> Optional[str]:
    try:
        # Common EXIF format: YYYY:MM:DD HH:MM:SS
        dt = datetime.strptime(exif_datetime.strip(), "%Y:%m:%d %H:%M:%S")
        return dt.isoformat()
    except Exception:
        # Other odd formats - attempt a generic parse with replacements
        try:
            cleaned = exif_datetime.replace("/", ":").replace("-", ":")
            parts = cleaned.split()
            if parts and ":" in parts[0]:
                date_part = parts[0].replace(":", "-", 2)
                cleaned = " ".join([date_part] + parts[1:])
            dt = datetime.fromisoformat(cleaned)
            return dt.isoformat()
        except Exception:
            return None


def extract_exif_with_pil(img: "Image.Image") -> Dict[str, str]:
    exif_data: Dict[str, str] = {}
    if Image is None or ExifTags is None:
        return exif_data
    try:
        raw = img.getexif()
        if not raw:
            return exif_data
        tag_map = {ExifTags.TAGS.get(k, str(k)): v for k, v in raw.items()}
        for k, v in tag_map.items():
            try:
                if isinstance(v, bytes):
                    v = v.decode(errors="ignore")
                exif_data[str(k)] = str(v)
            except Exception:
                pass
    except Exception:
        pass
    return exif_data


def extract_exif_with_exifread(path: str) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if exifread is None:
        return data
    try:
        with open(path, "rb") as f:
            tags = exifread.process_file(f, details=False)
        for k, v in tags.items():
            data[k] = str(v)
    except Exception:
        pass
    return data


def get_original_creation_date(path: str, img: Optional["Image.Image"]) -> Optional[str]:
    # Priority: EXIF DateTimeOriginal > DateTimeDigitized > DateTime > file mtime
    candidates: List[str] = []
    exif_data: Dict[str, str] = {}
    if img is not None:
        exif_data.update(extract_exif_with_pil(img))
    exif_data.update(extract_exif_with_exifread(path))

    keys_priority = [
        "DateTimeOriginal",
        "EXIF DateTimeOriginal",
        "DateTimeDigitized",
        "EXIF DateTimeDigitized",
        "DateTime",
        "Image DateTime",
    ]
    for k in keys_priority:
        if k in exif_data and exif_data[k]:
            candidates.append(exif_data[k])

    for candidate in candidates:
        iso = _to_iso8601_from_exif_str(candidate)
        if iso:
            return iso

    try:
        stat = os.stat(path)
        mtime = datetime.fromtimestamp(stat.st_mtime).isoformat()
        return mtime
    except Exception:
        return None


def parse_focal_length_mm(exif_data: Dict[str, str]) -> Optional[float]:
    for key in [
        "FocalLength",
        "EXIF FocalLength",
        "FocalLengthIn35mmFilm",
        "EXIF FocalLengthIn35mmFilm",
    ]:
        if key in exif_data:
            val = exif_data[key]
            try:
                if "/" in val:
                    num, den = val.split("/", 1)
                    return float(num) / float(den)
                return float(val)
            except Exception:
                continue
    return None


def detect_selfie_front_camera(path: str, img: Optional["Image.Image"]) -> Optional[bool]:
    exif_data: Dict[str, str] = {}
    if img is not None:
        exif_data.update(extract_exif_with_pil(img))
    exif_data.update(extract_exif_with_exifread(path))

    # Check lens/camera hints
    text_fields = []
    for key in [
        "LensModel",
        "EXIF LensModel",
        "LensMake",
        "Make",
        "Model",
        "ImageDescription",
        "Software",
        "LensSpecification",
        "AuxLens",
    ]:
        if key in exif_data and exif_data[key]:
            text_fields.append(str(exif_data[key]).lower())

    joined = " ".join(text_fields)
    if any(x in joined for x in ["front", "selfie", "truedepth", "face time", "facetime"]):
        return True
    if any(x in joined for x in ["back camera", "rear camera", "main camera", "triple camera"]):
        return False

    focal_mm = parse_focal_length_mm(exif_data)
    if focal_mm is not None:
        # Heuristic thresholds (mm) for smartphone lenses; front cameras often ~2-3mm, back ~3.8-7mm
        if focal_mm <= 3.0:
            return True
        if focal_mm >= 3.6:
            return False

    # Unknown
    return None


def ensure_clip_pipeline(device: str):
    global _CLIP_PIPELINE
    if _CLIP_PIPELINE is not None:
        return _CLIP_PIPELINE
    try:
        from transformers import pipeline  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "transformers is not installed. Install optional ML dependencies from requirements-ml.txt and rerun with --use-clip"
        ) from e

    model_name = "openai/clip-vit-base-patch32"
    _CLIP_PIPELINE = pipeline(
        "zero-shot-image-classification", model=model_name, device=0 if device == "cuda" else -1
    )
    return _CLIP_PIPELINE


BodyClass = Tuple[str, str]


def get_body_position_classes() -> List[BodyClass]:
    # (label, natural language candidate)
    return [
        ("standing on all fours", "a photo of a cat standing on all fours"),
        ("standing up on hind legs", "a photo of a cat standing up on its hind legs"),
        ("loaf", "a photo of a cat in a loaf pose with paws tucked under its body"),
        ("curled up", "a photo of a cat curled up in a circle"),
        ("between legs", "a photo of a cat sitting between a person's legs"),
        ("belly up", "a photo of a cat lying on its back with its belly up"),
        ("pretzel", "a photo of a cat twisted like a pretzel"),
        ("laying on side", "a photo of a cat lying on its side"),
        ("other", "a photo of a cat in another position"),
    ]


def classify_body_positions_with_clip(images: List["Image.Image"], device: str) -> List[str]:
    pipeline = ensure_clip_pipeline(device)
    classes = get_body_position_classes()
    labels = [c[1] for c in classes]
    results = pipeline(images, candidate_labels=labels)
    predictions: List[str] = []
    # Handle single image case where pipeline returns dict rather than list
    if isinstance(results, dict):
        results = [results]
    for res in results:  # type: ignore
        # res: list of dicts with 'label' (candidate string) and 'score'
        if isinstance(res, list):
            best = max(res, key=lambda x: x["score"]) if res else None
            if best is None:
                predictions.append("other")
                continue
            label_str = best["label"]
        else:
            label_str = res.get("label", "")  # type: ignore
        # Map back to our canonical label
        mapped = next((name for name, cand in classes if cand == label_str), "other")
        predictions.append(mapped)
    return predictions


def analyze_batch(
    batch_paths: List[str],
    args: argparse.Namespace,
    csv_writer: csv.writer,
    log: bool = False,
) -> None:
    images: List[Optional["Image.Image"]] = []
    rows: List[Dict[str, Optional[str]]] = []

    for path in batch_paths:
        img = open_image_for_processing(path)
        images.append(img)

        avg_hex = None
        orig_date = None
        selfie: Optional[bool] = None

        if img is not None:
            if not args.skip_avg_color:
                try:
                    avg_hex = compute_average_color(img)
                except Exception:
                    avg_hex = None
            if not args.skip_date:
                try:
                    orig_date = get_original_creation_date(path, img)
                except Exception:
                    orig_date = None
            if not args.skip_selfie:
                try:
                    selfie = detect_selfie_front_camera(path, img)
                except Exception:
                    selfie = None
        else:
            # Could not open image; still try to get date from file mtime if allowed
            if not args.skip_date:
                try:
                    stat = os.stat(path)
                    orig_date = datetime.fromtimestamp(stat.st_mtime).isoformat()
                except Exception:
                    orig_date = None

        rows.append(
            {
                "filename": os.path.relpath(path, args.input_dir),
                "original_creation_date": orig_date,
                "average_hex_color": avg_hex,
                "is_selfie": ("yes" if selfie is True else ("no" if selfie is False else "unknown"))
                if not args.skip_selfie
                else None,
                "body_position": None,
            }
        )

    # Body position classification
    if args.use_clip and Image is not None:
        images_for_clip: List["Image.Image"] = []
        clip_indices: List[int] = []
        for idx, img in enumerate(images):
            if img is None:
                continue
            images_for_clip.append(img)
            clip_indices.append(idx)
        if images_for_clip:
            try:
                preds = classify_body_positions_with_clip(images_for_clip, args.device)
                for idx, pred in zip(clip_indices, preds):
                    rows[idx]["body_position"] = pred
            except Exception:
                for idx in clip_indices:
                    rows[idx]["body_position"] = "other"
        for img in images_for_clip:
            try:
                img.close()
            except Exception:
                pass
    else:
        for row in rows:
            row["body_position"] = "other"

    # Write rows and close all images
    for row in rows:
        csv_writer.writerow(
            [
                row.get("filename", ""),
                row.get("original_creation_date", ""),
                row.get("average_hex_color", ""),
                row.get("is_selfie", ""),
                row.get("body_position", ""),
            ]
        )

    for img in images:
        if img is not None:
            try:
                img.close()
            except Exception:
                pass


def main() -> None:
    args = parse_args()
    input_dir = os.path.abspath(args.input_dir)
    output_csv = os.path.abspath(args.output)

    image_paths = find_images(input_dir, args.recursive)
    if args.limit and args.limit > 0:
        image_paths = image_paths[: args.limit]

    if not image_paths:
        print("No images found.")
        sys.exit(1)

    if not args.quiet:
        print(f"Found {len(image_paths)} images. Writing to: {output_csv}")
        if args.use_clip:
            print("Body position classification enabled (CLIP). This may download model weights on first run.")
        else:
            print("Body position classification disabled. All positions will be 'other'. Use --use-clip to enable.")

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    # Prepare header
    header = [
        "filename",
        "original_creation_date",
        "average_hex_color",
        "is_selfie",
        "body_position",
    ]

    # Process in batches for memory efficiency
    batch_size = max(1, int(args.batch_size))

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        total = len(image_paths)
        for start in range(0, total, batch_size):
            end = min(total, start + batch_size)
            batch = image_paths[start:end]
            if not args.quiet:
                print(f"Processing images {start+1}-{end} of {total}...")
            analyze_batch(batch, args, writer)

    if not args.quiet:
        print("Done.")


if __name__ == "__main__":
    main()