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
_CLIP_IMAGE_MODEL = None
_CLIP_IMAGE_PROCESSOR = None

# Global reference embeddings (if provided via --ref-dir)
_REF_LABELS: List[str] = []
_REF_EMB = None  # torch.Tensor or None

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
        "--use-clip", action="store_true", help="Enable CLIP classification for body position (requires transformers+torch)",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for ML (default cpu)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images processed (for quick tests)")
    parser.add_argument("--skip-avg-color", action="store_true", help="Skip computing average color")
    parser.add_argument("--skip-selfie", action="store_true", help="Skip selfie/front-camera heuristic")
    parser.add_argument("--skip-date", action="store_true", help="Skip date extraction")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging output")
    parser.add_argument(
        "--ref-dir",
        default=None,
        help=(
            "Directory of reference body position images. Use subfolders as labels, e.g., 'loaf/', 'belly_up/'. "
            "If provided with --use-clip, images will be classified by similarity to reference images."
        ),
    )
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

# --- New: reference-based CLIP embedding classification ---

def ensure_clip_image_model(device: str):
    global _CLIP_IMAGE_MODEL, _CLIP_IMAGE_PROCESSOR
    if _CLIP_IMAGE_MODEL is not None and _CLIP_IMAGE_PROCESSOR is not None:
        return _CLIP_IMAGE_MODEL, _CLIP_IMAGE_PROCESSOR
    try:
        import torch  # type: ignore
        from transformers import CLIPModel, CLIPProcessor  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "torch/transformers not installed. Install requirements-ml.txt and use --use-clip to enable reference-based classification."
        ) from e
    model_name = "openai/clip-vit-base-patch32"
    _CLIP_IMAGE_MODEL = CLIPModel.from_pretrained(model_name)
    _CLIP_IMAGE_PROCESSOR = CLIPProcessor.from_pretrained(model_name)
    if device == "cuda":
        _CLIP_IMAGE_MODEL = _CLIP_IMAGE_MODEL.to("cuda")
    _CLIP_IMAGE_MODEL.eval()
    return _CLIP_IMAGE_MODEL, _CLIP_IMAGE_PROCESSOR


def canonicalize_label(raw: str) -> str:
    name = raw.strip().lower().replace("_", " ").replace("-", " ")
    name = " ".join(name.split())
    synonyms = {
        "all fours": "standing on all fours",
        "on all fours": "standing on all fours",
        "four legs": "standing on all fours",
        "hind legs": "standing up on hind legs",
        "standing hind legs": "standing up on hind legs",
        "loaf": "loaf",
        "bread loaf": "loaf",
        "belly up": "belly up",
        "belly-up": "belly up",
        "on back": "belly up",
        "curled": "curled up",
        "curled up": "curled up",
        "curl": "curled up",
        "side": "laying on side",
        "lying on side": "laying on side",
        "laying on side": "laying on side",
        "between legs": "between legs",
        "pretzel": "pretzel",
        "other": "other",
    }
    # direct match
    if name in synonyms:
        return synonyms[name]
    # if name already matches one of our canonical labels, return
    for canonical, _ in get_body_position_classes():
        if name == canonical:
            return name
    # best-effort mapping based on keywords
    if "hind" in name:
        return "standing up on hind legs"
    if "four" in name or "all fours" in name or "all four" in name:
        return "standing on all fours"
    if "belly" in name or "back" in name:
        return "belly up"
    if "curl" in name:
        return "curled up"
    if "side" in name:
        return "laying on side"
    if "between" in name and "leg" in name:
        return "between legs"
    if "loaf" in name:
        return "loaf"
    if "pretzel" in name:
        return "pretzel"
    return "other"


def _list_images_in_dir(dir_path: str) -> List[str]:
    paths: List[str] = []
    if not os.path.isdir(dir_path):
        return paths
    for root, _dirs, files in os.walk(dir_path):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                paths.append(os.path.join(root, f))
    paths.sort()
    return paths


def load_reference_embeddings(ref_dir: str, device: str) -> None:
    # Build label -> list of image paths, using subfolder name as label
    global _REF_LABELS, _REF_EMB
    _REF_LABELS = []
    _REF_EMB = None

    if Image is None:
        raise RuntimeError("Pillow is required to load reference images.")

    import torch  # type: ignore

    model, processor = ensure_clip_image_model(device)

    # Collect subdirectories as labels; if none, try flat files and infer from filename prefix
    label_to_paths: Dict[str, List[str]] = {}

    subdirs = [d for d in os.listdir(ref_dir) if os.path.isdir(os.path.join(ref_dir, d))]
    if subdirs:
        for d in subdirs:
            label_raw = d
            label = canonicalize_label(label_raw)
            paths = _list_images_in_dir(os.path.join(ref_dir, d))
            if not paths:
                continue
            label_to_paths.setdefault(label, []).extend(paths)
    else:
        # Flat: use filename stem (no extension) as label
        for p in _list_images_in_dir(ref_dir):
            base = os.path.basename(p)
            stem, _ext = os.path.splitext(base)
            label = canonicalize_label(stem)
            label_to_paths.setdefault(label, []).append(p)

    if not label_to_paths:
        raise RuntimeError("No reference images found in ref-dir")

    labels: List[str] = sorted(label_to_paths.keys())
    # Compute mean embedding per label
    label_embs: List[torch.Tensor] = []

    with torch.no_grad():
        for label in labels:
            paths = label_to_paths[label]
            imgs: List[Image.Image] = []
            for p in paths:
                img = open_image_for_processing(p)
                if img is not None:
                    imgs.append(img)
            if not imgs:
                # If none could be loaded, skip this label
                continue
            # Batch through processor
            embs_per_label: List[torch.Tensor] = []
            batch_size = 8
            for i in range(0, len(imgs), batch_size):
                batch_imgs = imgs[i : i + batch_size]
                inputs = processor(images=batch_imgs, return_tensors="pt")
                if device == "cuda":
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                outputs = model.get_image_features(**inputs)
                # Normalize
                outputs = torch.nn.functional.normalize(outputs, p=2, dim=-1)
                embs_per_label.append(outputs)
            # Close images
            for im in imgs:
                try:
                    im.close()
                except Exception:
                    pass
            if not embs_per_label:
                continue
            all_embs = torch.cat(embs_per_label, dim=0)
            mean_emb = all_embs.mean(dim=0, keepdim=True)
            mean_emb = torch.nn.functional.normalize(mean_emb, p=2, dim=-1)
            label_embs.append(mean_emb)

    if not label_embs:
        raise RuntimeError("Failed to compute reference embeddings")

    _REF_LABELS = labels
    _REF_EMB = torch.cat(label_embs, dim=0)  # shape: (num_labels, dim)


def classify_with_clip_refs(images: List["Image.Image"], device: str) -> List[str]:
    # Requires _REF_LABELS and _REF_EMB be populated
    global _REF_LABELS, _REF_EMB
    if _REF_EMB is None or not _REF_LABELS:
        # Fallback to zero-shot text prompts
        return classify_body_positions_with_clip(images, device)

    import torch  # type: ignore

    model, processor = ensure_clip_image_model(device)

    preds: List[str] = []
    with torch.no_grad():
        batch_size = 8
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i : i + batch_size]
            inputs = processor(images=batch_imgs, return_tensors="pt")
            if device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            outputs = model.get_image_features(**inputs)
            outputs = torch.nn.functional.normalize(outputs, p=2, dim=-1)  # (N, D)
            # Cosine similarity with label means (L, D)
            sims = outputs @ _REF_EMB.T  # (N, L)
            top_idx = sims.argmax(dim=1).tolist()
            preds.extend([_REF_LABELS[j] for j in top_idx])
    return preds


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
                # Prefer reference-based if ref embeddings available
                preds = (
                    classify_with_clip_refs(images_for_clip, args.device)
                    if _REF_EMB is not None
                    else classify_body_positions_with_clip(images_for_clip, args.device)
                )
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
            if args.ref_dir:
                print("Body position classification enabled with reference images.")
            else:
                print("Body position classification enabled (CLIP zero-shot prompts).")
        else:
            print("Body position classification disabled. All positions will be 'other'. Use --use-clip to enable.")

    # If using refs, attempt to load embeddings once
    if args.use_clip and args.ref_dir:
        ref_dir = os.path.abspath(args.ref_dir)
        if not os.path.isdir(ref_dir):
            print(f"Reference directory not found: {ref_dir}")
        else:
            try:
                load_reference_embeddings(ref_dir, args.device)
                if not args.quiet:
                    print(f"Loaded {len(_REF_LABELS)} reference labels from {ref_dir}: {', '.join(_REF_LABELS)}")
            except Exception as e:
                print(f"Warning: failed to load reference embeddings: {e}. Falling back to zero-shot prompts.")

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    header = [
        "filename",
        "original_creation_date",
        "average_hex_color",
        "is_selfie",
        "body_position",
    ]

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