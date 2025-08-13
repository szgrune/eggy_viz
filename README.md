# Cat Photo Analyzer

Analyze a directory of cat photos and produce a CSV with:

- filename (relative to input dir)
- original_creation_date (from EXIF if available, else file mtime)
- average_hex_color (average RGB color of the image)
- is_selfie (front camera yes/no, or unknown if not determined)
- body_position (one of: standing on all fours, standing up on hind legs, loaf, curled up, between legs, belly up, pretzel, laying on side, other)

## Quick start

Install core dependencies (no ML):

```bash
pip install -r /workspace/requirements.txt
```

Run on your photo folder (non-recursive):

```bash
python /workspace/analyze_cats.py /workspace/eggy_photos --output /workspace/cat_analysis.csv
```

Recurse into subfolders and limit to first 50 images:

```bash
python /workspace/analyze_cats.py /workspace/eggy_photos --recursive --limit 50 --output /workspace/cat_analysis.csv
```

## Enable body position classification (optional, ML)

Install additional ML dependencies (this can be large and may take several minutes):

```bash
pip install -r /workspace/requirements-ml.txt
```

Then run with CLIP zero-shot classification enabled:

```bash
python /workspace/analyze_cats.py /workspace/eggy_photos --recursive --use-clip --batch-size 8 --device cpu --output /workspace/cat_analysis.csv
```

Notes:
- The first run will download model weights (`openai/clip-vit-base-patch32`).
- Use `--device cuda` if a CUDA GPU is available.
- If you do not enable `--use-clip`, `body_position` will be set to `other`.

## Heuristics and caveats

- Original date is taken from EXIF `DateTimeOriginal` when available, falling back to file modified time.
- Selfie detection is heuristic-based: looks for keywords (front/selfie/TrueDepth) in EXIF fields and uses focal length thresholds (≤3.0mm ≈ front, ≥3.6mm ≈ back). Unknowns are marked as `unknown`.
- Average color is computed over a resized image (`64x64`) for speed.
- Supported formats: jpg, jpeg, png, heic, heif, webp, bmp, tiff.