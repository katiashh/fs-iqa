# Data and split selection

Image datasets are not bundled. Supply a directory of images and, optionally,
the KonIQ-10k MOS CSV with columns `image_name` and `MOS_zscore`.

| Input | Example path | Required |
|---|---|---|
| Images | `data/koniq/` | Yes, for inference |
| MOS table | `data/koniq10k_scores_and_distributions.csv` | For correlation evaluation |
| Evaluation manifest | `data/test_images.txt` | Recommended for a fixed evaluation split |

A manifest contains one image path relative to `--image-dir` per line:

```text
image_001.jpg
image_002.jpg
```

These names illustrate the format; this is not an official split. Use your
actual held-out images. The manifest's order is preserved unless `--shuffle`
is selected. Without a manifest, the CLI scans image files in the top level of
the directory, sorts them by filename, and applies `--start-index`, optional
seeded shuffling, and `--limit`, in that order. It does not infer a held-out split.
The selected list is recorded in the output JSON, together with its SHA-256.

The paper specifies an 80/20 split. The original demo instead slices the sorted
directory at index 8,000, then shuffles its validation portion without fixing a
seed inside the file. The released CSV has no image identifiers. Consequently,
the exact 1,000-image selection cannot be reconstructed from the archive alone.
The cleaned CLI can approximate that selection policy with
`--start-index 8000 --shuffle --seed 0 --limit 1000`, but this is not an exact
reproduction of the original split or CSV.

## Preprocessing

The cleaned CLI decodes RGB with Pillow, resizes with OpenCV bilinear
interpolation, scales pixels to `[0, 1]`, and converts to CHW. DBCNN applies
ImageNet mean/std normalization internally.

Defaults are **width 512, height 384**, following the main paper. The supplied
demo calls `cv2.resize(image, (384, 512))`, which instead produces width 384,
height 512. Use `--width 384 --height 512` to retain that demo orientation.
The CLI never silently switches between the two conventions.

MOS values are preserved exactly as supplied in the selected column. SRC is
invariant under positive affine MOS rescaling; MSE and PLCC are not inferred
from normalized predictions and raw MOS here.
