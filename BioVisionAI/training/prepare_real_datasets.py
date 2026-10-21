"""Prepare real Kaggle datasets for BioVision AI training.

Run from BioVisionAI:
  python training/prepare_real_datasets.py --domains plant
  python training/prepare_real_datasets.py --domains plant --train

Kaggle CLI credentials must be configured first. Synthetic demo images are not used.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DOWNLOADS_DIR = BASE_DIR / "downloads"
DATASETS_DIR = BASE_DIR / "datasets"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SOURCES = {
    "plant": "emmarex/plantdisease",
    "human": "kmader/skin-cancer-mnist-ham10000",
    "animal": "muzammilwaseem/animal-disease-dataset",
}


def image_files(folder: Path) -> list[Path]:
    try:
        return [
            item for item in folder.iterdir()
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        ]
    except OSError:
        return []


def download_and_extract(domain: str) -> Path:
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "Kaggle CLI is missing. Install it with 'pip install kaggle' and "
            "configure your Kaggle API token before downloading datasets."
        )

    target = DOWNLOADS_DIR / domain
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", SOURCES[domain], "-p", str(target), "--force"],
        check=True,
    )
    archives = sorted(target.glob("*.zip"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not archives:
        raise RuntimeError(f"Kaggle did not produce a ZIP archive for {SOURCES[domain]}.")

    extracted = target / "extracted"
    extracted.mkdir(parents=True, exist_ok=True)
    # Use Python's zipfile to extract the archive without relying on shell utilities.
    import zipfile
    with zipfile.ZipFile(archives[0]) as archive:
        archive.extractall(extracted)
    return extracted


def copy_class_folders(source_root: Path, destination_root: Path) -> int:
    """Copy real images from leaf class folders into Keras flow_from_directory layout."""
    candidate_dirs = [
        path for path in source_root.rglob("*")
        if path.is_dir() and image_files(path)
    ]
    leaf_dirs = [
        folder for folder in candidate_dirs
        if not any(other != folder and folder in other.parents for other in candidate_dirs)
    ]
    copied = 0
    used_labels: set[str] = set()
    for folder in sorted(leaf_dirs):
        files = image_files(folder)
        if not files:
            continue
        label = folder.name.strip().replace("/", "_").replace("\\", "_")
        if label in used_labels:
            label = f"{folder.parent.name}_{label}"
        used_labels.add(label)
        out_dir = destination_root / label
        out_dir.mkdir(parents=True, exist_ok=True)
        for source in files:
            target = out_dir / source.name
            if target.exists():
                target = out_dir / f"{source.parent.name}_{source.name}"
            if not target.exists():
                shutil.copy2(source, target)
                copied += 1
    return copied


def prepare_ham10000(source_root: Path, destination_root: Path) -> int:
    """Organize HAM10000 dermatoscopic images by the dataset's diagnosis label (dx)."""
    metadata = next(
        (path for path in source_root.rglob("*.csv") if "metadata" in path.name.lower()),
        None,
    )
    if metadata is None:
        raise RuntimeError("HAM10000 metadata CSV was not found in the extracted dataset.")

    images_by_id: dict[str, Path] = {}
    for path in source_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images_by_id.setdefault(path.stem, path)

    copied = 0
    with metadata.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = {field.lower(): field for field in (reader.fieldnames or [])}
        image_col = fields.get("image_id")
        label_col = fields.get("dx")
        if not image_col or not label_col:
            raise RuntimeError("HAM10000 metadata must contain image_id and dx columns.")
        for row in reader:
            image_id = (row.get(image_col) or "").strip()
            label = (row.get(label_col) or "").strip().lower()
            source = images_by_id.get(image_id)
            if not image_id or not label or source is None:
                continue
            out_dir = destination_root / label
            out_dir.mkdir(parents=True, exist_ok=True)
            target = out_dir / f"{image_id}{source.suffix.lower()}"
            if not target.exists():
                shutil.copy2(source, target)
                copied += 1
    return copied


def prepare_domain(domain: str, replace_existing: bool = False) -> int:
    extracted = download_and_extract(domain)
    destination = DATASETS_DIR / {"plant": "plants", "human": "humans", "animal": "animals"}[domain]
    has_existing_images = destination.exists() and any(
        path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        for path in destination.rglob("*")
    )
    if has_existing_images and not replace_existing:
        raise RuntimeError(
            f"{destination} already contains images. To avoid mixing synthetic/demo or older "
            "data with the new source, move that folder away or rerun with --replace-existing."
        )
    if has_existing_images and replace_existing:
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    if domain == "human":
        copied = prepare_ham10000(extracted, destination)
    else:
        copied = copy_class_folders(extracted, destination)

    if copied == 0:
        raise RuntimeError(
            f"No images were organized for {domain}. Inspect {extracted} and arrange images "
            "into one subfolder per class before training."
        )
    print(f"{domain}: copied {copied} real images into {destination}")
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domains", nargs="+", choices=sorted(SOURCES), default=["plant"],
        help="Dataset domains to download and prepare (default: plant).",
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Train the selected model(s) after preparing the datasets.",
    )
    parser.add_argument(
        "--replace-existing", action="store_true",
        help="Replace existing images in selected dataset folders; use only if you intend to discard them.",
    )
    args = parser.parse_args()

    for domain in args.domains:
        try:
            prepare_domain(domain, replace_existing=args.replace_existing)
        except (RuntimeError, subprocess.CalledProcessError) as exc:
            print(f"ERROR preparing {domain}: {exc}", file=sys.stderr)
            return 1

    if args.train:
        for domain in args.domains:
            trainer = {
                "plant": "train_plants.py",
                "human": "train_humans.py",
                "animal": "train_animals.py",
            }[domain]
            subprocess.run([sys.executable, str(BASE_DIR / "training" / trainer)], cwd=BASE_DIR, check=True)
    else:
        print("Dataset preparation complete. Review class counts and licenses before training.")
        print("Run the corresponding training script(s) from the BioVisionAI directory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
