import csv
from pathlib import Path

from training.prepare_real_datasets import copy_class_folders, prepare_ham10000


def test_copy_class_folders_uses_real_class_folder_names(tmp_path):
    source = tmp_path / "raw"
    destination = tmp_path / "prepared"
    (source / "Tomato___Early_blight").mkdir(parents=True)
    (source / "Tomato___healthy").mkdir(parents=True)
    (source / "Tomato___Early_blight" / "one.jpg").write_bytes(b"image")
    (source / "Tomato___healthy" / "two.jpg").write_bytes(b"image")

    copied = copy_class_folders(source, destination)

    assert copied == 2
    assert (destination / "Tomato___Early_blight" / "one.jpg").exists()
    assert (destination / "Tomato___healthy" / "two.jpg").exists()


def test_ham10000_preparation_uses_metadata_diagnosis_labels(tmp_path):
    source = tmp_path / "ham"
    source.mkdir()
    (source / "HAM10000_images_part_1").mkdir()
    (source / "HAM10000_images_part_1" / "ISIC_001.jpg").write_bytes(b"image")
    metadata = source / "HAM10000_metadata.csv"
    with metadata.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "dx"])
        writer.writeheader()
        writer.writerow({"image_id": "ISIC_001", "dx": "mel"})

    destination = tmp_path / "prepared"
    copied = prepare_ham10000(source, destination)

    assert copied == 1
    assert (destination / "mel" / "ISIC_001.jpg").exists()
