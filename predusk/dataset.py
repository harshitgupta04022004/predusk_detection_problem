import os
import shutil
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

from config import DATA_ROOT, SPORTSMOT_HF_REPO, DATASET_YAML, SAMPLE_DIR


def _write_yaml(yolo_root: Path):
    cfg = {
        "path": str(yolo_root),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["person"],
    }
    with open(DATASET_YAML, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Dataset YAML written to {DATASET_YAML}")


def _already_prepared(yolo_root: Path) -> bool:
    train_imgs = list((yolo_root / "images" / "train").rglob("*.jpg")) if (yolo_root / "images" / "train").exists() else []
    val_imgs = list((yolo_root / "images" / "val").rglob("*.jpg")) if (yolo_root / "images" / "val").exists() else []
    return len(train_imgs) > 50 and len(val_imgs) > 10


def _get_sequence_names_from_hf() -> dict[str, list[str]]:
    """
    SportsMOT on Hugging Face stores only the split lists (text rows with
    sequence names). The real dataset is the MOT-format folder structure.
    """
    from datasets import load_dataset

    print("Loading SportsMOT split names from Hugging Face...")
    ds = load_dataset(SPORTSMOT_HF_REPO)

    split_map: dict[str, list[str]] = {}
    if "train" in ds:
        split_map["train"] = [row["text"].strip() for row in ds["train"] if row.get("text")]
    if "validation" in ds:
        split_map["val"] = [row["text"].strip() for row in ds["validation"] if row.get("text")]

    print(f"  train: {len(split_map.get('train', []))} sequences")
    print(f"  val:   {len(split_map.get('val', []))} sequences")
    return split_map


def _mot_gt_to_yolo(gt_path: Path, out_label_dir: Path, im_width: int, im_height: int):
    """
    Convert MOT gt.txt to YOLO labels.
    MOT row format:
    frame, id, x, y, w, h, conf, class, visibility
    """
    out_label_dir.mkdir(parents=True, exist_ok=True)
    frame_annotations: dict[int, list[str]] = {}

    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 9:
                continue

            frame_id = int(parts[0])
            x, y, w, h = map(float, parts[2:6])
            visibility = float(parts[8])

            if visibility < 0.1 or w <= 0 or h <= 0:
                continue

            xc = (x + w / 2.0) / im_width
            yc = (y + h / 2.0) / im_height
            wn = w / im_width
            hn = h / im_height

            xc = max(0.0, min(1.0, xc))
            yc = max(0.0, min(1.0, yc))
            wn = max(0.0, min(1.0, wn))
            hn = max(0.0, min(1.0, hn))

            frame_annotations.setdefault(frame_id, []).append(
                f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"
            )

    # Write labels for all frames in order
    max_frame = max(frame_annotations.keys(), default=0)
    for frame_id in range(1, max_frame + 1):
        lbl = out_label_dir / f"{frame_id:06d}.txt"
        lbl.write_text("\n".join(frame_annotations.get(frame_id, [])))


def _prepare_from_manual(dataset_root: Path, yolo_root: Path, max_train: int, max_val: int):
    """
    Expects the official SportsMOT MOT-Challenge structure:
      dataset/train/SEQ/img1/*.jpg
      dataset/train/SEQ/gt/gt.txt
      dataset/train/SEQ/seqinfo.ini
      dataset/val/SEQ/img1/*.jpg
      ...
    """
    import configparser

    print(f"Converting SportsMOT from {dataset_root}...")
    for split_name, max_s in [("train", max_train), ("val", max_val)]:
        split_dir = dataset_root / split_name
        if not split_dir.exists():
            print(f"  Skipping missing split: {split_dir}")
            continue

        sequences = sorted([d for d in split_dir.iterdir() if d.is_dir()])[:max_s]
        print(f"  {split_name}: converting {len(sequences)} sequences")

        for seq_dir in tqdm(sequences, desc=f"  {split_name}", unit="seq"):
            img_dir = seq_dir / "img1"
            gt_path = seq_dir / "gt" / "gt.txt"
            seqinfo = seq_dir / "seqinfo.ini"

            if not img_dir.exists():
                continue

            # Read image size from seqinfo.ini if available
            iw, ih = 1280, 720
            if seqinfo.exists():
                cfg = configparser.ConfigParser()
                cfg.read(seqinfo)
                iw = int(cfg["Sequence"].get("imWidth", iw))
                ih = int(cfg["Sequence"].get("imHeight", ih))

            img_out = yolo_root / "images" / split_name / seq_dir.name
            label_out = yolo_root / "labels" / split_name / seq_dir.name
            img_out.mkdir(parents=True, exist_ok=True)
            label_out.mkdir(parents=True, exist_ok=True)

            # Copy frames
            for f in sorted(img_dir.glob("*.jpg")):
                dst = img_out / f.name
                if not dst.exists():
                    shutil.copy2(f, dst)

            # Convert labels
            if gt_path.exists():
                _mot_gt_to_yolo(gt_path, label_out, iw, ih)

    print("Manual SportsMOT conversion done.")


def download_and_prepare(max_train_seqs: int = 20, max_val_seqs: int = 8) -> Path:
    yolo_root = DATA_ROOT / "sportsmot_yolo"

    if _already_prepared(yolo_root):
        print("Dataset already prepared, skipping conversion.")
        _write_yaml(yolo_root)
        return DATASET_YAML

    # Use Hugging Face only for split names.
    try:
        _ = _get_sequence_names_from_hf()
    except Exception as e:
        print(f"Could not read HF split names ({e}). Continuing with manual dataset conversion.")

    # Official SportsMOT must be available locally in MOT format.
    manual_path = DATA_ROOT / "raw" / "sportsmot" / "dataset"
    if not manual_path.exists():
        raise FileNotFoundError(
            f"SportsMOT dataset not found at {manual_path}\n"
            "Download the official SportsMOT release and extract it there in MOT format."
        )

    if not any(manual_path.rglob("seqinfo.ini")):
        raise FileNotFoundError(
            f"Found {manual_path}, but no MOT-format sequences were detected.\n"
            "Expected: dataset/train/SEQ/seqinfo.ini and dataset/train/SEQ/img1/*.jpg"
        )

    _prepare_from_manual(manual_path, yolo_root, max_train_seqs, max_val_seqs)
    _write_yaml(yolo_root)
    return DATASET_YAML