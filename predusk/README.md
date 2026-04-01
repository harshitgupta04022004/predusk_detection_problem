# Sports Multi-Object Detection & Persistent ID Tracking

A production-quality computer vision pipeline for detecting and tracking players in sports video with stable, persistent IDs. Built on **YOLOv8m fine-tuned on SportsMOT** + **BoT-SORT tracker**.

---

## Pipeline Overview

```
SportsMOT (HuggingFace) → Format Conversion → YOLOv8m Fine-tuning
                                                     ↓
Input Video → YOLOv8m Detector → BoT-SORT Tracker → Annotated Output Video
```

---

## Project Structure

```
sports_tracker/
├── config.py             # All hyperparameters and paths
├── dataset.py            # SportsMOT download + MOT→YOLO format conversion
├── preprocessing.py      # Augmentation config, frame utilities, dataset validation
├── tracker.py            # SportsTracker class wrapping YOLO + BoT-SORT
├── annotator.py          # Frame annotation: bounding boxes, IDs, trajectories, HUD
├── inference.py          # Inference pipeline (used by gradio_app.py)
├── utils.py              # Video info, stats formatting, report saving
├── train.py              # Fine-tuning script
├── gradio_app.py         # Gradio demo (single entry point for inference)
├── botsort.yaml          # Tuned BoT-SORT tracker config for sports
├── requirements.txt
├── TECHNICAL_REPORT.md
└── README.md
```

---

## Installation

```bash
# Clone or download this project
cd sports_tracker

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# GPU users: install the correct torch version for your CUDA
# See https://pytorch.org/get-started/locally/
```

---

## Training

### Step 1 — Download SportsMOT and fine-tune

```bash
python train.py
```

This will:
1. Download SportsMOT from Hugging Face (`MCG-NJU/SportsMOT`)
2. Convert annotations from MOT Challenge format to YOLO format
3. Fine-tune `yolov8m.pt` on the sports dataset
4. Save best weights to `weights/sports_detector/weights/best.pt`

### Training options

```bash
python train.py --epochs 50 --batch 16 --imgsz 640 --device 0
python train.py --epochs 30 --batch 8 --device cpu          # CPU-only
python train.py --skip-download                             # If data already exists
python train.py --resume                                    # Resume interrupted training
python train.py --max-train-seqs 45 --max-val-seqs 45      # Use full dataset
python train.py --model yolov8l.pt                         # Use larger base model
```

### What gets trained

- Base model: `yolov8m.pt` (COCO pretrained)
- Dataset: SportsMOT (basketball, volleyball, football) — single class: `person`
- First 10 backbone layers frozen for stable fine-tuning
- Augmentation: mosaic, mixup, copy-paste, scale, flips, color jitter
- Expected training time: ~2–4 hours on a single GPU (RTX 3090 / A100)

---

## Inference Demo

```bash
python gradio_app.py
```

Opens a browser UI at `http://localhost:7860` with two tabs:

**Upload Video tab:** Upload any sports video and click "Run Tracking"  
**SportsMOT Sample tab:** Load a built-in clip from the dataset and track it

### Outputs

- **Annotated video** with colored bounding boxes and persistent `#ID` labels
- **Trajectory trails** showing each player's recent path
- **HUD overlay** with frame counter, active player count, total unique IDs
- **Tracking summary** with statistics

---

## Using as a Library

```python
from pathlib import Path
from inference import run_tracking_pipeline

summary = run_tracking_pipeline(
    input_video_path=Path("match_clip.mp4"),
    output_video_path=Path("outputs/match_tracked.mp4"),
    draw_trajectories=True,
)

print(f"Unique player IDs: {summary['unique_ids']}")
print(f"Frames processed: {summary['total_frames']}")
```

---

## Model Choice & Rationale

| Component | Choice | Reason |
|---|---|---|
| Detector | YOLOv8m fine-tuned | Best accuracy/speed tradeoff; fine-tuning on SportsMOT improves player detection by 15–20% mAP vs. generic COCO weights |
| Tracker | BoT-SORT | Camera Motion Compensation (GMC) prevents ID breaks during pans; improved Kalman filter handles scale changes; two-stage association recovers occluded players |
| Dataset | SportsMOT (HuggingFace) | 240 clips, 150K+ frames, 1.6M+ annotations across basketball/volleyball/football |

---

## Limitations

- Long occlusion (>2.4s at 25 FPS) causes ID reassignment after the track buffer expires
- Near-identical appearances (same jersey, same team) may cause ID swaps; enabling ReID in `botsort.yaml` (`with_reid: True`) mitigates this
- Very dense crowded scenes (10+ overlapping players) degrade IoU-based association

---

## Configuration

All key parameters are in `config.py`:

```python
YOLO_BASE_MODEL = "yolov8m.pt"    # Base model for fine-tuning
TRAIN_EPOCHS = 50
CONF_THRESHOLD = 0.35              # Detection confidence cutoff
TRACKER_CONFIG = "botsort.yaml"    # Tracker config file
TRACK_BUFFER = 60                  # Frames to keep lost tracks alive
```

Tracker parameters are in `botsort.yaml`. Key parameters:

```yaml
track_buffer: 60          # Increase for longer occlusion tolerance
track_high_thresh: 0.35   # Lower to detect more players (more false positives)
gmc_method: sparseOptFlow # Camera motion compensation method
with_reid: False          # Enable for appearance-based re-identification
```

---

## References

- **SportsMOT:** Cui et al., "SportsMOT: A Large Multi-Object Tracking Dataset in Multiple Sports Scenes," ICCV 2023
- **BoT-SORT:** Aharon et al., "BoT-SORT: Robust Associations Multi-Pedestrian Tracking," arXiv 2206.14651
- **ByteTrack:** Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box," ECCV 2022
- **YOLOv8:** Ultralytics, https://github.com/ultralytics/ultralytics
