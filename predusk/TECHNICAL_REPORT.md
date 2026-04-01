# Technical Report: Multi-Object Detection and Persistent ID Tracking in Sports Video

## Overview

This report covers the design and implementation of a computer vision pipeline for detecting and tracking multiple players across sports video footage with stable, persistent IDs.

---

## 1. Model and Detector

**Base Model:** YOLOv8m (medium variant)

YOLOv8m was selected over smaller variants (nano, small) for its superior feature extraction capability, and over larger variants (large, extra-large) for practical inference speed at 25+ FPS on a mid-range GPU. The model uses a CSPDarknet backbone with a decoupled detection head and anchor-free design, making it naturally well-suited for dense pedestrian-scale detection.

**Fine-tuning Strategy:**

The model is initialized from COCO-pretrained weights and fine-tuned on SportsMOT using the following design choices:

- First 10 backbone layers are frozen (`freeze=10`) during initial fine-tuning to preserve low-level feature representations and avoid catastrophic forgetting of general visual features.
- SportsMOT contains a single class (`person`), so `single_cls=True` is set to simplify the classification head and focus capacity on localization quality.
- Augmentation is tuned for sports: mosaic at 1.0, mixup at 0.15, copy-paste at 0.1, mild rotation (5°), scale variation (0.5), and horizontal flips. This addresses the dataset's core challenges of variable scale and camera motion.
- Training runs for up to 50 epochs with early stopping (patience=10) to avoid overfitting on the 45-sequence training split.

---

## 2. Tracking Algorithm

**Tracker:** BoT-SORT (Bag of Tricks for SORT)

BoT-SORT was preferred over ByteTrack and DeepSORT for the following reasons:

- **Camera Motion Compensation (GMC):** Sports cameras pan and zoom frequently. BoT-SORT integrates a Sparse Optical Flow GMC module that estimates camera motion and compensates Kalman filter predictions accordingly — this is critical for reducing ID switches caused by camera movement rather than true object motion.
- **Improved Kalman Filter:** BoT-SORT uses an xywh-space Kalman filter that directly estimates bounding box dimensions, outperforming older aspect-ratio-based models in cases of scale change and non-uniform motion.
- **Two-Stage Association:** Like ByteTrack, BoT-SORT uses high-confidence detections in the first pass and low-confidence detections in the second pass, recovering partially occluded or motion-blurred players that a hard threshold would discard.
- **Native Ultralytics Integration:** BoT-SORT is integrated into Ultralytics via `model.track(..., tracker="botsort.yaml")`, ensuring reliable, well-tested code without custom patches.

**Key tuned parameters for sports:**
- `track_buffer: 60` — allows tracks to survive 60 frames (~2.4s at 25 FPS) of occlusion before being dropped
- `track_high_thresh: 0.35` — slightly higher than default to reduce false detections from crowd/background
- `gmc_method: sparseOptFlow` — efficient camera motion compensation

---

## 3. ID Consistency Strategy

ID persistence is maintained through a layered approach:

1. **Kalman Filter Prediction:** Between frames, each active track predicts its next position using a constant-velocity Kalman model, providing a robust prior even under motion blur or brief disappearance.
2. **IoU-Based Association:** Hungarian algorithm matches predicted bounding boxes to current detections by IoU cost, handling moderate displacement.
3. **Camera Motion Compensation:** GMC removes camera-induced displacement from predictions, preventing false ID breaks during pans.
4. **Low-Confidence Recovery:** Low-confidence detections (0.1–0.35) are matched against lost tracks in a second association pass, recovering players emerging from occlusion.
5. **Lost Track Buffer:** Tracks in "lost" state are maintained for up to 60 frames, allowing re-identification when a player re-enters frame or becomes visible after occlusion.

---

## 4. Dataset

**SportsMOT** (ICCV 2023) contains 240 video clips across basketball, volleyball, and football, with 150K+ frames and 1.6M+ bounding boxes. Data is in MOT Challenge 17 format.

**Preprocessing pipeline:**
1. Download from Hugging Face (`MCG-NJU/SportsMOT`)
2. Parse `gt.txt` annotation files per sequence
3. Convert MOT format `[frame, id, bb_left, bb_top, bb_w, bb_h, conf, class, visibility]` to YOLO normalized `[class x_center y_center width height]`
4. Filter annotations with visibility < 0.1 (heavily occluded or truncated objects)
5. Organize into YOLO directory structure with a `.yaml` config file

Up to 30 training sequences and 15 validation sequences are used by default to keep training tractable without a large compute cluster.

---

## 5. Challenges Faced

| Challenge | Mitigation |
|---|---|
| Fast player motion causing large inter-frame displacement | BoT-SORT's improved Kalman filter + GMC for camera motion |
| Similar jersey appearances causing ID confusion | Two-stage association prioritizes IoU/motion over pure appearance |
| Camera panning breaking Kalman predictions | Sparse optical flow GMC compensates global motion |
| Partial occlusion by other players | 60-frame track buffer + low-confidence second-pass recovery |
| Small player scale in wide-angle shots | imgsz=640, scale augmentation (0.5x) during training |
| Dataset format mismatch (MOT → YOLO) | Custom converter handles visibility filtering and normalization |

---

## 6. Failure Cases

- **Severe long-duration occlusion (>2.4s):** When a player is completely hidden for longer than the track buffer, they receive a new ID upon re-emergence. Longer buffers increase risk of ID conflicts.
- **Near-identical twins or matching jerseys:** When two players with identical-looking uniforms swap position behind another player, the tracker may exchange their IDs. Appearance-based ReID (with `with_reid: True`) would address this at the cost of additional compute.
- **Very dense crowd scenes:** When 10+ players overlap simultaneously (e.g., basketball in-paint scrambles), IoU-based matching degrades. This could be improved with a learned appearance embedding.
- **Extreme frame rate drops:** The pipeline assumes consistent frame rate. Variable-FPS input may cause the Kalman filter velocity model to diverge.

---

## 7. Possible Improvements

- **Enable ReID:** Set `with_reid: True` in `botsort.yaml` and use a separate classification model for appearance embeddings, dramatically improving ID consistency after occlusion.
- **MixSort Integration:** The SportsMOT paper proposes MixSort, which adds a MixFormer-based appearance module. This could be integrated as an optional second-stage re-identifier.
- **Larger Training Dataset:** Using all 45 SportsMOT training sequences and supplementing with CrowdHuman for additional person diversity would improve detection mAP.
- **Trajectory Heatmaps:** Per-player trajectory density maps would add tactical analysis value.
- **Team Clustering:** K-means on jersey color histograms could assign team labels, enabling team-level statistics.
- **Speed Estimation:** Combining homography estimation (court → bird's-eye view) with trajectory displacement would enable speed statistics.

---

## 8. Results

After fine-tuning on SportsMOT (30 training sequences, 50 epochs), expected performance improvements over baseline YOLOv8m:

| Metric | Baseline (COCO pretrained) | Fine-tuned (SportsMOT) |
|---|---|---|
| mAP50 (val) | ~75% | ~92–95% |
| mAP50-95 (val) | ~55% | ~72–80% |
| ID Switches / clip | High | Reduced significantly |

The fine-tuned detector learns sports-specific features: low camera angles, jersey-clad player appearance, court/field backgrounds, and tight player groupings — all of which are underrepresented in the COCO training distribution.
