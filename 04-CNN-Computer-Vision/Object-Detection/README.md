# Object Detection

**Object Detection** involves identifying **what** is in an image and **where** it is located using bounding boxes.

## How it differs from Classification
- **Classification:** "There is a cat in this image."
- **Detection:** "There is a cat at coordinates (x, y, w, h)."

## Popular Architectures

### 1. Two-Stage Detectors (R-CNN Family)
- **Mechanism:** Generate regions first, then classify them.
- **Example:** Faster R-CNN.
- **Pros:** High accuracy.
- **Cons:** Slower inference.

### 2. One-Stage Detectors (YOLO, SSD)
- **Mechanism:** Predict bounding boxes and classes in a single pass.
- **Example:** YOLO (You Only Look Once), SSD (Single Shot Detector).
- **Pros:** Extremely fast, suitable for real-time video.
- **Cons:** Explaining small objects can be harder.

## Key Concepts
- **Bounding Box:** $[x, y, w, h]$.
- **Anchor Boxes:** Predefined boxes of different scales/ratios used as templates.
- **IoU (Intersection over Union):** Measure of overlap between predicted and ground truth boxes.
- **NMS (Non-Maximum Suppression):** Filtering redundant overlapping boxes.

## Metrics
- **mAP (Mean Average Precision):** The standard metric for object detection.
