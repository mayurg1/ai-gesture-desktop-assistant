# Algorithms Documentation — AI Gesture Desktop Assistant

---

## Overview

This document catalogs every algorithm used in the project. Each entry includes the purpose, mathematical formulation, complexity analysis, and engineering rationale.

---

## Algorithm 1: Coordinate Mapping (Normalized → Pixel)

### Purpose

Convert MediaPipe's normalized landmark coordinates (range 0.0–1.0) to absolute pixel coordinates for OpenCV drawing operations.

### When Used

Every frame, for every detected hand, for all 21 landmarks.

### Mathematical Idea

MediaPipe returns landmark positions as fractions of the input image dimensions. The top-left corner is `(0.0, 0.0)`, the bottom-right is `(1.0, 1.0)`. To draw on the image, we need integer pixel positions.

### Formula

```
pixel_x = round(normalized_x × frame_width)
pixel_y = round(normalized_y × frame_height)
```

Where:
- `normalized_x`, `normalized_y` ∈ [0.0, 1.0]
- `frame_width`, `frame_height` are the image dimensions in pixels
- `pixel_x`, `pixel_y` are integer pixel coordinates

### Implementation

```python
cx, cy = int(lm.x * w), int(lm.y * h)
```

### Time Complexity

O(1) per landmark. O(21) per hand. O(21 × n) for n hands.

### Space Complexity

O(1) — no additional memory allocated beyond the pixel coordinate variables.

### Advantages

- Simple and fast (single multiplication + cast).
- No external dependencies.
- Works for any image resolution.

### Limitations

- Assumes the input image is not cropped or padded. If MediaPipe preprocesses the image (e.g., letterboxing), the mapping would be incorrect.
- Integer rounding discards sub-pixel precision, but this is acceptable for visualization.

### Why This Algorithm?

It is the standard approach used in all MediaPipe examples and tutorials. There is no reason to use a more complex mapping for 2D visualization.

---

## Algorithm 2: FPS Calculation (Instantaneous)

### Purpose

Measure and display the frame rate to monitor real-time performance.

### When Used

Every frame, before drawing the overlay.

### Mathematical Idea

FPS (frames per second) is the reciprocal of the time difference between consecutive frames.

### Formula

```
Δt = t_current - t_previous
FPS = 1.0 / Δt
```

Where:
- `t_current` = `time.time()` at the start of the current frame
- `t_previous` = `time.time()` at the start of the previous frame
- `Δt` = time elapsed between frames (in seconds)

### Implementation

```python
now = time.time()
fps = 1.0 / (now - prev_time)
prev_time = now
```

### Time Complexity

O(1) per frame.

### Space Complexity

O(1) — two float variables.

### Advantages

- Minimal computation.
- Responds immediately to performance changes (no lag in the metric).
- Easy to understand and implement.

### Limitations

- **Noisy:** Instantaneous FPS fluctuates significantly frame-to-frame due to OS scheduling, camera timing jitter, and variable inference time.
- **Not smooth:** A single slow frame causes a sharp FPS drop, even if the average is stable.

### Alternative: Sliding Window Average

A more robust approach is to average FPS over the last N frames:

```
FPS_smooth = N / (t_current - t_previous_N)
```

This gives a stable reading but introduces latency in the metric (it takes N frames to reflect a real change).

### Why Instantaneous?

For Milestone 2 (visualization), instantaneous FPS is sufficient. The metric is used for debugging and performance monitoring, not for control logic. Smoothing will be added in Milestone 3 when the FPS value is used for cursor control timing.

---

## Algorithm 3: Hand Skeleton Rendering

### Purpose

Draw the 20 connections between the 21 hand landmarks to form a visible hand skeleton.

### When Used

Every frame, for every detected hand, after drawing the landmarks.

### Mathematical Idea

The hand skeleton is a graph with 21 nodes (landmarks) and 20 edges (connections). Each edge connects two landmark indices. The graph is drawn by iterating over all edges and drawing a line between the corresponding pixel coordinates.

### Connection Topology

```
Thumb:      0 ─ 1 ─ 2 ─ 3 ─ 4
Index:      0 ─ 5 ─ 6 ─ 7 ─ 8
Middle:     5 ─ 9 ─ 10 ─ 11 ─ 12
Ring:       9 ─ 13 ─ 14 ─ 15 ─ 16
Pinky:      13 ─ 17 ─ 18 ─ 19 ─ 20
Palm base:  0 ─ 17
```

### Implementation

```python
HAND_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index
    (5, 9), (9, 10), (10, 11), (11, 12),      # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),    # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),   # Pinky
    (0, 17),                                    # Palm base
])

for a, b in HAND_CONNECTIONS:
    cv2.line(frame, pts[a], pts[b], COLOR_SKELETON, 2)
```

### Time Complexity

O(20) per hand — 20 line drawing operations.

### Space Complexity

O(21) per hand — storing the pixel coordinates of all 21 landmarks.

### Why This Algorithm?

The hand skeleton is a fixed, well-known topology defined by the MediaPipe team based on hand anatomy. There is no computation involved — it is a lookup-and-draw operation.

---

## Algorithm 4: Landmark Highlighting

### Purpose

Visually distinguish the index finger tip (landmark 8) from other landmarks.

### When Used

Every frame, for every detected hand, during landmark drawing.

### Mathematical Idea

During the landmark iteration, check if the current index equals `INDEX_FINGER_TIP` (8). If yes, draw a larger circle with a different colour and a border. If no, draw a standard small circle.

### Implementation

```python
if i == INDEX_FINGER_TIP:
    cv2.circle(frame, (cx, cy), 10, COLOR_TIP, cv2.FILLED)
    cv2.circle(frame, (cx, cy), 10, (255, 255, 255), 2)  # white border
else:
    cv2.circle(frame, (cx, cy), 5, COLOR_LANDMARK, cv2.FILLED)
```

### Visual Properties

| Property | Normal Landmark | Index Tip (Landmark 8) |
|----------|----------------|------------------------|
| Radius | 5 px | 10 px |
| Fill | Green (0, 255, 0) | Red (0, 0, 255) |
| Border | None | White, 2px thickness |

### Time Complexity

O(1) per landmark — single integer comparison.

### Why This Algorithm?

Simple conditional rendering. No complex logic needed. The index finger tip is highlighted because it will be the primary control point for cursor movement in Milestone 3.

---

## Algorithm 5: Handedness Display

### Purpose

Show whether each detected hand is Left or Right, along with the model's confidence.

### When Used

Every frame, during the info overlay.

### Mathematical Idea

The `HandLandmarkerResult.handedness` field contains a list of `Classifications` objects. Each `Classifications` object contains a list of categories. For hand detection, there is exactly one category per hand: `"Left"` or `"Right"` with a confidence score between 0.0 and 1.0.

### Implementation

```python
handedness = result.handedness[i][0].category_name   # "Left" or "Right"
confidence = result.handedness[i][0].score            # e.g. 0.95
```

### Why This Approach?

This is the standard API for accessing handedness in the MediaPipe Tasks API. The nested indexing `[i][0]` is required because:
- `result.handedness` is a list indexed by hand (i = 0, 1, ...)
- `result.handedness[i]` is a `Classifications` object containing a list of categories
- `result.handedness[i][0]` is the top (and only) classification for that hand

---

## Algorithm 6: Tip Coordinate Display

### Purpose

Show the real-time pixel coordinates of the index finger tip on screen.

### When Used

Every frame, during the info overlay.

### Mathematical Idea

Extract landmark 8 from the hand's landmark list, convert to pixel coordinates using Algorithm 1, and format as a string.

### Implementation

```python
tip = hand_landmarks[INDEX_FINGER_TIP]
h, w, _ = frame.shape
coord = f"  Tip#8: ({int(tip.x * w)}, {int(tip.y * h)})"
```

### Why This Algorithm?

Direct application of the coordinate mapping algorithm. The displayed coordinates are useful for:
- Debugging the coordinate mapping.
- Understanding the relationship between hand position and screen position.
- Calibrating cursor control in Milestone 3.

---

## Algorithm Summary

| # | Algorithm | Type | Complexity | Used In |
|---|-----------|------|------------|---------|
| 1 | Coordinate Mapping | Math (multiplication) | O(1) per landmark | `draw_landmarks()`, `draw_info()` |
| 2 | FPS Calculation | Math (division) | O(1) per frame | `main()` |
| 3 | Hand Skeleton Rendering | Graph traversal | O(20) per hand | `draw_landmarks()` |
| 4 | Landmark Highlighting | Conditional rendering | O(1) per landmark | `draw_landmarks()` |
| 5 | Handedness Display | API data access | O(1) per hand | `draw_info()` |
| 6 | Tip Coordinate Display | Math + string formatting | O(1) per hand | `draw_info()` |

---

## Future Algorithms (Milestone 3+)

### Exponential Moving Average (EMA)

For smoothing cursor movement:

```
smooth_x = α × raw_x + (1 - α) × smooth_x_previous
smooth_y = α × raw_y + (1 - α) × smooth_y_previous
```

Where α (alpha) is the smoothing factor (0.0–1.0). Lower α = smoother but more lag.

### Screen Coordinate Scaling

For mapping camera coordinates to screen coordinates:

```
screen_x = (camera_x / camera_width) × screen_width
screen_y = (camera_y / camera_height) × screen_height
```

### Euclidean Distance

For gesture recognition (e.g., pinch detection):

```
distance = √((x₂ - x₁)² + (y₂ - y₁)²)
```

### Bounding Box Calculation

For defining the interactive region:

```
min_x = min(landmarks[0].x, landmarks[5].x, landmarks[17].x)
max_x = max(landmarks[4].x, landmarks[8].x, landmarks[12].x, landmarks[16].x, landmarks[20].x)
min_y = min(landmarks[0].y, ...)
max_y = max(landmarks[4].y, ...)