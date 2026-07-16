# Architecture Documentation — AI Gesture Desktop Assistant

---

## Project Overview

An AI-powered desktop interaction system that uses hand gestures detected from a webcam to control the computer cursor and perform actions. Built with the MediaPipe Tasks API (NOT `mp.solutions`) and OpenCV.

---

## Current Folder Structure

```
hand/
│
├── test.py                  # Main entry point — camera, detection, visualization
├── camera_test.py           # Standalone webcam verification script
├── README.md                # Project overview
├── .gitignore
│
├── models/
│   └── hand_landmarker.task # MediaPipe HandLandmarker model file (16-bit float)
│
└── docs/
    ├── engineering-journal.md    # Session-by-session development log
    ├── architecture.md           # This file — system architecture
    ├── algorithms.md             # Algorithm documentation
    ├── debugging.md              # Bug database and troubleshooting
    ├── roadmap.md                # Milestone tracking and future plans
    └── interview-notes.md        # Interview preparation
```

**Note:** This is a single-file project architecture. The core logic lives entirely in `test.py`. Future milestones will modularize this into `core/` modules (camera, detector, renderer).

---

## Module Responsibilities (Current)

### `test.py`

**Purpose:** Main application entry point and the only module.

**Responsibilities:**
1. Initialize the webcam via `cv2.VideoCapture(0)`.
2. Load the MediaPipe HandLandmarker model from `models/hand_landmarker.task`.
3. Run the frame capture → detection → visualization loop.
4. Clean up resources on exit (release camera, destroy windows).

**Sub-components (all within `test.py`):**

| Component | Type | Responsibility |
|-----------|------|----------------|
| `HAND_CONNECTIONS` | Constant | Defines the 20-edge hand skeleton topology |
| `draw_landmarks()` | Function | Renders 21 landmarks + skeleton lines on the frame |
| `draw_info()` | Function | Renders text overlay (FPS, handedness, confidence, coordinates) |
| `main()` | Function | Orchestrator — manages the capture/detect/draw loop |

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    test.py (main loop)                      │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐      │
│  │  Camera   │    │  BGR →   │    │  mp.Image        │      │
│  │  (cv2)    │───►│  RGB     │───►│  (MediaPipe      │      │
│  │           │    │          │    │   Image wrapper)  │      │
│  └──────────┘    └──────────┘    └────────┬─────────┘      │
│         │                                  │                │
│         │                                  ▼                │
│         │                          ┌──────────────────┐      │
│         │                          │  HandLandmarker  │      │
│         │                          │  .detect()       │      │
│         │                          └────────┬─────────┘      │
│         │                                   │                │
│         │                                   ▼                │
│         │                          ┌──────────────────┐      │
│         │                          │  HandLandmarker  │      │
│         │                          │  Result          │      │
│         │                          └────────┬─────────┘      │
│         │                                   │                │
│         ▼                                   ▼                │
│  ┌───────────────────────────────────────────────────┐       │
│  │  draw_landmarks(frame, landmarks, w, h)           │       │
│  │  draw_info(frame, result, fps)                    │       │
│  └──────────────────────┬────────────────────────────┘       │
│                         │                                    │
│                         ▼                                    │
│  ┌───────────────────────────────────────────────────┐       │
│  │  cv2.imshow("AI Gesture Desktop Assistant",       │       │
│  │              frame_with_overlays)                  │       │
│  └──────────────────────┬────────────────────────────┘       │
│                         │                                    │
│                         ▼                                    │
│                   Display Window                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Interaction

### Frame Processing Pipeline (per iteration)

```
1. cap.read()                     →  BGR frame (numpy array)
2. cv2.cvtColor(frame, BGR2RGB)  →  RGB frame (numpy array)
3. mp.Image(format=SRGB, data=rgb) → MediaPipe image object
4. landmarker.detect(mp_image)    →  HandLandmarkerResult
5. for each hand in result:
     draw_landmarks(frame, landmarks, w, h)
6. draw_info(frame, result, fps)
7. cv2.imshow(title, frame)
8. if cv2.waitKey(1) == 'q': break
```

### Key Data Types

| Type | Source | Description |
|------|--------|-------------|
| `cv2.Mat` (numpy array) | `cap.read()` | BGR image, shape `(H, W, 3)`, dtype `uint8` |
| `mp.Image` | `mp.Image(...)` | MediaPipe image wrapper, expected format SRGB |
| `HandLandmarkerResult` | `landmarker.detect()` | Contains `.hand_landmarks`, `.handedness`, `.world_landmarks` |
| `NormalizedLandmark` | `result.hand_landmarks[i][j]` | Single landmark with `.x`, `.y`, `.z` (all normalized 0.0–1.0) |
| `Classifications` | `result.handedness[i][0]` | Contains `.category_name` (str) and `.score` (float) |

---

## Colour Space Strategy

```
┌─────────────┐     cv2.cvtColor      ┌─────────────┐
│  BGR Frame  │──────────────────────►│  RGB Frame   │
│  (original) │                       │  (copy)      │
│             │◄────── drawing ──────│              │
│             │     (cv2.circle,      │              │
│             │      cv2.line)        │              │
│             │                       │              │
│  ┌──────────┤                       │  ┌───────────┤
│  │ Display  │                       │  │ Detection  │
│  │ (imshow) │                       │  │ (detect)   │
│  └──────────┘                       │  └───────────┘
└─────────────┘                       └─────────────┘
```

**Why two paths?** OpenCV works in BGR; MediaPipe expects RGB. Converting the entire frame back and forth every iteration is wasteful. Instead, we:
1. Keep the original BGR frame for drawing and display.
2. Create a separate RGB copy only for detection (wrapped in `mp.Image`).

This means:
- `draw_landmarks()` and `draw_info()` operate on the BGR frame.
- `landmarker.detect()` receives the RGB `mp.Image`.
- `cv2.imshow()` receives the BGR frame with overlays.

---

## Error Handling Strategy

### Camera Errors

| Condition | Handling |
|-----------|----------|
| Camera not found / busy | `cap.isOpened()` returns `False`; print error and `return` |
| Frame read failure | `cap.read()` returns `(False, None)`; break loop |
| Camera disconnected mid-loop | Same as above — loop breaks cleanly |

### Model Errors

| Condition | Handling |
|-----------|----------|
| Model file not found | `HandLandmarker.create_from_options()` raises exception |
| Corrupted model file | Same — exception propagates to crash (fail-fast) |

### Detection Errors

| Condition | Handling |
|-----------|----------|
| No hands detected | `result.hand_landmarks` is empty list; loop continues without drawing |
| Partial detection | Hand detected but landmarks incomplete (theoretical, not observed) |

---

## Future Architecture (Milestone 3+)

When the project is modularized, the architecture will look like:

```
hand/
│
├── main.py                          # Entry point
├── core/
│   ├── __init__.py
│   ├── camera.py                    # Webcam wrapper class
│   ├── detector.py                  # HandLandmarker wrapper
│   └── renderer.py                  # Drawing functions
├── controllers/
│   ├── __init__.py
│   └── cursor_controller.py         # Cursor control logic
├── utils/
│   ├── __init__.py
│   ├── geometry.py                  # Coordinate mapping, smoothing
│   └── constants.py                 # Hand connections, colours
├── models/
│   └── hand_landmarker.task
└── docs/
    └── ...
```

This modular structure follows the Single Responsibility Principle:
- **camera.py** — owns the `cv2.VideoCapture` lifecycle
- **detector.py** — owns the `HandLandmarker` loading and detection
- **renderer.py** — owns all drawing operations
- **cursor_controller.py** — owns mouse control (future)
- **geometry.py** — owns coordinate math and filtering (future)