# Project Roadmap — AI Gesture Desktop Assistant

---

## Milestone Status

| Milestone | Status | Description |
|-----------|--------|-------------|
| **Milestone 1** | ✅ Complete | Webcam + Hand Detection |
| **Milestone 2** | ✅ Complete | Hand Visualization |
| **Milestone 3** | ⬜ Next | Cursor Control |
| **Milestone 4** | ⬜ Planned | Gesture Recognition |
| **Milestone 5** | ⬜ Planned | Click & Scroll |
| **Milestone 6** | ⬜ Planned | Desktop Automation |
| **Milestone 7** | ⬜ Planned | Production Polish |

---

## Milestone 1: Webcam + Hand Detection ✅

**Objective:** Verify that the webcam works and that MediaPipe can detect hands from the video feed.

**Deliverables:**
- [x] `camera_test.py` — standalone webcam capture test
- [x] `test.py` — basic MediaPipe HandLandmarker integration
- [x] Model file `models/hand_landmarker.task` downloaded and working
- [x] Console output showing detected hand count

**Key technical achievement:** First successful end-to-end pipeline — camera → frame → detection → console output.

**Files created:**
- `camera_test.py` — new file
- `test.py` — new file (initial version, script-level code)
- `models/hand_landmarker.task` — downloaded model (not tracked in git)

---

## Milestone 2: Hand Visualization ✅

**Objective:** Render detected hands on the video frame with landmarks, skeleton, and information overlay.

**Deliverables:**
- [x] Draw all 21 landmarks as green circles
- [x] Draw the 20-connection hand skeleton as grey lines
- [x] Highlight index finger tip (landmark 8) with red circle + white border
- [x] Display FPS counter
- [x] Display hand count
- [x] Display handedness (Left/Right) with confidence
- [x] Display index finger tip screen coordinates
- [x] Refactor code into `main()` function with `__name__` guard
- [x] Eliminate `mp.solutions` dependency

**Key technical achievement:** Real-time visual feedback — the user can see exactly what the AI sees.

**Files modified:**
- `test.py` — major update (152 lines, from 44)

**Files created:**
- `docs/engineering-journal.md` — Session 1
- `docs/architecture.md` — full architecture documentation
- `docs/algorithms.md` — algorithm catalog
- `docs/debugging.md` — bug database
- `docs/roadmap.md` — this file
- `docs/interview-notes.md` — interview preparation

---

## Milestone 3: Cursor Control ⬜

**Objective:** Use the index finger tip position (landmark 8) to control the mouse cursor on the desktop.

**Required sub-tasks:**
- [ ] Extract index finger tip coordinates from detected hands
- [ ] Implement coordinate mapping from camera frame to screen coordinates
- [ ] Add coordinate smoothing (Exponential Moving Average)
- [ ] Integrate with mouse control API (`pyautogui` or Windows `user32.dll`)
- [ ] Handle multi-monitor configurations
- [ ] Add visual cursor feedback in the camera window
- [ ] Toggle cursor control on/off with a key press
- [ ] Handle edge cases: hand exits frame, low confidence, multiple hands

**Technical challenges:**
- **Coordinate remapping:** The camera frame is typically 640×480, but the screen may be 1920×1080 or higher. Direct scaling without smoothing causes jerky, unusable cursor movement.
- **Latency:** The round-trip from hand movement → camera capture → inference → screen update must be < 50ms for natural-feeling control.
- **Noise filtering:** Hand tremor and detection jitter must be smoothed out without introducing noticeable lag.

**Expected files:**
- `core/cursor_controller.py` — new module for cursor control logic
- `core/geometry.py` — coordinate mapping and smoothing utilities
- Potential modifications to existing files

---

## Milestone 4: Gesture Recognition ⬜

**Objective:** Recognize static hand gestures (e.g., peace sign, thumbs up, open palm, fist).

**Required sub-tasks:**
- [ ] Define gesture classes and their landmark signatures
- [ ] Implement landmark angle calculations (finger bending angles)
- [ ] Implement landmark distance calculations (tip-to-tip distances)
- [ ] Build a gesture classifier (rule-based or ML-based)
- [ ] Evaluate gesture recognition accuracy

**Technical approach (rule-based):** Use the relative positions of fingertips to determine finger states (extended vs. curled). For example:
- Open palm: All 5 fingertips are above their respective PIP joints (landmarks 6, 10, 14, 18).
- Fist: All 5 fingertips are below their respective PIP joints.
- Peace sign: Only index and middle fingertips are extended.

**Technical approach (ML-based, alternative):** Collect labelled gesture data and train a small classifier (e.g., Random Forest or tiny neural network) on the 63 normalized landmark coordinates (21 × 3).

---

## Milestone 5: Click & Scroll ⬜

**Objective:** Implement mouse click (left, right) and scroll gestures.

**Required sub-tasks:**
- [ ] Define pinch gesture (thumb tip to index tip distance threshold)
- [ ] Implement pinch detection for left click
- [ ] Implement double-pinch for double click
- [ ] Implement three-finger gesture for right click
- [ ] Implement scroll gesture (two-finger vertical/horizontal swipe)
- [ ] Add click feedback (visual and/or audio)
- [ ] Tune gesture thresholds to avoid false positives

---

## Milestone 6: Desktop Automation ⬜

**Objective:** Perform desktop actions beyond cursor control — window management, application launching, system controls.

**Possible features:**
- [ ] Application launcher (draw a letter to open an app)
- [ ] Volume control (gesture-based)
- [ ] Media playback control (play/pause, next/previous)
- [ ] Virtual desktop switching
- [ ] Screenshot gesture
- [ ] Custom gesture mapping (user-defined)

---

## Milestone 7: Production Polish ⬜

**Objective:** Make the application production-ready.

**Required sub-tasks:**
- [ ] Create `main.py` as the single entry point
- [ ] Modularize into `core/`, `controllers/`, `utils/` packages
- [ ] Add configuration file (YAML or JSON) for gesture thresholds, colours, etc.
- [ ] Add command-line argument parsing
- [ ] Add logging (structured logging with rotation)
- [ ] Add unit tests (pytest)
- [ ] Add error recovery (camera reconnection, model reloading)
- [ ] Create installer script (pip installable)
- [ ] Add user documentation with screenshots
- [ ] Performance optimization (GPU delegate, model quantization)

---

## Future Improvements (Post-Milestone)

- [ ] **3D Hand Tracking** — Use `world_landmarks` for 3D gesture recognition.
- [ ] **Multiple Camera Support** — Allow user to select camera by index.
- [ ] **Background Mode** — Run without the display window (minimized to system tray).
- [ ] **Cross-Platform Support** — Test on Linux and macOS. Handle different camera APIs.
- [ ] **Custom Model Training** — Fine-tune the hand landmarker model for specific use cases.
- [ ] **Accessibility Features** — Configurable gesture difficulty, visual/auditory feedback.
- [ ] **Gesture Recording and Playback** — Record gestures for testing and demonstration.
- [ ] **Web Interface** — Stream the camera feed with overlays to a browser for remote use.

---

## Current Focus

**Milestone 3: Cursor Control** is the next milestone. The visualization layer built in Milestone 2 provides the foundation — the index finger tip position is already being extracted and displayed. The next session will:

1. Extract the tip coordinates programmatically (not just for display).
2. Implement coordinate smoothing.
3. Map camera coordinates to screen coordinates.
4. Move the mouse cursor using the smoothed, mapped position.