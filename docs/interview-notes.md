# Interview Notes — AI Gesture Desktop Assistant

---

## Overview

This document contains interview-ready explanations for every feature in the project. Use these as study material for technical interviews. Each section follows the format:

- **What problem does it solve?**
- **How does it work?**
- **Design decisions and tradeoffs**
- **Expected interview questions with model answers**

---

## Feature 1: Hand Detection (Milestone 1)

### What Problem Does It Solve?

Hand detection is the foundation of the entire system. Before we can control a cursor, recognize gestures, or perform any action, we need to detect where the hands are and identify their keypoints (landmarks). Without this, the rest of the system cannot exist.

### How Does It Work?

We use MediaPipe's `HandLandmarker` class from the Tasks API (`mp.tasks.vision`). The pipeline is:

1. Capture a frame from the webcam using OpenCV.
2. Convert the frame from BGR (OpenCV's native format) to RGB (MediaPipe's expected format).
3. Wrap the RGB frame in an `mp.Image` object.
4. Call `landmarker.detect(mp_image)` — this runs the pre-trained TensorFlow Lite model on the CPU.
5. The result contains: 21 normalized 3D landmarks per hand, handedness classification, and world landmarks.

### Design Decisions

**Why MediaPipe Tasks API instead of `mp.solutions`?**
The `mp.solutions` module is deprecated. MediaPipe has migrated to the Tasks API (`mp.tasks.vision`) which provides a more consistent, typed, and future-proof interface. The Tasks API also supports running modes (IMAGE, VIDEO, LIVE_STREAM) that give finer control over inference timing.

**Why the pre-trained model instead of training our own?**
MediaPipe's hand landmarker model is trained on millions of images across diverse skin tones, lighting conditions, and hand poses. Training a comparable model from scratch would require thousands of labelled images, significant GPU time, and computer vision expertise. There is no benefit to re-training for this use case.

### Tradeoffs

| Approach | Pros | Cons |
|----------|------|------|
| Pre-trained MediaPipe model | Ready to use, high accuracy, diverse training data | Cannot customize for specific use cases |
| Custom trained model | Full control over landmarks, domain-specific | Requires thousands of labelled images, GPU training |
| Classical CV (e.g., skin colour segmentation) | No ML dependencies | Fragile, low accuracy, fails in varied lighting |

### Interview Questions

**Q: Why did you choose MediaPipe over OpenCV's built-in hand detection?**

A: "OpenCV does not have a built-in hand detector. Classical computer vision approaches like skin colour segmentation or contour detection are highly sensitive to lighting, background, and skin tone. MediaPipe uses a deep learning model trained on diverse data, making it robust in real-world conditions. It also provides 21 consistent landmarks per hand, which is essential for gesture recognition."

**Q: How does the HandLandmarker model work internally?**

A: "MediaPipe uses a two-stage pipeline: First, a palm detection model (BlazePalm) localizes the hand region in the image using a lightweight single-shot detector. It outputs a bounding box around the palm. Second, a hand landmark model (also based on BlazePalm architecture) crops the hand region and predicts 21 3D landmarks. This two-stage approach is more efficient than running a full-body pose estimator because it only processes the hand region."

---

## Feature 2: Hand Visualization (Milestone 2)

### What Problem Does It Solve?

Raw landmark coordinates (21 × 3 floats per hand) are meaningless to a human. Visualization renders them on the video frame so the user can see:
- Whether the detection is working correctly.
- Where the landmarks are located relative to the hand.
- Real-time performance metrics (FPS).
- The index finger tip position for debugging cursor control.

### How Does It Work?

Three visual layers are drawn on each frame:

1. **Landmarks:** 21 filled circles at the pixel coordinates of each landmark. Normal landmarks are green (radius 5). The index finger tip (landmark 8) is red with a white border (radius 10).

2. **Skeleton:** 20 grey lines connecting the landmarks according to the official MediaPipe hand topology. The connections define the thumb chain, finger chains, and palm base.

3. **Info overlay:** FPS, hand count, handedness labels with confidence scores, and the pixel coordinates of the index fingertip.

### Key Algorithm: Coordinate Mapping

MediaPipe returns normalized coordinates (0.0 to 1.0). To draw on the frame:

```
pixel_x = landmark_normalized_x × frame_width
pixel_y = landmark_normalized_y × frame_height
```

### Design Decisions

**Why hardcode HAND_CONNECTIONS instead of importing from mp.solutions?**
The `mp.solutions.hands.HAND_CONNECTIONS` constant is in the deprecated module. Hardcoding 20 connection pairs is trivial, self-documenting, and eliminates a dependency on deprecated code.

**Why separate drawing functions?**
`draw_landmarks()` and `draw_info()` handle independent responsibilities. Separating them follows the Single Responsibility Principle. If we want to change the overlay style (e.g., different font, additional metrics), we modify only `draw_info()` without touching landmark rendering.

**Why draw on the BGR frame instead of RGB?**
OpenCV's `imshow()` expects BGR. MediaPipe's `detect()` expects RGB. By keeping the original BGR frame for drawing and creating a separate RGB copy only for detection, we avoid unnecessary colour space conversions.

### Tradeoffs

| Decision | Alternative | Tradeoff |
|----------|------------|----------|
| Hardcoded HAND_CONNECTIONS | Import from mp.solutions | More code now, but future-proof against API deprecation |
| Separate drawing functions | One monolithic drawing function | Slightly more function calls, but cleaner separation |
| Draw on BGR frame | Draw on RGB frame, convert back | Extra memory for RGB copy, but simpler pipeline |

### Interview Questions

**Q: How do you convert MediaPipe's landmark coordinates to pixel positions?**

A: "MediaPipe returns normalized coordinates where (0,0) is the top-left of the input image and (1,1) is the bottom-right. To convert to pixel coordinates, I multiply by the frame width and height and cast to integers. This is a linear mapping — it assumes the image is not cropped or scaled differently between detection and display."

**Q: Why did you highlight the index finger tip specifically?**

A: "The index finger tip (landmark 8) is the primary control point for cursor movement in the next milestone. By highlighting it visually, I can debug the coordinate mapping and verify that it's being tracked correctly before implementing cursor control. It also gives the user a clear visual anchor point."

**Q: How do you ensure the OpenCV window stays responsive?**

A: "OpenCV requires periodic event processing to keep the window responsive. I call `cv2.waitKey(1)` at the end of every frame iteration, which gives the window manager 1 millisecond to process events before continuing the loop. This is standard practice for real-time OpenCV applications."

---

## Feature 3: FPS Counter

### What Problem Does It Solve?

Real-time performance monitoring. The user needs to know if the system is running fast enough for responsive gesture control. FPS is the most直观 (intuitive) metric for this.

### How Does It Work?

Before the main loop, record the current time: `prev_time = time.time()`. At the start of each frame iteration, compute:

```
fps = 1.0 / (current_time - prev_time)
```

This is the instantaneous FPS — the reciprocal of the time between consecutive frames.

### Design Decisions

**Why instantaneous FPS instead of a sliding window average?**

Instantaneous FPS responds immediately to performance changes. A sliding window average is smoother but lags behind real-time performance changes. For visualization and debugging, instantaneous FPS is sufficient. For cursor control, a smoothed version would be better to avoid control jitter.

### Interview Questions

**Q: How accurate is your FPS counter?**

A: "The instantaneous FPS counter is accurate for steady-state performance but noisy — a single slow frame causes a sharp temporary drop. For production use, I'd implement a sliding window average over the last 30 frames, which gives a stable reading. The current implementation is adequate for debugging."

**Q: What factors affect FPS in your system?**

A: "Three main factors: camera resolution (higher resolution = more pixels to process), CPU load (MediaPipe runs on CPU via TensorFlow Lite), and the number of hands detected (each additional hand adds ~1ms of drawing overhead). The model inference is the bottleneck, typically taking 10-30ms per frame."

---

## Feature 4: Main Function Refactor

### What Problem Does It Solve?

Script-level code executes on import, which prevents unit testing and makes the module unusable as a library. Wrapping in `main()` follows Python best practices.

### How Does It Work?

```python
def main() -> None:
    # ... all application logic ...

if __name__ == "__main__":
    main()
```

The `if __name__ == "__main__"` guard ensures `main()` only runs when the script is executed directly, not when imported as a module.

### Interview Questions

**Q: Why is it important to use `if __name__ == "__main__"`?**

A: "Two reasons. First, it prevents the code from running when the module is imported — for example, when writing unit tests or importing functions from this file in another module. Second, it's a Python convention that signals 'this file is meant to be run as a script.' Without it, the file cannot be safely imported."

---

## System Design Questions

### Q: "Design a gesture-controlled desktop assistant from scratch."

This is a common system design interview question. Here's how to structure the answer:

**1. Requirements:**
- Real-time hand tracking from webcam
- Cursor control via hand movement
- Gesture-based click, scroll, and actions
- Low latency (< 50ms)
- Works in varied lighting conditions

**2. High-level Architecture:**
```
Webcam → Frame Capture → Hand Detection → Landmark Extraction
                                                   ↓
                              ┌────────────────────┼────────────────────┐
                              ▼                    ▼                    ▼
                         Cursor Control      Gesture Recognition   Click/Scroll
                              │                    │                    │
                              └────────────────────┼────────────────────┘
                                                   ▼
                                             OS Input API
```

**3. Component Breakdown:**
- **Camera Module:** Wraps OpenCV's `VideoCapture`. Handles camera selection, resolution configuration, and frame rate control.
- **Detection Module:** Wraps MediaPipe's `HandLandmarker`. Handles model loading, inference, and result parsing.
- **Renderer Module:** Draws landmarks, skeleton, and overlays on the frame for debugging.
- **Cursor Controller:** Maps hand positions to screen coordinates, applies smoothing, controls mouse via OS API.
- **Gesture Engine:** Classifies hand configurations (open palm, fist, pinch, etc.) using landmark geometry.
- **Action Mapper:** Maps recognized gestures to OS actions (click, scroll, volume, etc.).

**4. Key Design Decisions:**
- Use MediaPipe Tasks API (not deprecated `mp.solutions`)
- Pre-trained model (not custom training)
- Exponential Moving Average for cursor smoothing
- Rule-based gesture recognition (faster and simpler than ML for this use case)

**5. Tradeoffs:**
- CPU-based inference vs. GPU: CPU is more compatible but slower
- Rule-based gestures vs. ML: Rule-based is faster to implement but less flexible
- Single hand vs. multi-hand: Multi-hand enables more gestures but complicates the control logic

**6. Scalability and Extensibility:**
- Modular architecture (`core/`, `controllers/`, `utils/`) makes it easy to add new gestures or control modes
- Configuration files allow non-developers to tune thresholds
- The pipeline design makes it easy to swap components (e.g., use a different hand detector)

---

### Q: "How would you make this production-ready?"

**Answer structure:**

1. **Error Handling:** Add graceful degradation for camera disconnection, model loading failure, and low FPS.
2. **Logging:** Replace `print()` with structured logging (rotation, levels, file output).
3. **Configuration:** Externalize all thresholds, colours, and settings to a YAML/JSON config file.
4. **Testing:** Add unit tests for coordinate mapping, smoothing, and gesture logic. Add integration tests for the full pipeline (using recorded video).
5. **Packaging:** Create a `setup.py` or `pyproject.toml` for pip installation.
6. **Performance:** Add GPU delegate support for MediaPipe, optimize camera resolution, implement frame skipping for slow systems.
7. **Accessibility:** Add visual/auditory feedback for gestures, configurable sensitivity.
8. **Documentation:** Complete the `docs/` folder with architecture, algorithms, and user guide.

---

### Q: "How do you handle the latency vs. accuracy tradeoff in real-time hand tracking?"

**Answer:**

"Latency and accuracy trade off in three areas:

1. **Camera Resolution:** Higher resolution gives more accurate landmark positions (more pixels per finger) but increases inference time. I'd use 640×480 as a baseline and only increase to 1280×720 if needed for accuracy.

2. **Model Selection:** The MediaPipe hand landmarker comes in different versions (full, lite). The lite model is faster but slightly less accurate. I'd start with the full model and benchmark; if FPS is below 20, switch to lite.

3. **Smoothing:** Aggressive smoothing (low alpha) gives smooth cursor movement but introduces lag. Conservative smoothing (high alpha) is responsive but jittery. The optimal alpha depends on the user's tolerance for jitter vs. lag. I'd make it configurable with a default of 0.5 and let the user adjust.

The key insight is that accuracy is less important for smooth cursor control than consistent, low-latency tracking. A slightly inaccurate but responsive system feels better than a perfectly accurate but laggy one."

---

## Coding Interview Questions

### Q: "Implement the normalized-to-pixel coordinate mapping function."

```python
def map_landmark_to_pixel(
    landmarks: list,
    frame_width: int,
    frame_height: int
) -> list[tuple[int, int]]:
    """
    Convert normalized MediaPipe landmarks to pixel coordinates.
    
    Args:
        landmarks: List of 21 NormalizedLandmark objects.
        frame_width: Width of the video frame in pixels.
        frame_height: Height of the video frame in pixels.
    
    Returns:
        List of 21 (x, y) tuples in pixel coordinates.
    """
    pixel_coords = []
    for lm in landmarks:
        px = int(lm.x * frame_width)
        py = int(lm.y * frame_height)
        pixel_coords.append((px, py))
    return pixel_coords
```

**Complexity:** O(21) = O(1) per hand.

### Q: "Implement a simple FPS counter class."

```python
import time

class FPSCounter:
    """Simple FPS counter using a sliding window."""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.timestamps = []
    
    def update(self) -> float:
        """Record a frame and return the current FPS."""
        self.timestamps.append(time.time())
        
        # Keep only the last N timestamps
        while len(self.timestamps) > self.window_size:
            self.timestamps.pop(0)
        
        if len(self.timestamps) < 2:
            return 0.0
        
        elapsed = self.timestamps[-1] - self.timestamps[0]
        return (len(self.timestamps) - 1) / elapsed if elapsed > 0 else 0.0
```

**Note:** This implementation uses a sliding window of N frames, which gives a smoother reading than instantaneous FPS.

---

## Common Interview Mistakes to Avoid

1. **Using deprecated APIs:** Saying "I used `mp.solutions`" is a red flag. Always specify "MediaPipe Tasks API."
2. **Not mentioning the two-stage pipeline:** The palm detection + landmark refinement pipeline is a key MediaPipe architecture point. Mention it to show depth.
3. **Ignoring colour spaces:** Failing to explain BGR vs. RGB conversion shows a lack of understanding of the OpenCV/MediaPipe integration.
4. **No error handling:** Saying "it just works" without mentioning camera failure, model loading errors, or edge cases.
5. **Not understanding tradeoffs:** Every design decision has tradeoffs. Showing that you understand them (e.g., "I chose X over Y because of Z tradeoff") demonstrates senior-level thinking.

---

## Key Terminology for Interviews

| Term | Definition |
|------|------------|
| **Normalized coordinates** | Values between 0.0 and 1.0 relative to image dimensions |
| **Pixel coordinates** | Absolute integer positions on the image grid |
| **HandLandmarkerResult** | Typed result from MediaPipe containing landmarks, handedness, and world landmarks |
| **XNNPACK delegate** | TensorFlow Lite CPU acceleration for ARM/x86 processors |
| **BlazePalm** | MediaPipe's palm detection model (first stage) |
| **21 Landmarks** | The 21 keypoints detected per hand: wrist, knuckles, fingertips |
| **Exponential Moving Average (EMA)** | Smoothing technique: `smooth = α * raw + (1-α) * smooth_prev` |
| **Frozenset** | Python immutable set, used for HAND_CONNECTIONS to prevent modification |