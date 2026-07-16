# Debugging Notes — AI Gesture Desktop Assistant

---

## Purpose

A living document recording every bug, error, and unexpected behaviour encountered during development. Each entry includes the problem, root cause, solution, and prevention strategy.

---

## Bug #1: ModuleNotFoundError — No module named 'mediapipe'

### Status

✅ Resolved

### Date

2026-07-16

### Problem

Running `python test.py` from the terminal failed with:

```
ModuleNotFoundError: No module named 'mediapipe'
```

### Symptoms

- The error occurred immediately on import, before any code executed.
- `pip list` showed mediapipe as installed.
- `import mediapipe` worked in the Anaconda prompt but not in the default terminal.

### Root Cause

The system has multiple Python installations:

| Python Location | Version | Has mediapipe? |
|----------------|---------|----------------|
| `C:\Users\mayur\AppData\Local\Programs\Python\Python313\python.exe` | 3.13 | ❌ No |
| `C:\Users\mayur\anaconda3\python.exe` | 3.13 (Anaconda base) | ❌ No |
| `C:\Users\mayur\anaconda3\envs\handtrack\python.exe` | 3.11 | ✅ Yes |

The default `python` command in PowerShell resolves to the first entry in the system PATH, which is Python 3.13 (without mediapipe). The `pip` command, however, resolves to the Anaconda pip (which installed mediapipe into the `handtrack` environment). This mismatch caused the confusion.

### Solution

Always use the full path to the correct interpreter:

```
C:\Users\mayur\anaconda3\envs\handtrack\python.exe test.py
```

### Prevention

1. **VS Code configuration:** Set the Python interpreter in VS Code to the `handtrack` environment:
   - Press `Ctrl+Shift+P` → "Python: Select Interpreter"
   - Choose `C:\Users\mayur\anaconda3\envs\handtrack\python.exe`
2. **PowerShell profile:** Create an alias or activate the environment:
   ```
   conda activate handtrack
   python test.py
   ```

### How to Recognize

If `import mediapipe` fails but `pip list | findstr mediapipe` shows it as installed, you are using the wrong Python interpreter.

---

## Bug #2: Pylint False-Positive Import Errors

### Status

✅ Resolved (by understanding — not a real bug)

### Date

2026-07-16

### Problem

VS Code shows red squiggles under `import cv2` and `import mediapipe` with the error:

```
Unable to import 'cv2'  pylint(import-error)
Unable to import 'mediapipe'  pylint(import-error)
```

### Symptoms

- The code runs successfully when executed.
- Only the VS Code editor shows errors.
- The errors appear on every import from non-standard-library packages.

### Root Cause

Pylint (the Python linter used by VS Code) is configured to use the default Python 3.13 interpreter. It checks for packages in that interpreter's `site-packages` directory. Since mediapipe and opencv-python are installed in the Anaconda environment, Pylint cannot find them.

### Solution

This is a VS Code workspace configuration issue. Two options:

**Option A:** Change the VS Code Python interpreter to the Anaconda environment (recommended):
1. `Ctrl+Shift+P` → "Python: Select Interpreter"
2. Choose `C:\Users\mayur\anaconda3\envs\handtrack\python.exe`

**Option B:** Ignore the linter errors (acceptable if you cannot change the interpreter):
- The errors are false positives.
- The code compiles and runs correctly.

### Prevention

Always configure the VS Code Python interpreter to match the runtime environment before starting development.

### How to Recognize

If Pylint shows import errors but the script runs without `ModuleNotFoundError`, it is a linter configuration issue, not a real bug.

---

## Bug #3: OpenCV Window Not Responding / Freezing

### Status

✅ Not encountered (prevented by design)

### Date

2026-07-16

### Problem (Anticipated)

The `cv2.imshow()` window appears but becomes unresponsive (spinning wheel / "Not Responding" in the title bar).

### Root Cause

OpenCV requires periodic `cv2.waitKey()` calls to process window events (paint, keyboard input, window close). If the main loop blocks for too long (e.g., due to slow inference) or if `waitKey()` is not called frequently enough, the OS marks the window as unresponsive.

### Solution (Already Implemented)

The existing code calls `cv2.waitKey(1)` at the end of every frame iteration. The `1` means "wait 1 millisecond for a key press, then continue." This is sufficient to keep the window responsive.

### Prevention

- Always call `cv2.waitKey(1)` (or higher) in every iteration of the video loop.
- Never use `cv2.waitKey(0)` in a video loop (it blocks indefinitely).
- If inference is slow (< 10 FPS), consider increasing `waitKey` to match the frame rate.

### How to Recognize

If the display window shows "Not Responding" in the title bar, the `waitKey()` call is missing or the loop is blocked.

---

## Bug #4: Hand Landmarks Drawn at Wrong Positions

### Status

✅ Not encountered (prevented by design)

### Date

2026-07-16

### Problem (Anticipated)

Landmarks appear at incorrect positions on the frame — e.g., mirrored, offset, or scaled incorrectly.

### Root Cause

Drawing on the wrong image (e.g., drawing on the RGB copy instead of the BGR original) or using incorrect coordinate mapping.

### Solution (Already Implemented)

- Always draw on the original BGR frame.
- Convert normalized coordinates to pixel coordinates using `int(lm.x * w)` and `int(lm.y * h)`.
- Verify that `w` and `h` match the frame dimensions used for detection.

### Prevention

- Keep the BGR frame for drawing and display.
- Create a separate RGB copy only for detection.
- Never modify the RGB copy after detection.

### How to Recognize

If landmarks appear at the wrong positions, check:
1. Are you drawing on the BGR frame or the RGB copy?
2. Are `w` and `h` the correct frame dimensions?
3. Is the frame flipped horizontally? (MediaPipe expects non-mirrored input.)

---

## Bug #5: No Hands Detected

### Status

✅ Not encountered (expected behaviour)

### Date

2026-07-16

### Problem (Anticipated)

The application runs but never detects any hands. `result.hand_landmarks` is always empty.

### Root Cause

Multiple possible causes:
1. Hand is not in the camera frame.
2. Lighting is too dark or too bright.
3. Hand is too far from the camera.
4. Hand is moving too fast (motion blur).
5. Model file is corrupted or wrong version.

### Solution

1. Ensure the hand is clearly visible in the center of the frame.
2. Ensure good, even lighting (no backlighting).
3. Keep the hand within 30-100 cm of the camera.
4. Move slowly.
5. Verify the model file: `models/hand_landmarker.task` should be ~15 MB.

### Prevention

- Add a visual indicator when no hands are detected (already done — "Hands: 0" is displayed).
- Add a confidence threshold check in future milestones.

### How to Recognize

The overlay shows "Hands: 0" and no landmarks are drawn. The FPS counter is still active.

---

## Bug #6: Low FPS (< 15)

### Status

✅ Not encountered (performance observation)

### Date

2026-07-16

### Problem (Anticipated)

The application runs but FPS is too low for real-time interaction.

### Root Cause

1. Camera resolution is too high (e.g., 1920x1080).
2. CPU is under heavy load from other processes.
3. MediaPipe is running on CPU without hardware acceleration.
4. Too many drawing operations per frame.

### Solution

1. Reduce camera resolution: `cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)` and `cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)`.
2. Close other CPU-intensive applications.
3. The XNNPACK delegate is automatically used by MediaPipe on compatible CPUs.
4. Drawing operations are negligible (< 2ms per frame).

### Prevention

- Monitor FPS as part of the overlay (already implemented).
- Add resolution configuration as a command-line argument in future milestones.

### How to Recognize

The FPS counter shows values below 15 consistently.

---

## Debugging Commands

### Check Python Interpreter

```powershell
# Windows
where python

# Show all available interpreters
Get-Command python* | Select-Object Source
```

### Check Package Installation

```powershell
# List installed packages (Windows)
pip list | findstr mediapipe
pip list | findstr opencv

# Show package details
pip show mediapipe
```

### Test MediaPipe Import

```powershell
# Quick import test
python -c "import mediapipe as mp; print(mp.__version__)"

# With full path
C:\Users\mayur\anaconda3\envs\handtrack\python.exe -c "import mediapipe as mp; print(mp.__version__)"
```

### Test Camera

```powershell
# Run the standalone camera test
python camera_test.py
```

### Check Model File

```powershell
# Verify model file exists and size
dir models\hand_landmarker.task
```

### Monitor FPS

The FPS is displayed in the top-left corner of the application window. For more detailed timing:

```python
# Add timing instrumentation (future)
import time
t0 = time.time()
# ... detection ...
t1 = time.time()
# ... drawing ...
t2 = time.time()
print(f"Detection: {(t1-t0)*1000:.1f}ms, Drawing: {(t2-t1)*1000:.1f}ms")
```

---

## Common Error Messages

| Error Message | Likely Cause | Solution |
|--------------|--------------|----------|
| `ModuleNotFoundError: No module named 'mediapipe'` | Wrong Python interpreter | Use the Anaconda environment interpreter |
| `FileNotFoundError: models/hand_landmarker.task` | Model file missing | Download the model or check the path |
| `cv2.error: ... is not a numpy array` | Frame read failed | Check camera connection |
| `AttributeError: 'HandLandmarkerResult' object has no attribute 'hand_landmarks'` | Wrong API version | Update mediapipe to 0.10.x |
| `TypeError: create_from_options() got an unexpected keyword argument` | API mismatch | Check MediaPipe Tasks API documentation |