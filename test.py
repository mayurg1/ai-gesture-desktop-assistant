import cv2
import mediapipe as mp
import time

# ---------------------------------------------------------------------------
# MediaPipe Tasks API imports (NOT mp.solutions)
# ---------------------------------------------------------------------------
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

MODEL = "models/hand_landmarker.task"

# ---------------------------------------------------------------------------
# Official MediaPipe hand skeleton connections (21 landmarks, 20 connections)
# Copied from the open-source MediaPipe codebase – no mp.solutions dependency.
# ---------------------------------------------------------------------------
HAND_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),      # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),    # Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),   # Pinky
    (0, 17),                                    # Palm base
])

# Landmark indices for each fingertip
INDEX_FINGER_TIP = 8

# ---------------------------------------------------------------------------
# Active control region
# ---------------------------------------------------------------------------
CONTROL_REGION_RATIO = 0.70          # fraction of the frame the region occupies
COLOR_CONTROL_REGION = (0, 255, 0)   # green border
CONTROL_REGION_BORDER_THICKNESS = 2
COLOR_TIP_INSIDE = (0, 255, 0)       # green when fingertip is inside region
COLOR_TIP_OUTSIDE = (0, 0, 255)      # red when fingertip is outside region

# ---------------------------------------------------------------------------
# Colours (BGR)
# ---------------------------------------------------------------------------
COLOR_SKELETON = (180, 180, 180)    # light grey
COLOR_LANDMARK = (0, 255, 0)        # green
COLOR_TIP = (0, 0, 255)        # red (BGR)
COLOR_TEXT = (255, 255, 255)    # white

# ---------------------------------------------------------------------------
# Helper: point-in-rectangle test
# ---------------------------------------------------------------------------


def is_point_inside_region(x: int, y: int, region: dict) -> bool:
    """Return True if (x, y) falls within the active control region."""
    return (region["left"] <= x <= region["right"]
            and region["top"] <= y <= region["bottom"])


# ---------------------------------------------------------------------------
# Helper: draw all landmarks for a single hand
# ---------------------------------------------------------------------------


def draw_landmarks(
    frame: cv2.Mat,
    landmarks: list,
    w: int,
    h: int,
    region: dict | None = None,
) -> None:
    """Draw the 21 landmarks + skeleton for one hand on the frame.

    Parameters
    ----------
    region : dict or None
        If provided, the index fingertip colour reflects whether it is
        inside (green) or outside (red) the active control region.
    """
    pts = []
    for i, lm in enumerate(landmarks):
        cx, cy = int(lm.x * w), int(lm.y * h)
        pts.append((cx, cy))

        if i == INDEX_FINGER_TIP:
            # Choose colour based on region membership
            if region is not None and is_point_inside_region(cx, cy, region):
                tip_color = COLOR_TIP_INSIDE
            else:
                tip_color = COLOR_TIP_OUTSIDE
            cv2.circle(frame, (cx, cy), 10, tip_color, cv2.FILLED)
            cv2.circle(frame, (cx, cy), 10, (255, 255, 255), 2)
        else:
            cv2.circle(frame, (cx, cy), 5, COLOR_LANDMARK, cv2.FILLED)

    # Draw skeleton connections
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], COLOR_SKELETON, 2)

# ---------------------------------------------------------------------------
# Helper: compute active control region boundaries
# ---------------------------------------------------------------------------


def compute_control_region(frame_width: int, frame_height: int) -> dict:
    """Return a dict of control-region boundaries calculated dynamically.

    The region is a centred rectangle that occupies CONTROL_REGION_RATIO
    of the frame area, leaving equal margins on all four sides.

    Returns
    -------
    dict
        Keys: left, right, top, bottom, width, height (all ints).
    """
    region_width = int(frame_width * CONTROL_REGION_RATIO)
    region_height = int(frame_height * CONTROL_REGION_RATIO)

    left = (frame_width - region_width) // 2
    top = (frame_height - region_height) // 2

    return {
        "left": left,
        "right": left + region_width,
        "top": top,
        "bottom": top + region_height,
        "width": region_width,
        "height": region_height,
    }


# ---------------------------------------------------------------------------
# Helper: draw the active control region rectangle
# ---------------------------------------------------------------------------


def draw_control_region(frame: cv2.Mat, region: dict) -> None:
    """Draw a green rectangle outlining the active control region."""
    cv2.rectangle(
        frame,
        (region["left"], region["top"]),
        (region["right"], region["bottom"]),
        COLOR_CONTROL_REGION,
        CONTROL_REGION_BORDER_THICKNESS,
    )


# ---------------------------------------------------------------------------
# Helper: draw info overlay (handedness, confidence, FPS)
# ---------------------------------------------------------------------------


def draw_info(
    frame: cv2.Mat,
    result,
    fps: float,
    region: dict | None = None,
) -> None:
    """Overlay handedness, confidence, FPS, and region status.

    Parameters
    ----------
    region : dict or None
        If provided, an INSIDE / OUTSIDE label is shown for each hand.
    """
    y_offset = 30
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
    y_offset += 30

    num_hands = len(result.hand_landmarks)
    cv2.putText(frame, f"Hands: {num_hands}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
    y_offset += 30

    for i, hand_landmarks in enumerate(result.hand_landmarks):
        handedness = result.handedness[i][0].category_name
        confidence = result.handedness[i][0].score

        label = f"Hand {i+1}: {handedness} ({confidence:.2f})"
        cv2.putText(frame, label, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)
        y_offset += 25

        # Index-finger-tip (landmark 8) screen coords
        tip = hand_landmarks[INDEX_FINGER_TIP]
        h, w, _ = frame.shape
        tip_x, tip_y = int(tip.x * w), int(tip.y * h)
        coord = f"  Tip#8: ({tip_x}, {tip_y})"
        cv2.putText(frame, coord, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)
        y_offset += 25

        # Region status
        if region is not None:
            inside = is_point_inside_region(tip_x, tip_y, region)
            status_text = "  INSIDE" if inside else "  OUTSIDE"
            status_color = COLOR_TIP_INSIDE if inside else COLOR_TIP_OUTSIDE
            cv2.putText(frame, status_text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
            y_offset += 25

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL),
        running_mode=VisionRunningMode.IMAGE,
    )

    landmarker = HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # FPS calculation
        now = time.time()
        fps = 1.0 / (now - prev_time)
        prev_time = now

        h, w, _ = frame.shape

        # Compute active control region (adapts to any resolution)
        region = compute_control_region(w, h)

        # Convert BGR -> RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Detect
        result = landmarker.detect(mp_image)

        # Draw each detected hand (pass region for tip colouring)
        for hand_landmarks in result.hand_landmarks:
            draw_landmarks(frame, hand_landmarks, w, h, region)

        # Overlay info (pass region for status text)
        draw_info(frame, result, fps, region)

        # Draw active control region (on top of everything)
        draw_control_region(frame, region)

        cv2.imshow("AI Gesture Desktop Assistant", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
