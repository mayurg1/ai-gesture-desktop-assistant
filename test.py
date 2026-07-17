import mediapipe as mp
import os
import cv2
import time
import screeninfo
import pyautogui
import numpy as np

from config import (
    ENABLE_MOUSE_CONTROL,
    CURSOR_DEADZONE,
    MIRROR_CAMERA,
    MIRROR_CURSOR,
    PINCH_THRESHOLD,
    CLICK_COOLDOWN_MS,
    DRAG_HOLD_MS,
)

# Suppress verbose MediaPipe / TensorFlow Lite logging
os.environ["GLOG_minloglevel"] = "2"  # 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL


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
THUMB_TIP = 4
INDEX_FINGER_TIP = 8
MIDDLE_FINGER_TIP = 12

# ---------------------------------------------------------------------------
# Active control region
# ---------------------------------------------------------------------------
CONTROL_REGION_RATIO = 0.70          # fraction of the frame the region occupies
COLOR_CONTROL_REGION = (0, 255, 0)   # green border
CONTROL_REGION_BORDER_THICKNESS = 2
COLOR_TIP_INSIDE = (0, 255, 0)       # green when fingertip is inside region
COLOR_TIP_OUTSIDE = (0, 0, 255)      # red when fingertip is outside region

# ---------------------------------------------------------------------------
# Tracking validation
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.80          # minimum confidence for valid tracking

# ---------------------------------------------------------------------------
# Colours (BGR)
# ---------------------------------------------------------------------------
COLOR_SKELETON = (180, 180, 180)    # light grey
COLOR_LANDMARK = (0, 255, 0)        # green
COLOR_TIP = (0, 0, 255)        # red (BGR)
COLOR_TEXT = (255, 255, 255)    # white
COLOR_SUCCESS = (0, 255, 0)     # green checkmark
COLOR_FAILURE = (0, 0, 255)     # red cross
COLOR_VIRTUAL_CURSOR = (255, 128, 0)   # blue (BGR) — filled circle (smoothed)
# orange (BGR) — filled circle while dragging
COLOR_DRAGGING = (0, 128, 255)
COLOR_RAW_CURSOR = (0, 0, 255)         # red (BGR) — filled circle (raw)
SMOOTHED_CURSOR_RADIUS = 12
RAW_CURSOR_RADIUS = 6

# ---------------------------------------------------------------------------
# Display configuration
# ---------------------------------------------------------------------------
DISPLAY_WIDTH = 1280   # Target display width for preview window
DISPLAY_HEIGHT = 720   # Target display height for preview window

# ---------------------------------------------------------------------------
# Helper: resize frame while preserving aspect ratio
# ---------------------------------------------------------------------------


def resize_with_aspect_ratio(
    frame: cv2.Mat, target_width: int, target_height: int
) -> cv2.Mat:
    """Resize frame to fit within target dimensions while preserving aspect ratio.

    Adds black padding if necessary to avoid stretching.
    When MIRROR_CAMERA is True, flips the frame horizontally for mirror effect.

    Returns
    -------
    cv2.Mat
        Resized frame with preserved aspect ratio.
    """
    h, w = frame.shape[:2]

    # Calculate scaling factor to fit within target
    scale = min(target_width / w, target_height / h)

    # New dimensions maintaining aspect ratio
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize the frame
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Flip horizontally for mirror effect if MIRROR_CAMERA is enabled
    if MIRROR_CAMERA:
        resized = cv2.flip(resized, 1)

    # Create canvas with target dimensions and black background
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # Calculate padding to center the image
    pad_x = (target_width - new_w) // 2
    pad_y = (target_height - new_h) // 2

    # Place resized frame on canvas
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    return canvas


# ---------------------------------------------------------------------------
# Helper: point-in-rectangle test
# ---------------------------------------------------------------------------


def is_point_inside_region(x: int, y: int, region: dict) -> bool:
    """Return True if (x, y) falls within the active control region."""
    return (region["left"] <= x <= region["right"]
            and region["top"] <= y <= region["bottom"])


# ---------------------------------------------------------------------------
# Helper: validate mapping bounds
# ---------------------------------------------------------------------------


def is_mapping_valid(
    x: int, y: int, region: dict, screen_width: int, screen_height: int
) -> bool:
    """Return True if the interpolated point lands inside screen bounds.

    Checks the raw linear-interpolation result (before clamping) to
    determine whether the mapped cursor would naturally fall on-screen.
    """
    if screen_width <= 0 or screen_height <= 0:
        return False
    mapped_x = (x - region["left"]) / region["width"] * screen_width
    mapped_y = (y - region["top"]) / region["height"] * screen_height
    return 0 <= mapped_x <= screen_width and 0 <= mapped_y <= screen_height


# ---------------------------------------------------------------------------
# Helper: validate tracking state and return structured result
# ---------------------------------------------------------------------------


TrackingResult = dict  # type alias for readability


def validate_tracking(
    hand_detected: bool,
    inside_region: bool,
    mapping_valid: bool,
    confidence: float,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> TrackingResult:
    """Evaluate whether cursor control should be allowed.

    Returns a dict with keys:
        hand_detected     — bool
        inside_region     — bool
        mapping_valid     — bool
        tracking_confidence — bool  (confidence >= threshold)
        can_control_cursor — bool  (all conditions must be True)
    """
    tracking_confidence = confidence >= confidence_threshold
    can_control_cursor = (
        hand_detected
        and inside_region
        and mapping_valid
        and tracking_confidence
    )

    return {
        "hand_detected": hand_detected,
        "inside_region": inside_region,
        "mapping_valid": mapping_valid,
        "tracking_confidence": tracking_confidence,
        "can_control_cursor": can_control_cursor,
    }


# ---------------------------------------------------------------------------
# Helper: map camera coordinates to screen coordinates
# ---------------------------------------------------------------------------


def map_to_screen(
    x: int,
    y: int,
    region: dict,
    screen_width: int,
    screen_height: int,
) -> tuple[int, int]:
    """Map a point inside the active region to screen coordinates.

    Uses linear interpolation: the point's relative position within the
    region rectangle is projected onto the full screen dimensions.

    When MIRROR_CURSOR is True, the X coordinate is mirrored so that
    left movement in the mirrored preview maps to left movement on screen.

    Returns
    -------
    tuple[int, int]
        (mapped_x, mapped_y) clamped to [0, screen_width] × [0, screen_height].
    """
    # Calculate relative position within region
    rel_x = (x - region["left"]) / region["width"]
    rel_y = (y - region["top"]) / region["height"]

    # Mirror X coordinate if MIRROR_CURSOR is enabled
    if MIRROR_CURSOR:
        rel_x = 1.0 - rel_x

    mapped_x = rel_x * screen_width
    mapped_y = rel_y * screen_height

    # Clamp to valid screen bounds
    mapped_x = max(0, min(mapped_x, screen_width))
    mapped_y = max(0, min(mapped_y, screen_height))

    return int(mapped_x), int(mapped_y)


# ---------------------------------------------------------------------------
# Helper: map screen coordinates back to camera-preview coordinates
# ---------------------------------------------------------------------------


def map_to_preview(
    screen_x: int,
    screen_y: int,
    region: dict,
    screen_width: int,
    screen_height: int,
) -> tuple[int, int]:
    """Inverse of map_to_screen — convert screen coords to preview position.

    This is used to draw the virtual cursor at the correct location on the
    camera frame, *not* to move the OS cursor.
    When MIRROR_CAMERA is True, the X coordinate is mirrored to match
    the horizontally flipped display.
    """
    # Calculate relative position on screen
    rel_x = screen_x / screen_width
    rel_y = screen_y / screen_height

    # Mirror X coordinate if MIRROR_CAMERA is enabled (to match flipped display)
    if MIRROR_CAMERA:
        rel_x = 1.0 - rel_x

    preview_x = rel_x * region["width"] + region["left"]
    preview_y = rel_y * region["height"] + region["top"]
    return int(preview_x), int(preview_y)


# ---------------------------------------------------------------------------
# Cursor smoother (Exponential Moving Average)
# ---------------------------------------------------------------------------


class CursorSmoother:
    """EMA filter for cursor coordinates.

    Formula
    -------
        smoothed = previous + alpha * (current - previous)

    Parameters
    ----------
    alpha : float
        Smoothing factor (0.0–1.0).  Lower = smoother but more lag.
    """

    def __init__(self, alpha: float = 0.20) -> None:
        self.alpha = alpha
        self._sx: float | None = None
        self._sy: float | None = None

    def update(self, x: float, y: float) -> tuple[float, float]:
        """Push a new raw coordinate pair and return the smoothed result.

        On the first call (or after reset) the raw value is returned
        directly since there is no previous value to blend with.
        """
        if self._sx is None or self._sy is None:
            self._sx, self._sy = float(x), float(y)
        else:
            self._sx += self.alpha * (x - self._sx)
            self._sy += self.alpha * (y - self._sy)
        return self._sx, self._sy

    def reset(self) -> None:
        """Clear the stored previous position."""
        self._sx = self._sy = None


# ---------------------------------------------------------------------------
# Helper: draw the virtual cursor on the camera frame
# ---------------------------------------------------------------------------


def draw_virtual_cursor(
    frame: cv2.Mat,
    raw_screen_x: int,
    raw_screen_y: int,
    smooth_screen_x: int,
    smooth_screen_y: int,
    region: dict,
    screen_width: int,
    screen_height: int,
    can_control: bool,
    is_dragging: bool = False,
) -> None:
    """Draw raw (small red) and smoothed (large blue/orange) cursors.

    Both cursors are only drawn when *can_control* is True.  When disabled
    the cursors are hidden.  When *is_dragging* is True the smoothed cursor
    changes to orange to indicate drag state.
    """
    if not can_control:
        return

    # --- Raw cursor (small red) ---
    rpx, rpy = map_to_preview(raw_screen_x, raw_screen_y, region,
                              screen_width, screen_height)
    cv2.circle(frame, (rpx, rpy), RAW_CURSOR_RADIUS,
               COLOR_RAW_CURSOR, cv2.FILLED)

    # --- Smoothed cursor (large blue/orange with white border) ---
    cursor_color = COLOR_DRAGGING if is_dragging else COLOR_VIRTUAL_CURSOR
    spx, spy = map_to_preview(smooth_screen_x, smooth_screen_y, region,
                              screen_width, screen_height)
    cv2.circle(frame, (spx, spy), SMOOTHED_CURSOR_RADIUS,
               cursor_color, cv2.FILLED)
    cv2.circle(frame, (spx, spy), SMOOTHED_CURSOR_RADIUS,
               (255, 255, 255), 2)


# ---------------------------------------------------------------------------
# Real Windows cursor controller
# ---------------------------------------------------------------------------


class MouseController:
    """Controls the real Windows cursor via pyautogui.

    Only moves when the position changes by more than *threshold* pixels
    to avoid flooding the OS with unnecessary mouse events.
    """

    def __init__(self, threshold: int = CURSOR_DEADZONE) -> None:
        self._last_x: int | None = None
        self._last_y: int | None = None
        self._threshold = threshold
        self._enabled = False

    def enable(self) -> None:
        """Allow cursor movement."""
        self._enabled = True

    def disable(self) -> None:
        """Prevent cursor movement and clear stored position."""
        self._enabled = False
        self._last_x = self._last_y = None

    def move(self, x: int, y: int) -> None:
        """Move the Windows cursor to (x, y) if the change exceeds threshold."""
        if not self._enabled:
            return

        # Skip if distance from last position is below threshold
        if self._last_x is not None and self._last_y is not None:
            dx = x - self._last_x
            dy = y - self._last_y
            if dx * dx + dy * dy < self._threshold * self._threshold:
                return

        pyautogui.moveTo(x, y)
        self._last_x, self._last_y = x, y


# ---------------------------------------------------------------------------
# Gesture Recognizer for pinch detection
# ---------------------------------------------------------------------------


class GestureRecognizer:
    """Detects hand gestures and triggers mouse actions.

    Implements a state machine for pinch-to-click and pinch-to-drag.

    States
    ------
    IDLE       — no pinch detected
    CLICK      — quick pinch (< DRAG_HOLD_MS) → left click
    DRAGGING   — sustained pinch (>= DRAG_HOLD_MS) → mouseDown held
    """

    def __init__(self, pinch_threshold: int = PINCH_THRESHOLD,
                 cooldown_ms: int = CLICK_COOLDOWN_MS,
                 drag_hold_ms: int = DRAG_HOLD_MS) -> None:
        self._pinch_threshold = pinch_threshold
        self._cooldown_ms = cooldown_ms
        self._drag_hold_ms = drag_hold_ms
        self._pinch_active = False
        self._is_dragging = False
        self._pinch_start_time = 0.0
        self._last_click_time = 0.0

    def calculate_distance(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """Calculate Euclidean distance between two points.

        Returns
        -------
        float
            Distance in pixels.
        """
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    def detect_pinch(self, thumb_x: int, thumb_y: int,
                     middle_x: int, middle_y: int) -> bool:
        """Check if thumb and middle finger are pinched.

        Returns
        -------
        bool
            True if distance is below pinch threshold.
        """
        distance = self.calculate_distance(
            thumb_x, thumb_y, middle_x, middle_y)
        return distance < self._pinch_threshold

    def is_dragging(self) -> bool:
        """Return True if currently in drag state."""
        return self._is_dragging

    def update(self, thumb_x: int, thumb_y: int,
               middle_x: int, middle_y: int) -> str:
        """Update gesture state machine.

        State machine:
        - IDLE → Pinch starts → record start time
        - Pinch held < DRAG_HOLD_MS → CLICK (return "click")
        - Pinch held >= DRAG_HOLD_MS → DRAGGING (return "drag_start")
        - DRAGGING + pinch held → (return "drag_move")
        - Pinch released → (return "drag_end" if dragging, else "none")

        Returns
        -------
        str
            One of: "none", "click", "drag_start", "drag_move", "drag_end"
        """
        current_time = time.time() * 1000  # Convert to milliseconds

        # Check if pinch is detected
        is_pinch = self.detect_pinch(thumb_x, thumb_y, middle_x, middle_y)

        if is_pinch and not self._pinch_active:
            # Pinch just started — record time
            self._pinch_active = True
            self._pinch_start_time = current_time
            return "none"

        if is_pinch and self._pinch_active:
            # Pinch held — check if we should start drag
            elapsed = current_time - self._pinch_start_time
            if not self._is_dragging and elapsed >= self._drag_hold_ms:
                # Cooldown check
                if current_time - self._last_click_time >= self._cooldown_ms:
                    self._is_dragging = True
                    self._last_click_time = current_time
                    return "drag_start"
            elif self._is_dragging:
                return "drag_move"
            else:
                # Quick pinch — check if we should trigger click
                if elapsed < self._drag_hold_ms:
                    # Still within click window, do nothing yet
                    pass
            return "none"

        if not is_pinch:
            # Pinch released
            if self._is_dragging:
                self._is_dragging = False
                self._pinch_active = False
                return "drag_end"

            if self._pinch_active:
                # Pinch was active but not dragging — check if it was a click
                elapsed = current_time - self._pinch_start_time
                self._pinch_active = False
                if elapsed < self._drag_hold_ms:
                    if current_time - self._last_click_time >= self._cooldown_ms:
                        self._last_click_time = current_time
                        return "click"

            self._pinch_active = False
            return "none"

        return "none"

    def get_pinch_distance(self, thumb_x: int, thumb_y: int,
                           middle_x: int, middle_y: int) -> float:
        """Get current pinch distance for display purposes."""
        return self.calculate_distance(thumb_x, thumb_y, middle_x, middle_y)


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
# Helper: draw unified debug panel
# ---------------------------------------------------------------------------

# Panel layout constants
_PNL_X = 14
_PNL_LH = 30
_PNL_FS = 0.50
_PNL_THICK = 2
_PNL_THIN = 1
_PNL_SEP_COLOR = (80, 80, 80)      # dim grey for separator lines
_PNL_HEADING_COLOR = (200, 200, 200)  # light grey for section headings


def _txt(frame: cv2.Mat, text: str, y: int,
         color: tuple = COLOR_TEXT,
         thick: int = _PNL_THICK) -> None:
    """Single-stroke text helper — consistent x, font, scale."""
    cv2.putText(frame, text, (_PNL_X, y), cv2.FONT_HERSHEY_SIMPLEX,
                _PNL_FS, color, thick, cv2.LINE_AA)


def _sep(frame: cv2.Mat, y: int) -> None:
    """Draw a thin horizontal separator line."""
    cv2.line(frame, (_PNL_X, y), (_PNL_X + 220, y),
             _PNL_SEP_COLOR, 1, cv2.LINE_AA)


def _section(frame: cv2.Mat, title: str, y: int) -> int:
    """Draw a section heading and return the next y position."""
    _txt(frame, title, y, _PNL_HEADING_COLOR, _PNL_THIN)
    return y + _PNL_LH


def draw_debug_panel(
    frame: cv2.Mat,
    result,
    fps: float,
    region: dict | None = None,
    screen_width: int = 0,
    screen_height: int = 0,
    raw_mx: int = 0,
    raw_my: int = 0,
    smooth_mx: int = 0,
    smooth_my: int = 0,
    gesture_recognizer: GestureRecognizer | None = None,
) -> None:
    """Render a single, clean, professional debug panel on the left side.

    Sections are separated by thin horizontal lines for visual grouping.

    Parameters
    ----------
    region : dict or None
        If provided, region status, coordinate mapping, and validation
        are shown.  When None only performance metrics are displayed.
    screen_width, screen_height : int
        Physical screen resolution for mapped coordinate display.
    raw_mx, raw_my, smooth_mx, smooth_my : int
        Raw and smoothed screen coordinates for the cursor comparison display.
    """
    num_hands = len(result.hand_landmarks)
    y = _PNL_LH

    # ------------------------------------------------------------------
    # Section 1 — Performance
    # ------------------------------------------------------------------
    y = _section(frame, "Performance", y)
    _txt(frame, f"FPS           {fps:.1f}", y)
    y += _PNL_LH
    _txt(frame, f"Hands         {num_hands}", y)
    y += _PNL_LH + _PNL_LH // 2

    # Bail out early if there is no region (nothing more to show)
    if region is None:
        return

    # Gather data for the first hand (used in multiple sections)
    if num_hands > 0:
        h0 = result.hand_landmarks[0]
        tip = h0[INDEX_FINGER_TIP]
        fh, fw, _ = frame.shape
        tip_x, tip_y = int(tip.x * fw), int(tip.y * fh)
        inside = is_point_inside_region(tip_x, tip_y, region)
        raw_confidence = result.handedness[0][0].score
        handedness = result.handedness[0][0].category_name
        mapping_ok = (is_mapping_valid(tip_x, tip_y, region,
                                       screen_width, screen_height)
                      if inside else False)
        tracking = validate_tracking(
            hand_detected=True,
            inside_region=inside,
            mapping_valid=mapping_ok,
            confidence=raw_confidence,
        )
    else:
        tip_x = tip_y = 0
        inside = False
        raw_confidence = 0.0
        handedness = "—"
        tracking = validate_tracking(
            hand_detected=False, inside_region=False,
            mapping_valid=False, confidence=0.0,
        )

    # ------------------------------------------------------------------
    # Section 2 — Detection info
    # ------------------------------------------------------------------
    y = _section(frame, "Detection", y)

    _txt(frame, f"Hand          {handedness}", y)
    y += _PNL_LH
    _txt(frame, f"Confidence    {raw_confidence:.2f}", y)
    y += _PNL_LH

    status_color = (COLOR_TIP_INSIDE if inside else COLOR_TIP_OUTSIDE)
    _txt(frame, "Status        ", y)
    _txt(frame, "INSIDE" if inside else "OUTSIDE", y + 82, status_color)
    y += _PNL_LH + _PNL_LH // 2

    # ------------------------------------------------------------------
    # Section 3 — Coordinate mapping
    # ------------------------------------------------------------------
    y = _section(frame, "Coordinates", y)

    _txt(frame, "Camera", y, _PNL_HEADING_COLOR, _PNL_THIN)
    y += _PNL_LH
    _txt(frame, f"  X           {tip_x}", y, thick=_PNL_THIN)
    y += _PNL_LH
    _txt(frame, f"  Y           {tip_y}", y, thick=_PNL_THIN)
    y += _PNL_LH

    # Down arrow
    _txt(frame, "  \u2193", y, _PNL_SEP_COLOR, _PNL_THICK)
    y += _PNL_LH

    _txt(frame, "Screen", y, _PNL_HEADING_COLOR, _PNL_THIN)
    y += _PNL_LH

    if num_hands > 0 and inside and screen_width > 0 and screen_height > 0:
        _txt(frame, f"  X           {raw_mx}", y, thick=_PNL_THIN)
        y += _PNL_LH
        _txt(frame, f"  Y           {raw_my}", y, thick=_PNL_THIN)
        y += _PNL_LH
    else:
        _txt(frame, "  N/A", y, thick=_PNL_THIN)
        y += _PNL_LH
    y += _PNL_LH // 2

    # ------------------------------------------------------------------
    # Section 4 — Validation
    # ------------------------------------------------------------------
    y = _section(frame, "Validation", y)

    checks = [
        ("Hand Detected",  tracking["hand_detected"]),
        ("Inside Region",  tracking["inside_region"]),
        ("Mapping Valid",  tracking["mapping_valid"]),
        ("Confidence OK",  tracking["tracking_confidence"]),
        ("Cursor Ready",   tracking["can_control_cursor"]),
    ]
    for label, ok in checks:
        sym = "\u2713" if ok else "\u2717"
        clr = COLOR_SUCCESS if ok else COLOR_FAILURE
        _txt(frame, f"  {sym}  {label}", y, clr, _PNL_THICK)
        y += _PNL_LH
    y += _PNL_LH // 2

    # ------------------------------------------------------------------
    # Section 5 — Virtual Cursor
    # ------------------------------------------------------------------
    y = _section(frame, "Virtual Cursor", y)
    cursor_ok = tracking["can_control_cursor"]
    cursor_sym = "\u2713" if cursor_ok else "\u2717"
    cursor_color = COLOR_SUCCESS if cursor_ok else COLOR_FAILURE
    cursor_label = "READY" if cursor_ok else "DISABLED"
    _txt(frame, f"  {cursor_sym}  {cursor_label}", y, cursor_color)
    y += _PNL_LH

    # Raw vs Smoothed coordinates comparison
    if num_hands > 0 and inside:
        _txt(frame, "Raw", y, _PNL_HEADING_COLOR, _PNL_THIN)
        y += _PNL_LH
        _txt(frame, f"  X           {raw_mx}", y, thick=_PNL_THIN)
        y += _PNL_LH
        _txt(frame, f"  Y           {raw_my}", y, thick=_PNL_THIN)
        y += _PNL_LH

        _txt(frame, "Smoothed", y, _PNL_HEADING_COLOR, _PNL_THIN)
        y += _PNL_LH
        _txt(frame, f"  X           {smooth_mx}", y, thick=_PNL_THIN)
        y += _PNL_LH
        _txt(frame, f"  Y           {smooth_my}", y, thick=_PNL_THIN)
        y += _PNL_LH
    y += _PNL_LH // 2

    # ------------------------------------------------------------------
    # Section 6 — Mouse Control
    # ------------------------------------------------------------------
    y = _section(frame, "Mouse Control", y)
    mouse_enabled = ENABLE_MOUSE_CONTROL and tracking["can_control_cursor"]
    mouse_sym = "\u2713" if mouse_enabled else "\u2717"
    mouse_color = COLOR_SUCCESS if mouse_enabled else COLOR_FAILURE
    mouse_label = "MOVING" if mouse_enabled else "STOPPED"
    _txt(frame, f"  {mouse_sym}  {mouse_label}", y, mouse_color)
    y += _PNL_LH

    # Section 7 — Gesture
    y = _section(frame, "Gesture", y)
    if num_hands > 0:
        # Get thumb and middle finger tips for pinch detection
        thumb = h0[THUMB_TIP]
        middle = h0[MIDDLE_FINGER_TIP]
        thumb_x, thumb_y = int(thumb.x * fw), int(thumb.y * fh)
        middle_x, middle_y = int(middle.x * fw), int(middle.y * fh)
        pinch_distance = gesture_recognizer.get_pinch_distance(
            thumb_x, thumb_y, middle_x, middle_y)
        pinch_active = gesture_recognizer.detect_pinch(
            thumb_x, thumb_y, middle_x, middle_y)
        is_dragging = gesture_recognizer.is_dragging()
        gesture_sym = "\u2713" if pinch_active else "\u2717"
        gesture_color = COLOR_SUCCESS if pinch_active else COLOR_FAILURE
        if is_dragging:
            gesture_label = "DRAGGING"
            gesture_color = COLOR_DRAGGING
        elif pinch_active:
            gesture_label = "LEFT CLICK"
        else:
            gesture_label = "—"
        _txt(frame, f"  {gesture_sym}  {gesture_label}", y, gesture_color)
        y += _PNL_LH
        _txt(frame, f"  Thumb-Middle {pinch_distance:.0f}px", y,
             _PNL_HEADING_COLOR, _PNL_THIN)
    else:
        _txt(frame, "  —", y, _PNL_HEADING_COLOR, _PNL_THIN)


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

    # Get primary monitor dimensions (once at startup)
    try:
        monitor = screeninfo.get_monitors()[0]
        screen_width = monitor.width
        screen_height = monitor.height
    except Exception:
        screen_width = 1920
        screen_height = 1080
        print("⚠️  Could not detect screen resolution, using 1920×1080")

    # Enable pyautogui failsafe (move mouse to top-left corner to abort)
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.0  # no internal delay — we control timing

    # Initialise cursor smoother (alpha = 0.20)
    cursor_smoother = CursorSmoother(alpha=0.20)

    # Initialise mouse controller
    mouse_controller = MouseController()
    if ENABLE_MOUSE_CONTROL:
        mouse_controller.enable()

    # Initialise gesture recognizer
    gesture_recognizer = GestureRecognizer()

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

        # Compute cursor data (used by both virtual cursor and debug panel)
        raw_mx = raw_my = 0
        smooth_mx = smooth_my = 0
        can_control = False

        if len(result.hand_landmarks) > 0:
            h0 = result.hand_landmarks[0]
            tip = h0[INDEX_FINGER_TIP]
            tip_x, tip_y = int(tip.x * w), int(tip.y * h)
            inside = is_point_inside_region(tip_x, tip_y, region)
            confidence = result.handedness[0][0].score
            mapping_valid = (
                is_mapping_valid(tip_x, tip_y, region,
                                 screen_width, screen_height)
                if inside else False
            )
            tracking = validate_tracking(
                hand_detected=True,
                inside_region=inside,
                mapping_valid=mapping_valid,
                confidence=confidence,
            )
            can_control = tracking["can_control_cursor"]

            if can_control:
                raw_mx, raw_my = map_to_screen(tip_x, tip_y, region,
                                               screen_width, screen_height)
                # Update smoother — both raw and smoothed are floats
                smooth_mx, smooth_my = cursor_smoother.update(raw_mx, raw_my)

                # Ensure mouse controller is enabled when control is valid
                if ENABLE_MOUSE_CONTROL:
                    mouse_controller.enable()

                # Move real Windows cursor using smoothed coordinates
                mouse_controller.move(int(smooth_mx), int(smooth_my))

            # Gesture detection (always check when hand is detected)
            # Uses Thumb Tip (4) + Middle Finger Tip (12)
            thumb = h0[THUMB_TIP]
            middle = h0[MIDDLE_FINGER_TIP]
            thumb_x, thumb_y = int(thumb.x * w), int(thumb.y * h)
            middle_x, middle_y = int(middle.x * w), int(middle.y * h)
            gesture_action = gesture_recognizer.update(
                thumb_x, thumb_y, middle_x, middle_y)
            if gesture_action == "click":
                pyautogui.click()
            elif gesture_action == "drag_start":
                pyautogui.mouseDown()
            elif gesture_action == "drag_end":
                pyautogui.mouseUp()
        else:
            cursor_smoother.reset()
            mouse_controller.disable()

        # Overlay debug panel (pass raw and smoothed coords for display)
        draw_debug_panel(frame, result, fps, region,
                         screen_width, screen_height,
                         raw_mx, raw_my, int(smooth_mx), int(smooth_my),
                         gesture_recognizer)

        # Draw virtual cursor (raw + smoothed)
        draw_virtual_cursor(frame, raw_mx, raw_my,
                            int(smooth_mx), int(smooth_my),
                            region, screen_width, screen_height,
                            can_control,
                            gesture_recognizer.is_dragging())

        # Draw active control region (on top of everything)
        draw_control_region(frame, region)

        # Resize frame for display while preserving aspect ratio
        display_frame = resize_with_aspect_ratio(
            frame, DISPLAY_WIDTH, DISPLAY_HEIGHT)

        cv2.imshow("AI Gesture Desktop Assistant", display_frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
