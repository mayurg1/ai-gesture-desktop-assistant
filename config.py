# ---------------------------------------------------------------------------
# Configuration for AI Gesture Desktop Assistant
# ---------------------------------------------------------------------------

# Mouse control - set True to enable Windows cursor movement
ENABLE_MOUSE_CONTROL = True

# Minimum pixel distance to trigger mouse movement (deadzone)
CURSOR_DEADZONE = 2

# Mirror settings for natural cursor movement
MIRROR_CAMERA = True   # Flip frame horizontally for mirror-like preview
MIRROR_CURSOR = True   # Map X coordinate to match mirrored preview

# Pinch gesture settings
PINCH_THRESHOLD = 35       # Maximum distance (pixels) for pinch detection
CLICK_COOLDOWN_MS = 300    # Minimum time between clicks (milliseconds)
DRAG_HOLD_MS = 300         # Hold duration (ms) before drag starts
