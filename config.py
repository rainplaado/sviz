# SnailViz Configuration
# Default settings and constants

# App metadata
APP_VERSION = "2.0.0"
APP_TITLE = "SnailCam Visualizer"

# Default heatmap settings
DEFAULT_SMOOTHING = 0.40
DEFAULT_PIXEL_SIZE = 3  # meters
DEFAULT_SEARCH_RADIUS = 56  # meters
DEFAULT_EXTRAPOLATION = 58  # meters (fill distance)
DEFAULT_BLUR_SIGMA = 3.36
DEFAULT_APPLY_BLUR = True  # Smooth edges enabled by default

# Default zone settings
DEFAULT_ZONE_COUNT = 3
DEFAULT_THRESHOLDS = [0.5, 1.5]  # Lower defaults for low snail count data
DEFAULT_MIN_AREA = 0
DEFAULT_CLEAR_THRESHOLD = 0.1  # Below this density = "clear" (no snails)

# Default detection map settings
DEFAULT_MARKER_RADIUS = 3
DEFAULT_ZOOM = 17

# Color settings — zone scheme: 0=no data, 1=clear, 2+=risk zones
NO_DATA_COLOR = (0.95, 0.95, 0.95)     # Light gray
CLEAR_ZONE_COLOR = (0.7, 0.95, 0.7)    # Pale green — no snails detected
RISK_ZONE_COLORS = {
    1: [(1.0, 0.0, 0.0)],                                           # 1 risk zone: red (bait zone)
    2: [(1.0, 1.0, 0.0), (1.0, 0.0, 0.0)],                         # 2 risk zones: yellow, red
    3: [(1.0, 1.0, 0.0), (1.0, 0.6, 0.0), (1.0, 0.0, 0.0)],       # 3 risk zones: yellow, orange, red
}

# Legacy ZONE_COLORS kept for any remaining references (5 entries for 3-zone mode)
ZONE_COLORS = [
    NO_DATA_COLOR,          # 0: No data
    CLEAR_ZONE_COLOR,       # 1: Clear
    (1.0, 1.0, 0.0),       # 2: Low risk (yellow)
    (1.0, 0.6, 0.0),       # 3: Medium risk (orange)
    (1.0, 0.0, 0.0),       # 4: High risk (red)
]

ZONE_COLOR_NAMES = ["gray", "green", "yellow", "orange", "red"]

# Heatmap colormap colors
HEATMAP_COLORS = ["green", "yellow", "orange", "red"]

# Default snail count range
DEFAULT_MIN_COUNT = 0
DEFAULT_MAX_COUNT = 3  # Lower for typical low-count data

# Confidence filtering
DEFAULT_MIN_CONFIDENCE = 0.3
DEFAULT_MAX_CONFIDENCE = 1.0

# Required columns for different file types
REQUIRED_COLS_HARVEST = ["Longitude", "Latitude", "Conical", "Italian"]
REQUIRED_COLS_DETECTIONS = ["Longitude", "Latitude", "Snail_Type", "Confidence"]
