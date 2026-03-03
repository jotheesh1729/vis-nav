"""
Configuration file for visual navigation system
All hyperparameters and paths in one place
"""

import os

# ============================================================================
# PATHS
# ============================================================================
DATA_DIR = "data/images_final/"
CACHE_DIR = "cache/"
SAVE_DIR = "exploration_data/"

# SuperGlue model path
SUPERGLUE_PATH = "SuperGluePretrainedNetwork"

# Create directories if they don't exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================================
# RESNET SETTINGS
# ============================================================================
RESNET_MODEL = "resnet50"  # or 'resnet18'
FEATURE_DIM = 2048  # 2048 for resnet50, 512 for resnet18
IMG_SIZE = (224, 224)
USE_GPU = True

# ============================================================================
# VLAD SETTINGS
# ============================================================================
VLAD_CLUSTERS = 64  # Number of visual words
KMEANS_INIT = "k-means++"
KMEANS_N_INIT = 3

# ============================================================================
# SUPERPOINT + SUPERGLUE SETTINGS
# ============================================================================
SUPERPOINT_CONFIG = {
    'nms_radius': 4,
    'keypoint_threshold': 0.005,
    'max_keypoints': 2000
}

SUPERGLUE_CONFIG = {
    'weights': 'outdoor',
    'sinkhorn_iterations': 20,
    'match_threshold': 0.2,
}

# Visual Odometry
VO_SCALE_FACTOR = 2.0  # Scale for trajectory
VO_MAX_TRANSLATION = 0.06  # Reject large jumps

# Camera intrinsics (from game)
CAMERA_FX = 92
CAMERA_FY = 92
CAMERA_CX = 160
CAMERA_CY = 120

# ============================================================================
# GRAPH SETTINGS
# ============================================================================
K_NEIGHBORS = 10  # Number of spatial edges per node
TEMPORAL_WINDOW = 5  # Don't connect nearby frames spatially
SIMILARITY_THRESHOLD = 0.7  # Minimum similarity for spatial edges
TEMPORAL_EDGE_WEIGHT = 1.0
SPATIAL_EDGE_WEIGHT_MULT = 2.0

# ============================================================================
# A* PATHFINDING SETTINGS
# ============================================================================
HEURISTIC_WEIGHT = 1.5

# ============================================================================
# TRAJECTORY VISUALIZATION SETTINGS
# ============================================================================
TRAJ_MAP_SIZE = 720  # 720x720 pixel map
TRAJ_ORIGIN_X = 360  # Center of map
TRAJ_ORIGIN_Y = 360
TRAJ_SCALE = 1.0  # Scale factor for display

# Colors (BGR format for OpenCV)
COLOR_MANUAL_TRAIL = (255, 0, 0)      # Blue trail for manual exploration
COLOR_AUTO_TRAIL = (0, 255, 0)        # Green trail for autonomous navigation
COLOR_CURRENT_MANUAL = (0, 255, 255)  # Yellow for current position (manual)
COLOR_CURRENT_AUTO = (0, 0, 255)      # Red for current position (auto)
COLOR_GOAL = (0, 0, 255)              # Red for goal marker

# Line thickness
TRAIL_THICKNESS = 2
MARKER_RADIUS = 5
MARKER_THICKNESS = -1  # Filled circle

# ============================================================================
# RECORDING SETTINGS
# ============================================================================
SAMPLE_RATE = 2  # Save every Nth frame during exploration

# ============================================================================
# WINDOW POSITIONS (to avoid overlap)
# ============================================================================
WINDOW_POSITIONS = {
    'fpv': (0, 0),
    'matching': (350, 0),
    'trajectory': (900, 0),
    'next_best_view': (0, 300)
}

# ============================================================================
# NAVIGATION SETTINGS
# ============================================================================
REPLAN_INTERVAL = 10  # Replan path every N steps
GOAL_THRESHOLD = 0.15  # Distance threshold to consider goal reached

# ============================================================================
# DEBUG SETTINGS
# ============================================================================
DEBUG_MODE = False
SHOW_FEATURE_MATCHES = True
VERBOSE_LOGGING = True





