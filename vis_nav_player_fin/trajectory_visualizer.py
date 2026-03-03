"""
2D Trajectory Visualization
Displays top-down view of robot path with orientation
"""

import cv2
import numpy as np
import pickle
import os
import logging
from config import *

logging.basicConfig(level=logging.INFO)


class TrajectoryVisualizer:
    """
    Visualize 2D trajectory with current pose and trail
    """
    
    def __init__(self, window_name="Trajectory", map_size=None):
        """
        Initialize trajectory visualizer
        
        Args:
            window_name: name of visualization window
            map_size: size of map (pixels)
        """
        self.window_name = window_name
        self.map_size = map_size if map_size else TRAJ_MAP_SIZE
        
        # Trajectory storage
        self.points = []  # List of (x, y) points
        self.manual_trail = []  # Trail from manual exploration
        
        # Current pose
        self.current_x = 0
        self.current_y = 0
        self.current_heading = 0
        
        # Display settings
        self.origin_x = TRAJ_ORIGIN_X
        self.origin_y = TRAJ_ORIGIN_Y
        self.scale = TRAJ_SCALE
        
        logging.info(f"Trajectory visualizer initialized: {window_name}")
    
    def reset(self):
        """
        Reset trajectory (clear all points)
        """
        self.points = []
        self.current_x = 0
        self.current_y = 0
        self.current_heading = 0
        logging.info("Trajectory reset")
    
    def add_point(self, x, y, heading=None):
        """
        Add a point to trajectory
        
        Args:
            x: x coordinate (world space)
            y: y coordinate (world space)
            heading: heading angle (radians, optional)
        """
        self.current_x = x
        self.current_y = y
        
        if heading is not None:
            self.current_heading = heading
        
        # Convert to screen coordinates
        screen_x = int(x * self.scale) + self.origin_x
        screen_y = int(y * self.scale) + self.origin_y
        
        # Add to points list (avoid duplicates)
        if len(self.points) == 0 or (screen_x, screen_y) != self.points[-1]:
            self.points.append((screen_x, screen_y))
    
    def save_trail(self, filename="manual_trail.pkl"):
        """
        Save current trajectory as manual trail
        
        Args:
            filename: filename to save trail
        """
        filepath = os.path.join(SAVE_DIR, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(self.points.copy(), f)
        logging.info(f"Trail saved to {filepath}")
    
    def load_manual_trail(self, filename="manual_trail.pkl"):
        """
        Load manual trail for overlay
        
        Args:
            filename: filename of saved trail
        """
        filepath = os.path.join(SAVE_DIR, filename)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.manual_trail = pickle.load(f)
            logging.info(f"Manual trail loaded: {len(self.manual_trail)} points")
        else:
            logging.warning(f"Manual trail file not found: {filepath}")
    
    def draw_trail(self, img, points, color, thickness=TRAIL_THICKNESS):
        """
        Draw trajectory trail on image
        
        Args:
            img: image to draw on
            points: list of (x, y) points
            color: BGR color
            thickness: line thickness
        """
        if len(points) < 2:
            return
        
        for i in range(1, len(points)):
            pt1 = points[i - 1]
            pt2 = points[i]
            cv2.line(img, pt1, pt2, color, thickness)
    
    def draw_pose_marker(self, img, x, y, heading, color, size=MARKER_RADIUS):
        """
        Draw current pose marker with orientation arrow
        
        Args:
            img: image to draw on
            x: x coordinate (screen space)
            y: y coordinate (screen space)
            heading: heading angle (radians)
            color: BGR color
            size: marker size
        """
        # Draw circle for position
        cv2.circle(img, (x, y), size, color, MARKER_THICKNESS)
        
        # Draw arrow for heading
        arrow_length = size * 3
        end_x = int(x + arrow_length * np.cos(heading))
        end_y = int(y + arrow_length * np.sin(heading))
        cv2.arrowedLine(img, (x, y), (end_x, end_y), color, 2, tipLength=0.5)
    
    def render(self, is_manual=True, show_manual_trail=False):
        """
        Render trajectory visualization
        
        Args:
            is_manual: whether this is manual exploration (affects colors)
            show_manual_trail: whether to overlay manual trail
            
        Returns:
            img: rendered image
        """
        # Create blank image
        img = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        
        # Draw manual trail if requested
        if show_manual_trail and len(self.manual_trail) > 0:
            self.draw_trail(img, self.manual_trail, COLOR_MANUAL_TRAIL)
        
        # Draw current trail
        if is_manual:
            trail_color = COLOR_MANUAL_TRAIL
            pose_color = COLOR_CURRENT_MANUAL
        else:
            trail_color = COLOR_AUTO_TRAIL
            pose_color = COLOR_CURRENT_AUTO
        
        self.draw_trail(img, self.points, trail_color)
        
        # Draw current pose
        if len(self.points) > 0:
            screen_x = self.points[-1][0]
            screen_y = self.points[-1][1]
            self.draw_pose_marker(img, screen_x, screen_y, 
                                 self.current_heading, pose_color)
        
        # Add info text
        self._add_info_text(img, is_manual)
        
        return img
    
    def _add_info_text(self, img, is_manual):
        """
        Add information text overlay
        
        Args:
            img: image to draw on
            is_manual: whether this is manual mode
        """
        # Background rectangle
        cv2.rectangle(img, (10, 20), (600, 70), (0, 0, 0), -1)
        
        # Mode text
        mode = "MANUAL EXPLORATION" if is_manual else "AUTONOMOUS NAVIGATION"
        cv2.putText(img, mode, (20, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Coordinates
        text = f"Position: x={self.current_x:.1f}m, y={self.current_y:.1f}m"
        cv2.putText(img, text, (20, 65),
                   cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
    
    def show(self, is_manual=True, show_manual_trail=False):
        """
        Render and display trajectory
        
        Args:
            is_manual: whether this is manual exploration
            show_manual_trail: whether to overlay manual trail
        """
        img = self.render(is_manual, show_manual_trail)
        cv2.imshow(self.window_name, img)
        cv2.waitKey(1)
    
    def update_and_show(self, x, y, heading=None, is_manual=True, show_manual_trail=False):
        """
        Update pose and display
        
        Args:
            x: x coordinate
            y: y coordinate
            heading: heading angle (optional)
            is_manual: whether this is manual mode
            show_manual_trail: whether to overlay manual trail
        """
        self.add_point(x, y, heading)
        self.show(is_manual, show_manual_trail)


if __name__ == "__main__":
    # Test trajectory visualizer
    logging.info("Testing trajectory visualizer...")
    
    viz = TrajectoryVisualizer()
    
    # Simulate a path
    for i in range(100):
        t = i / 10.0
        x = 50 * np.cos(t)
        y = 50 * np.sin(t)
        heading = t
        
        viz.update_and_show(x, y, heading, is_manual=True)
        cv2.waitKey(30)
    
    # Save trail
    viz.save_trail("test_trail.pkl")
    
    # Test overlay
    viz.reset()
    viz.load_manual_trail("test_trail.pkl")
    
    for i in range(50):
        t = i / 10.0
        x = 50 * np.cos(t) + 10
        y = 50 * np.sin(t) + 10
        heading = t
        
        viz.update_and_show(x, y, heading, is_manual=False, show_manual_trail=True)
        cv2.waitKey(30)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()