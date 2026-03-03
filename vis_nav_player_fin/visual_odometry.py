"""
SuperPoint + SuperGlue based Visual Odometry
Tracks camera motion for 2D trajectory visualization
"""

import numpy as np
import cv2
import sys
import torch
import matplotlib.cm as cm
import logging
from config import *

# Add SuperGlue to path
sys.path.append(SUPERGLUE_PATH)

try:
    from models.matching import Matching
    from models.utils import frame2tensor, make_matching_plot_fast
    SUPERGLUE_AVAILABLE = True
except ImportError:
    SUPERGLUE_AVAILABLE = False
    logging.warning("SuperGlue not available! Visual odometry will be disabled.")

logging.basicConfig(level=logging.INFO)


class VisualOdometry:
    """
    SuperPoint + SuperGlue based visual odometry
    Estimates 2D pose (x, y, heading) from consecutive frames
    """
    
    def __init__(self):
        """
        Initialize visual odometry system
        """
        if not SUPERGLUE_AVAILABLE:
            logging.error("SuperGlue not available!")
            self.enabled = False
            return
        
        self.enabled = True
        
        # SuperGlue configuration
        config = {
            'superpoint': SUPERPOINT_CONFIG,
            'superglue': SUPERGLUE_CONFIG
        }
        
        # Initialize matching model
        self.device = 'cuda' if torch.cuda.is_available() and USE_GPU else 'cpu'
        self.matching = Matching(config).eval().to(self.device)
        
        # Camera intrinsics
        self.K = np.array([
            [CAMERA_FX, 0, CAMERA_CX],
            [0, CAMERA_FY, CAMERA_CY],
            [0, 0, 1]
        ])
        
        # Pose state
        self.cur_R = np.eye(3)
        self.cur_t = np.zeros(3)
        self.cur_heading = -np.pi / 4  # Initial heading
        
        # Initialize rotation matrix with heading
        self.cur_R[0, 0] = np.cos(self.cur_heading)
        self.cur_R[0, 2] = np.sin(self.cur_heading)
        self.cur_R[2, 0] = -np.sin(self.cur_heading)
        self.cur_R[2, 2] = np.cos(self.cur_heading)
        self.cur_R[1, 1] = 1
        
        # Previous frame
        self.prev_frame = None
        
        # Matching window
        self.show_matches = SHOW_FEATURE_MATCHES
        
        logging.info(f"Visual Odometry initialized (device: {self.device})")
    
    def reset(self):
        """
        Reset pose to origin
        """
        self.cur_R = np.eye(3)
        self.cur_t = np.zeros(3)
        self.cur_heading = -np.pi / 4
        
        # Reset rotation
        self.cur_R[0, 0] = np.cos(self.cur_heading)
        self.cur_R[0, 2] = np.sin(self.cur_heading)
        self.cur_R[2, 0] = -np.sin(self.cur_heading)
        self.cur_R[2, 2] = np.cos(self.cur_heading)
        self.cur_R[1, 1] = 1
        
        self.prev_frame = None
        
        logging.info("Visual odometry reset")
    
    def extract_and_match(self, img1, img2):
        """
        Extract SuperPoint features and match with SuperGlue
        
        Args:
            img1: first image (grayscale)
            img2: second image (grayscale)
            
        Returns:
            kpts0: keypoints in img1
            kpts1: keypoints in img2
            mkpts0: matched keypoints in img1
            mkpts1: matched keypoints in img2
            color: confidence colors for visualization
        """
        # Convert to grayscale if needed
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = img1
            
        if len(img2.shape) == 3:
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = img2
        
        # Convert to tensors
        frame1_tensor = frame2tensor(gray1, self.device)
        frame2_tensor = frame2tensor(gray2, self.device)
        
        # Extract features and match
        keys = ['keypoints', 'scores', 'descriptors']
        data1 = self.matching.superpoint({'image': frame1_tensor})
        data1 = {k + '0': data1[k] for k in keys}
        data1['image0'] = frame1_tensor
        
        pred = self.matching({**data1, 'image1': frame2_tensor})
        
        # Get results
        kpts0 = data1['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].detach().cpu().numpy()
        
        # Extract matched keypoints
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = confidence[valid]
        
        # Color for visualization
        color = cm.jet(mconf)
        
        return kpts0, kpts1, mkpts0, mkpts1, color
    
    def visualize_matches(self, img1, img2, kpts0, kpts1, mkpts0, mkpts1, color):
        """
        Visualize feature matches in a window
        
        Args:
            img1, img2: images
            kpts0, kpts1: all keypoints
            mkpts0, mkpts1: matched keypoints
            color: confidence colors
        """
        text = [
            'SuperPoint + SuperGlue',
            'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
            'Matches: {}'.format(len(mkpts0))
        ]
        
        small_text = [
            'Keypoint Threshold: {:.3f}'.format(SUPERPOINT_CONFIG['keypoint_threshold']),
            'Match Threshold: {:.2f}'.format(SUPERGLUE_CONFIG['match_threshold']),
        ]
        
        # Convert to grayscale for visualization
        if len(img1.shape) == 3:
            vis_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        else:
            vis_img1 = img1
            
        if len(img2.shape) == 3:
            vis_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            vis_img2 = img2
        
        out = make_matching_plot_fast(
            vis_img1, vis_img2, kpts0, kpts1, mkpts0, mkpts1, color, text,
            path=None, show_keypoints=True, small_text=small_text
        )
        
        cv2.imshow('SuperPoint Matches', out)
        cv2.waitKey(1)
    
    def estimate_motion(self, mkpts0, mkpts1):
        """
        Estimate motion from matched keypoints using homography
        
        Args:
            mkpts0: matched keypoints in frame 1
            mkpts1: matched keypoints in frame 2
            
        Returns:
            R: rotation matrix (3x3)
            t: translation vector (3,)
            valid: whether estimation was successful
        """
        if len(mkpts0) < 8:
            return None, None, False
        
        try:
            # Estimate homography
            H, mask = cv2.findHomography(mkpts1, mkpts0, cv2.RANSAC, 5.0)
            
            if H is None:
                return None, None, False
            
            # Decompose homography
            num, Rs, ts, normals = cv2.decomposeHomographyMat(H, self.K)
            
            if num == 0:
                return None, None, False
            
            # Select best decomposition (most inliers with positive depth)
            best_idx = 0
            best_score = -1
            
            for i in range(num):
                # Simple heuristic: prefer small translations
                score = -np.linalg.norm(ts[i])
                if score > best_score:
                    best_score = score
                    best_idx = i
            
            R = np.array(Rs[best_idx]).reshape(3, 3)
            t = np.array(ts[best_idx]).reshape(3)
            
            # Reject large translations (likely errors)
            if np.linalg.norm(t) > VO_MAX_TRANSLATION:
                return None, None, False
            
            return R, t, True
            
        except Exception as e:
            logging.debug(f"Motion estimation failed: {e}")
            return None, None, False
    
    def update(self, frame, show_matches=None):
        """
        Update pose with new frame
        
        Args:
            frame: current frame (BGR or grayscale)
            show_matches: override show_matches setting
            
        Returns:
            x: current x position
            y: current y position
            heading: current heading angle
        """
        if not self.enabled:
            return 0, 0, 0
        
        if show_matches is None:
            show_matches = self.show_matches
        
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Initialize on first frame
        if self.prev_frame is None:
            self.prev_frame = gray.copy()
            return self.cur_t[0], self.cur_t[2], self.cur_heading
        
        # Extract and match features
        try:
            kpts0, kpts1, mkpts0, mkpts1, color = self.extract_and_match(
                self.prev_frame, gray
            )
            
            # Visualize matches
            if show_matches and len(mkpts0) > 0:
                self.visualize_matches(
                    self.prev_frame, gray, kpts0, kpts1, mkpts0, mkpts1, color
                )
            
            # Estimate motion
            if len(mkpts0) >= 8:
                R, t, valid = self.estimate_motion(mkpts0, mkpts1)
                
                if valid:
                    # Update pose
                    scale = VO_SCALE_FACTOR
                    self.cur_t = self.cur_t + scale * self.cur_R.dot(t)
                    self.cur_R = R.dot(self.cur_R)
                    
                    # Update heading from rotation matrix
                    self.cur_heading = np.arctan2(self.cur_R[2, 0], self.cur_R[0, 0])
        
        except Exception as e:
            logging.debug(f"VO update failed: {e}")
        
        # Update previous frame
        self.prev_frame = gray.copy()
        
        return self.cur_t[0], self.cur_t[2], self.cur_heading
    
    def get_pose(self):
        """
        Get current pose
        
        Returns:
            x, y, heading
        """
        return self.cur_t[0], self.cur_t[2], self.cur_heading


if __name__ == "__main__":
    # Test visual odometry
    if SUPERGLUE_AVAILABLE:
        vo = VisualOdometry()
        logging.info("Visual odometry test initialization successful")
    else:
        logging.error("Cannot test: SuperGlue not available")