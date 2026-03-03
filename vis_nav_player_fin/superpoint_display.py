"""
SuperPoint + SuperGlue Feature Matching Display
Shows impressive feature tracking between frames
"""

import cv2
import numpy as np
import torch
import matplotlib.cm as cm
import sys
import logging
from config_demo import *

sys.path.append(SUPERGLUE_PATH)

try:
    from models.matching import Matching
    from models.utils import frame2tensor, make_matching_plot_fast
    SUPERGLUE_AVAILABLE = True
except ImportError:
    SUPERGLUE_AVAILABLE = False
    logging.warning("SuperGlue not available!")

logging.basicConfig(level=logging.INFO)


class SuperPointDisplay:
    """
    Display SuperPoint feature matches between consecutive frames
    """
    
    def __init__(self):
        """
        Initialize SuperPoint + SuperGlue matcher
        """
        if not SUPERGLUE_AVAILABLE:
            logging.error("SuperGlue not available!")
            self.enabled = False
            return
        
        self.enabled = True
        
        # Initialize matching model
        config = {
            'superpoint': SUPERPOINT_CONFIG,
            'superglue': SUPERGLUE_CONFIG
        }
        
        self.device = 'cuda' if torch.cuda.is_available() and USE_GPU else 'cpu'
        self.matching = Matching(config).eval().to(self.device)
        
        self.prev_frame = None
        
        logging.info(f"SuperPoint display initialized (device: {self.device})")
    
    def process_and_display(self, frame, window_name="SuperPoint Feature Matching"):
        """
        Process frame and display matches with previous frame
        
        Args:
            frame: current frame (BGR)
            window_name: OpenCV window name
        """
        if not self.enabled:
            return
        
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Initialize on first frame
        if self.prev_frame is None:
            self.prev_frame = gray.copy()
            return
        
        try:
            # Extract and match features
            kpts0, kpts1, mkpts0, mkpts1, color = self._extract_and_match(
                self.prev_frame, gray
            )
            
            # Display matches
            if len(mkpts0) > 0:
                self._visualize_matches(
                    self.prev_frame, gray, 
                    kpts0, kpts1, 
                    mkpts0, mkpts1, 
                    color, 
                    window_name
                )
        
        except Exception as e:
            logging.debug(f"Matching failed: {e}")
        
        # Update previous frame
        self.prev_frame = gray.copy()
    
    def _extract_and_match(self, img1, img2):
        """
        Extract SuperPoint features and match with SuperGlue
        """
        # Convert to tensors
        frame1_tensor = frame2tensor(img1, self.device)
        frame2_tensor = frame2tensor(img2, self.device)
        
        # Extract and match
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
    
    def _visualize_matches(self, img1, img2, kpts0, kpts1, mkpts0, mkpts1, color, window_name):
        """
        Create and display matching visualization
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
        
        out = make_matching_plot_fast(
            img1, img2, kpts0, kpts1, mkpts0, mkpts1, color, text,
            path=None, show_keypoints=True, small_text=small_text
        )
        
        # Force window creation
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, out)
        cv2.waitKey(1)
    
    def reset(self):
        """
        Reset previous frame
        """
        self.prev_frame = None


if __name__ == "__main__":
    # Test with dummy frames
    display = SuperPointDisplay()
    
    if display.enabled:
        # Create test frames
        frame1 = np.random.randint(0, 255, (240, 320), dtype=np.uint8)
        frame2 = frame1 + np.random.randint(-10, 10, (240, 320), dtype=np.uint8)
        
        display.process_and_display(frame1)
        display.process_and_display(frame2)
        
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
        
        logging.info("SuperPoint display test complete")