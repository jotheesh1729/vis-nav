"""
Main Player - Orchestrates complete navigation system
First run: Manual exploration with 4 windows
Second run: Autonomous navigation with 3 windows
"""

from vis_nav_game import Player, Action, Phase
import pygame
import cv2
import numpy as np
import os
import json
import pickle
import time
import logging
from config import *

# Import our modules
from feature_extractor import ResNetExtractor
from vlad_encoder import VLADEncoder
from visual_odometry import VisualOdometry
from graph_planner import NavigationGraph
from trajectory_visualizer import TrajectoryVisualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


class MainPlayer(Player):
    """
    Complete navigation system with manual exploration and autonomous replay
    """
    
    def __init__(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.keymap = None
        super(MainPlayer, self).__init__()
        
        # ====================================================================
        # MODULES
        # ====================================================================
        self.resnet = ResNetExtractor(RESNET_MODEL)
        self.vlad = VLADEncoder()
        self.vo = VisualOdometry()
        self.graph_planner = NavigationGraph()
        self.traj_viz = TrajectoryVisualizer("2D Trajectory")
        
        # ====================================================================
        # RECORDING (first run)
        # ====================================================================
        self.actions = []  # Recorded action sequence
        self.images = []   # Sampled images
        self.poses = []    # Recorded poses from VO
        self.frame_count = 0
        
        # ====================================================================
        # NAVIGATION (second run)
        # ====================================================================
        self.resnet_features = None
        self.resnet_image_files = None
        self.current_path = None
        self.goal_idx = None
        self.navigation_index = 0
        
        # Next best view display
        self.next_best_view_img = None
        
        # ====================================================================
        # MODE DETECTION
        # ====================================================================
        actions_file = os.path.join(SAVE_DIR, "actions.json")
        if os.path.exists(actions_file):
            # Replay mode - load recorded actions
            with open(actions_file, "r") as f:
                self.actions = json.load(f)
            self.replay_mode = True
            logging.info(f"REPLAY MODE: Loaded {len(self.actions)} actions")
            
            # Load manual trail for overlay
            self.traj_viz.load_manual_trail()
        else:
            # Recording mode
            self.replay_mode = False
            logging.info("RECORDING MODE")
        
        # ====================================================================
        # DISPLAY
        # ====================================================================
        self.show_matches = SHOW_FEATURE_MATCHES
        
        # Throttling for action recording
        self.last_action_time = 0
        self.action_delay = 0.05  # Faster response
        
        # Pre-load ResNet features for Next Best View
        self._preload_database_features()
        
        logging.info("="*70)
        if self.replay_mode:
            logging.info("MODE: AUTONOMOUS REPLAY")
            logging.info("3 Windows: SuperPoint Matches | Main FPV | 2D Trajectory")
        else:
            logging.info("MODE: MANUAL EXPLORATION")
            logging.info("4 Windows: SuperPoint Matches | 2D Trajectory | Main FPV | Next Best View")
        logging.info("="*70)
    
    def _preload_database_features(self):
        """
        Pre-load ResNet features from images_subsample for Next Best View
        This runs in background during exploration
        """
        if not os.path.exists(DATA_DIR):
            logging.warning(f"Data directory {DATA_DIR} not found, Next Best View disabled")
            return
        
        try:
            logging.info("Pre-loading database features for Next Best View...")
            self.resnet_features, self.resnet_image_files = self.resnet.extract_database(
                DATA_DIR, 
                cache_file='database_resnet_features.pkl'
            )
            logging.info(f"Loaded {len(self.resnet_features)} features for Next Best View")
        except Exception as e:
            logging.warning(f"Could not pre-load features: {e}")
            self.resnet_features = None
            self.resnet_image_files = None
    
    def reset(self):
        """
        Reset player state
        """
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        
        pygame.init()
        
        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT
        }
    
    def pre_exploration(self):
        """
        Called before exploration phase starts
        """
        logging.info("="*70)
        logging.info("EXPLORATION PHASE STARTED")
        logging.info("Use arrow keys to navigate, M to toggle matches, ESC to save")
        logging.info("="*70)
        super(MainPlayer, self).pre_exploration()
    
    def act(self):
        """
        Handle player actions
        """
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if not self.replay_mode:
                    self.save_and_quit()
                return Action.QUIT
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if not self.replay_mode:
                        self.save_and_quit()
                    return Action.QUIT
                
                elif event.key == pygame.K_m:
                    # Toggle feature matching display
                    self.show_matches = not self.show_matches
                    logging.info(f"Feature matching: {'ON' if self.show_matches else 'OFF'}")
                
                elif event.key in self.keymap:
                    self.last_act |= self.keymap[event.key]
                else:
                    self.show_target_images()
            
            if event.type == pygame.KEYUP:
                if event.key in self.keymap:
                    self.last_act ^= self.keymap[event.key]
        
        # REPLAY MODE - autonomous navigation
        if self.replay_mode and self._state and self._state[1] == Phase.NAVIGATION:
            return self._autonomous_navigation()
        
        # RECORDING MODE - manual control with action recording
        if not self.replay_mode:
            return self._manual_exploration()
        
        return self.last_act
    
    def _manual_exploration(self):
        """
        Handle manual exploration mode
        
        Returns:
            action to execute
        """
        # Record every action immediately (no throttling for IDLE actions)
        if self.last_act != Action.IDLE and self.last_act != Action.QUIT:
            # Throttle only to avoid duplicate recordings
            current_time = time.time()
            if current_time - self.last_action_time < self.action_delay:
                return self.last_act  # Still executing, but don't record duplicate
            
            self.last_action_time = current_time
            
            # Record action
            self.actions.append(self.last_act.value)
            self.frame_count += 1
            
            # Update visual odometry and trajectory
            if self.vo.enabled and self.fpv is not None:
                x, y, heading = self.vo.update(self.fpv, show_matches=self.show_matches)
                self.poses.append([x, y, heading])
                self.traj_viz.add_point(x, y, heading)
            
            # Sample images
            if self.frame_count % SAMPLE_RATE == 0 and self.fpv is not None:
                self.images.append(self.fpv.copy())
            
            # Log progress
            if len(self.actions) % 50 == 0:
                logging.info(f"Recorded: {len(self.actions)} actions, {len(self.images)} images")
        
        return self.last_act
    
    def _autonomous_navigation(self):
        """
        Handle autonomous navigation mode
        
        Returns:
            action to execute
        """
        # Use recorded actions to navigate
        if self.navigation_index < len(self.actions):
            action_value = self.actions[self.navigation_index]
            self.navigation_index += 1
            
            if self.navigation_index % 10 == 0:
                logging.info(f"Replaying action {self.navigation_index}/{len(self.actions)}")
            
            return Action(action_value)
        else:
            # Reached end of recorded actions
            logging.info("Replay complete, checking in at goal")
            return Action.CHECKIN
    
    def see(self, fpv):
        """
        Process first-person view
        
        Args:
            fpv: first-person view image
        """
        if fpv is None or len(fpv.shape) < 3:
            return
        
        self.fpv = fpv
        
        # Initialize screen
        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))
        
        # DISPLAY FPV
        self._display_fpv(fpv)
        
        # Update based on mode
        if self._state:
            phase = self._state[1]
            
            if phase == Phase.EXPLORATION and not self.replay_mode:
                self._exploration_display()
            
            elif phase == Phase.NAVIGATION:
                self._navigation_display()
    
    def _display_fpv(self, fpv):
        """
        Display first-person view window
        
        Args:
            fpv: FPV image
        """
        def convert_opencv_img_to_pygame(opencv_image):
            opencv_image = opencv_image[:, :, ::-1]  # BGR->RGB
            shape = opencv_image.shape[1::-1]
            pygame_image = pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')
            return pygame_image
        
        mode = "REPLAY" if self.replay_mode else "RECORDING"
        pygame.display.set_caption(f"{mode} - {len(self.actions)} actions")
        
        rgb = convert_opencv_img_to_pygame(fpv)
        self.screen.blit(rgb, (0, 0))
        pygame.display.update()
    
    def _exploration_display(self):
        """
        Display windows during exploration phase (manual mode)
        Window 1: SuperPoint matches (handled by VO)
        Window 2: 2D trajectory
        Window 3: Main FPV (handled in _display_fpv)
        Window 4: Next best view
        """
        # Update trajectory display
        self.traj_viz.show(is_manual=True, show_manual_trail=False)
        
        # Display next best view (if ResNet features are available)
        if self.resnet_features is not None and self.fpv is not None:
            self._display_next_best_view()
    
    def _navigation_display(self):
        """
        Display windows during navigation phase
        Window 1: SuperPoint matches (handled by VO)
        Window 2: 2D trajectory (with overlay)
        Window 3: Main FPV (handled in _display_fpv)
        """
        # Update visual odometry
        if self.vo.enabled and self.fpv is not None:
            x, y, heading = self.vo.update(self.fpv, show_matches=self.show_matches)
            self.traj_viz.add_point(x, y, heading)
        
        # Display trajectory with manual trail overlay
        self.traj_viz.show(is_manual=False, show_manual_trail=True)
    
    def _display_next_best_view(self):
        """
        Display next best view using ResNet similarity
        """
        if self.resnet_features is None or self.fpv is None:
            return
        
        # Extract features from current view
        current_feat = self.resnet.extract_single(self.fpv)
        
        # Find most similar image
        indices, similarities = self.resnet.find_most_similar(
            current_feat, self.resnet_features, top_k=1
        )
        
        best_idx = indices[0]
        similarity = similarities[0]
        
        # Load and display the image
        if self.resnet_image_files and best_idx < len(self.resnet_image_files):
            img_path = os.path.join(DATA_DIR, self.resnet_image_files[best_idx])
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                
                # Add text overlay
                text = f"Most Similar: #{best_idx} (sim: {similarity:.3f})"
                cv2.putText(img, text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("Next Best View", img)
                cv2.waitKey(1)
    
    def pre_navigation(self):
        """
        Preparation phase between exploration and navigation
        """
        super(MainPlayer, self).pre_navigation()
        
        if not self.replay_mode:
            # Save exploration data
            self.save_data()
        
        if len(self.images) == 0:
            logging.error("No images recorded!")
            return
        
        logging.info("="*70)
        logging.info("PRE-NAVIGATION COMPUTATION")
        logging.info("="*70)
        
        # Extract ResNet features
        logging.info("[1/4] Extracting ResNet features...")
        self.resnet_features, _ = self.resnet.extract_database(
            image_list=self.images,
            cache_file='resnet_exploration_features.pkl'
        )
        
        # Build VLAD database
        logging.info("[2/4] Building VLAD database...")
        sift_desc = self.vlad.compute_sift_features(self.images)
        self.vlad.build_codebook(sift_desc)
        self.vlad.build_database(self.images)
        self.vlad.build_tree()
        
        # Build navigation graph
        logging.info("[3/4] Building navigation graph...")
        self.graph_planner.build_graph(self.resnet_features, self.actions)
        
        # Find goal location
        logging.info("[4/4] Finding goal location...")
        targets = self.get_target_images()
        if targets is not None and len(targets) > 0:
            # Use VLAD to find goal
            goal_indices, _ = self.vlad.find_nearest(targets[0], k=1)
            goal_image_idx = goal_indices[0]
            
            # Convert to action index
            goal_action_idx = goal_image_idx * SAMPLE_RATE
            
            logging.info(f"Goal found at image {goal_image_idx} -> action {goal_action_idx}")
            
            # Trim actions to reach goal
            if goal_action_idx < len(self.actions):
                self.actions = self.actions[:goal_action_idx]
                logging.info(f"Trimmed to {len(self.actions)} actions")
        
        # Save trajectory trail
        self.traj_viz.save_trail()
        
        # Reset trajectory for navigation phase
        self.traj_viz.reset()
        self.vo.reset()
        
        logging.info("="*70)
        logging.info("READY FOR NAVIGATION!")
        logging.info("="*70)
    
    def save_data(self):
        """
        Save exploration data
        """
        if len(self.actions) == 0:
            logging.error("No actions to save!")
            return
        
        logging.info("="*70)
        logging.info("SAVING EXPLORATION DATA")
        
        # Save actions
        with open(os.path.join(SAVE_DIR, "actions.json"), "w") as f:
            json.dump(self.actions, f)
        logging.info(f"Saved {len(self.actions)} actions")
        
        # Save poses
        with open(os.path.join(SAVE_DIR, "poses.pkl"), "wb") as f:
            pickle.dump(self.poses, f)
        logging.info(f"Saved {len(self.poses)} poses")
        
        # Save images
        images_dir = os.path.join(SAVE_DIR, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        for i, img in enumerate(self.images):
            cv2.imwrite(os.path.join(images_dir, f"{i}.jpg"), img)
        logging.info(f"Saved {len(self.images)} images")
        
        logging.info("="*70)
    
    def save_and_quit(self):
        """
        Save data and quit
        """
        self.save_data()
        pygame.quit()
        self.last_act = Action.QUIT
    
    def show_target_images(self):
        """
        Display target images in 2x2 grid
        """
        targets = self.get_target_images()
        if targets is None or len(targets) == 0:
            return
        
        # Save target images
        for i, img in enumerate(targets):
            if img is not None and img.size > 0:
                cv2.imwrite(os.path.join(SAVE_DIR, f"target_{i}.jpg"), img)
        
        # Create 2x2 grid
        if all(img is not None and img.size > 0 for img in targets):
            hor1 = cv2.hconcat(targets[:2])
            hor2 = cv2.hconcat(targets[2:])
            concat = cv2.vconcat([hor1, hor2])
            
            w, h = concat.shape[:2]
            color = (0, 0, 0)
            
            # Add grid lines
            concat = cv2.line(concat, (int(h/2), 0), (int(h/2), w), color, 2)
            concat = cv2.line(concat, (0, int(w/2)), (h, int(w/2)), color, 2)
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(concat, 'Front View', (10, 25), font, 0.75, color, 1, cv2.LINE_AA)
            cv2.putText(concat, 'Right View', (int(h/2)+10, 25), font, 0.75, color, 1, cv2.LINE_AA)
            cv2.putText(concat, 'Back View', (10, int(w/2)+25), font, 0.75, color, 1, cv2.LINE_AA)
            cv2.putText(concat, 'Left View', (int(h/2)+10, int(w/2)+25), font, 0.75, color, 1, cv2.LINE_AA)
            
            cv2.imshow('Target Images', concat)
            cv2.waitKey(1)
    
    def set_target_images(self, images):
        """
        Set target images
        """
        super(MainPlayer, self).set_target_images(images)
        self.show_target_images()


if __name__ == "__main__":
    import vis_nav_game
    vis_nav_game.play(the_player=MainPlayer())