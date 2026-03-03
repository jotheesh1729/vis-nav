"""
Hybrid Autonomous Navigation Player
- Uses ResNet50 for place recognition
- Records exploration actions
- Replays actions autonomously during navigation
- Manual override when stuck (press any arrow key)
- Toggle back to auto with 'A' key
"""

from vis_nav_game import Player, Action, Phase
import pygame
import cv2
import numpy as np
import os
import pickle
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.neighbors import BallTree

class HybridAutonomousPlayer(Player):
    def __init__(self):
        print("="*70)
        print("HYBRID AUTONOMOUS NAVIGATION PLAYER")
        print("ResNet50 + Action Replay + Manual Override")
        print("="*70)
        
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.keymap = None
        
        # Paths
        self.data_dir = "data/images_final/"
        self.cache_dir = "cache/"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Feature extraction
        self.feature_extractor = None
        self.features = None
        self.image_files = None
        self.tree = None
        
        # Action recording/replay
        self.actions_file = os.path.join(self.cache_dir, "exploration_actions.json")
        self.actions = []
        self.replay_index = 0
        self.goal_action_index = None
        
        # Navigation state
        self.AUTO_NAV = True  # Toggle for auto/manual
        self.goal_idx = None
        self.localize_counter = 0
        self.localize_interval = 5
        
        # Check if we have recorded actions
        if os.path.exists(self.actions_file):
            with open(self.actions_file, 'r') as f:
                self.actions = json.load(f)
            print(f"[LOADED] {len(self.actions)} recorded actions")
            self.mode = "REPLAY"
        else:
            print("[RECORDING MODE] Navigate manually, actions will be saved")
            self.mode = "RECORD"
        
        super(HybridAutonomousPlayer, self).__init__()
    
    def reset(self):
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
    
    def load_feature_extractor(self):
        """Load ResNet50 feature extractor"""
        print("\n[1/3] Loading ResNet50 feature extractor...")
        
        model = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
        self.feature_extractor.eval()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor.to(self.device)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        print(f"  Device: {self.device}")
    
    def extract_features(self, image):
        """Extract ResNet50 features from image"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.feature_extractor(img_tensor)
        
        features = features.squeeze().cpu().numpy()
        features = features / (np.linalg.norm(features) + 1e-6)
        
        return features
    
    def load_or_extract_features(self):
        """Load cached features or extract from exploration images"""
        cache_file = os.path.join(self.cache_dir, "resnet_features.pkl")
        
        if os.path.exists(cache_file):
            print("\n[2/3] Loading cached features...")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            self.features = data['features']
            self.image_files = data['image_files']
            print(f"  Features: {len(self.features)} vectors")
        else:
            print("\n[2/3] Extracting features from exploration images...")
            self.image_files = sorted([f for f in os.listdir(self.data_dir) 
                                      if f.endswith(('.png', '.jpg', '.jpeg'))])
            
            print(f"  Found {len(self.image_files)} images")
            
            features_list = []
            for i, img_file in enumerate(self.image_files):
                if i % 100 == 0:
                    print(f"  Progress: {i}/{len(self.image_files)}")
                
                img_path = os.path.join(self.data_dir, img_file)
                img = Image.open(img_path).convert('RGB')
                feat = self.extract_features(img)
                features_list.append(feat)
            
            self.features = np.array(features_list)
            
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'features': self.features,
                    'image_files': self.image_files
                }, f)
            print(f"  Cached features to {cache_file}")
    
    def build_index(self):
        """Build BallTree index for fast nearest neighbor search"""
        print("\n[3/3] Building spatial index...")
        self.tree = BallTree(self.features, leaf_size=40)
        print("  Index ready")
    
    def find_closest_node(self, query_image):
        """Find closest node in database to query image"""
        query_feat = self.extract_features(query_image)
        _, indices = self.tree.query(query_feat.reshape(1, -1), k=1)
        return indices[0][0]
    
    def find_goal_in_actions(self):
        """Find goal location in recorded actions and trim sequence"""
        print("\n[GOAL DETECTION] Finding goal in exploration data...")
        
        targets = self.get_target_images()
        if targets is None or len(targets) == 0:
            print("  ERROR: No target images!")
            return
        
        # Find goal using front view
        goal_node = self.find_closest_node(targets[0])
        self.goal_idx = goal_node
        
        print(f"  Goal found at image index: {goal_node}")
        print(f"  Total exploration images: {len(self.image_files)}")
        
        # Trim actions to reach goal
        # Assuming 1 action per image (adjust if different sampling rate)
        if goal_node < len(self.actions):
            self.actions = self.actions[:goal_node + 10]  # Add buffer
            print(f"  Trimmed to {len(self.actions)} actions")
        else:
            print(f"  WARNING: Goal index {goal_node} beyond actions {len(self.actions)}")
        
        self.goal_action_index = len(self.actions) - 1
    
    def pre_navigation(self):
        """Prepare for navigation phase"""
        super(HybridAutonomousPlayer, self).pre_navigation()
        
        print("\n" + "="*70)
        print("PRE-NAVIGATION PHASE")
        print("="*70)
        
        # Load feature extractor and features
        self.load_feature_extractor()
        self.load_or_extract_features()
        self.build_index()
        
        print("\n" + "="*70)
        print("SYSTEM READY")
        print("="*70)
        
        if self.mode == "REPLAY":
            self.find_goal_in_actions()
            print(f"\n[AUTO MODE] Press any arrow key for manual control")
            print(f"[AUTO MODE] Press 'A' to resume autonomous navigation")
        else:
            # Save recorded actions
            with open(self.actions_file, 'w') as f:
                json.dump(self.actions, f)
            print(f"\n[SAVED] {len(self.actions)} actions to {self.actions_file}")
            print("Run again to replay autonomously!")
    
    def act(self):
        """Handle actions - autonomous replay or manual control"""
        user_input = None
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    self.last_act = Action.QUIT
                    return Action.QUIT
                
                # Toggle auto/manual with 'A' key
                elif event.key == pygame.K_a:
                    self.AUTO_NAV = not self.AUTO_NAV
                    mode_str = "AUTONOMOUS" if self.AUTO_NAV else "MANUAL"
                    print(f"\n[TOGGLE] Switched to {mode_str} mode")
                
                # Arrow keys for manual control
                elif event.key in self.keymap:
                    self.last_act |= self.keymap[event.key]
                    user_input = self.last_act
                    
                    # Any arrow key disables auto mode
                    if event.key in [pygame.K_LEFT, pygame.K_RIGHT, 
                                    pygame.K_UP, pygame.K_DOWN]:
                        if self.AUTO_NAV:
                            self.AUTO_NAV = False
                            print("\n[MANUAL OVERRIDE] Switched to manual control")
                            print("[MANUAL OVERRIDE] Press 'A' to resume autonomous")
                
                else:
                    self.show_target_images()
            
            if event.type == pygame.KEYUP:
                if event.key in self.keymap:
                    self.last_act ^= self.keymap[event.key]
        
        # Get current phase
        state = self.get_state()
        
        # EXPLORATION PHASE - Record actions
        if state and state[1] == Phase.EXPLORATION:
            if self.last_act != Action.IDLE and self.last_act != Action.QUIT:
                self.actions.append(self.last_act.value)
                if len(self.actions) % 50 == 0:
                    print(f"[RECORDING] {len(self.actions)} actions recorded")
            return self.last_act
        
        # NAVIGATION PHASE - Replay or manual control
        if state and state[1] == Phase.NAVIGATION:
            # Localization check
            if self.localize_counter % self.localize_interval == 0 and self.fpv is not None:
                current_node = self.find_closest_node(self.fpv)
                
                if self.goal_idx is not None:
                    distance = abs(current_node - self.goal_idx)
                    if distance <= 3:
                        print(f"\n[SUCCESS] Near goal! Distance: {distance}")
                        return Action.CHECKIN
            
            self.localize_counter += 1
            
            # AUTONOMOUS MODE - Replay actions
            if self.AUTO_NAV and self.replay_index < len(self.actions):
                action_value = self.actions[self.replay_index]
                self.replay_index += 1
                
                if self.replay_index % 20 == 0:
                    progress = (self.replay_index / len(self.actions)) * 100
                    print(f"[AUTO] Progress: {self.replay_index}/{len(self.actions)} ({progress:.1f}%)")
                
                return Action(action_value)
            
            # Replay complete
            elif self.AUTO_NAV and self.replay_index >= len(self.actions):
                print("\n[AUTO] Replay complete, checking in")
                return Action.CHECKIN
            
            # MANUAL MODE - User control
            else:
                return self.last_act
        
        return self.last_act
    
    def show_target_images(self):
        """Display target images in 2x2 grid"""
        targets = self.get_target_images()
        if targets is None or len(targets) == 0:
            return
        
        hor1 = cv2.hconcat(targets[:2])
        hor2 = cv2.hconcat(targets[2:])
        concat_img = cv2.vconcat([hor1, hor2])
        
        w, h = concat_img.shape[:2]
        color = (0, 0, 0)
        
        concat_img = cv2.line(concat_img, (int(h/2), 0), (int(h/2), w), color, 2)
        concat_img = cv2.line(concat_img, (0, int(w/2)), (h, int(w/2)), color, 2)
        
        w_offset = 25
        h_offset = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        line = cv2.LINE_AA
        size = 0.75
        stroke = 1
        
        cv2.putText(concat_img, 'Front View', (h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Right View', (int(h/2) + h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Back View', (h_offset, int(w/2) + w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Left View', (int(h/2) + h_offset, int(w/2) + w_offset), font, size, color, stroke, line)
        
        cv2.imshow('Target Images', concat_img)
        cv2.waitKey(1)
    
    def set_target_images(self, images):
        super(HybridAutonomousPlayer, self).set_target_images(images)
        self.show_target_images()
    
    def see(self, fpv):
        """Process and display current first-person view"""
        if fpv is None or len(fpv.shape) < 3:
            return
        
        self.fpv = fpv
        
        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))
        
        def convert_opencv_img_to_pygame(opencv_image):
            opencv_image = opencv_image[:, :, ::-1]
            shape = opencv_image.shape[1::-1]
            pygame_image = pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')
            return pygame_image
        
        # Update caption with mode
        mode_str = "AUTO" if self.AUTO_NAV else "MANUAL"
        pygame.display.set_caption(f"Hybrid Player - {mode_str} - {self.replay_index}/{len(self.actions)}")
        
        rgb = convert_opencv_img_to_pygame(fpv)
        self.screen.blit(rgb, (0, 0))
        pygame.display.update()


if __name__ == "__main__":
    import vis_nav_game
    print("\n" + "="*70)
    print("HYBRID AUTONOMOUS NAVIGATION")
    print("="*70)
    print("FIRST RUN: Navigate manually to explore (arrow keys)")
    print("  Actions will be recorded automatically")
    print("  Press ESC when done exploring")
    print("\nSECOND RUN: Autonomous navigation with manual override")
    print("  Robot will replay recorded actions automatically")
    print("  Press any arrow key if robot gets stuck")
    print("  Press 'A' to resume autonomous mode")
    print("="*70 + "\n")
    
    vis_nav_game.play(the_player=HybridAutonomousPlayer())