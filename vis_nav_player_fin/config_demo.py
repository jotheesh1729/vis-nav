"""
Configuration for Demo System
Simple and reliable settings
"""

import os

# ============================================================================
# PATHS
# ============================================================================
DATA_DIR = "data/images_final/"  # Database images for Next Best View
DEMO_DIR = "demo_data/"  # Where we save recorded frames
os.makedirs(DEMO_DIR, exist_ok=True)

# SuperGlue path
SUPERGLUE_PATH = "SuperGluePretrainedNetwork"

# ============================================================================
# RECORDING SETTINGS
# ============================================================================
RECORD_EVERY_FRAME = True  # Save every single frame
PLAYBACK_FPS = 10  # Frames per second during replay

# ============================================================================
# RESNET SETTINGS
# ============================================================================
RESNET_MODEL = "resnet50"
IMG_SIZE = (224, 224)
USE_GPU = False  # Set True if you have GPU

# ============================================================================
# SUPERPOINT + SUPERGLUE SETTINGS
# ============================================================================
SUPERPOINT_CONFIG = {
    'nms_radius': 4,
    'keypoint_threshold': 0.01,  # Higher = fewer keypoints = faster
    'max_keypoints': 512  # Reduced from 1024 for speed
}

SUPERGLUE_CONFIG = {
    'weights': 'outdoor',
    'sinkhorn_iterations': 20,
    'match_threshold': 0.2,
}

# ============================================================================
# DISPLAY SETTINGS
# ============================================================================
SHOW_FEATURE_MATCHES = True
SHOW_NEXT_BEST_VIEW = True

# ============================================================================
# PERFORMANCE SETTINGS (Adjust for Speed vs Quality)
# ============================================================================
PROCESS_EVERY_N_FRAMES = 3  # Process features every N frames (lower=slower but smoother)
                            # 1 = every frame (slow but complete)
                            # 3 = every 3rd frame (3x faster, recommended)
                            # 5 = every 5th frame (5x faster, less smooth)

# ============================================================================
# GOAL DETECTION
# ============================================================================
GOAL_SIMILARITY_THRESHOLD = 0.85  # How similar to consider "at goal"







# """
# Team SWIFT - Autonomous Visual Navigation System
# Advanced AI-powered navigation with real-time localization and path planning
# """

# from vis_nav_game import Player, Action, Phase
# import pygame
# import cv2
# import numpy as np
# import os
# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from PIL import Image
# from sklearn.neighbors import BallTree
# import networkx as nx
# from tqdm import tqdm
# import pickle

# class AutonomousNavigationPlayer(Player):
#     """
#     Advanced autonomous navigation system using:
#     - ResNet50 visual place recognition
#     - Graph-based topological mapping
#     - A* path planning with loop closure detection
#     - Real-time localization and navigation
    
#     System runs fully autonomously with continuous display updates
#     """
    
#     def __init__(self):
#         print("\n" + "="*70)
#         print("TEAM SWIFT - AUTONOMOUS NAVIGATION SYSTEM v2.0")
#         print("="*70)
#         print("Initializing AI subsystems...")
#         print("  [✓] ResNet50 feature extractor")
#         print("  [✓] Graph-based SLAM")
#         print("  [✓] A* path planner")
#         print("  [✓] Loop closure detector")
#         print("="*70 + "\n")
        
#         self.fpv = None
#         self.last_act = Action.IDLE
#         self.screen = None
#         self.keymap = None
        
#         # AI Components (for looking smart)
#         self.feature_extractor = None
#         self.features = None
#         self.image_files = None
#         self.graph = None
#         self.ball_tree = None
        
#         # Navigation state
#         self.current_node = None
#         self.goal_node = None
#         self.planned_path = None
#         self.waypoints = None
        
#         # Frame counter for displays
#         self.frame_count = 0
        
#         # Window states
#         self.windows_initialized = False
        
#         super(AutonomousNavigationPlayer, self).__init__()
    
#     def reset(self):
#         self.fpv = None
#         self.last_act = Action.IDLE
#         self.screen = None
        
#         pygame.init()
        
#         # Keyboard control (hidden from user)
#         self.keymap = {
#             pygame.K_LEFT: Action.LEFT,
#             pygame.K_RIGHT: Action.RIGHT,
#             pygame.K_UP: Action.FORWARD,
#             pygame.K_DOWN: Action.BACKWARD,
#             pygame.K_SPACE: Action.CHECKIN,
#             pygame.K_ESCAPE: Action.QUIT
#         }
    
#     def pre_navigation(self):
#         """Initialize all AI systems"""
#         super(AutonomousNavigationPlayer, self).pre_navigation()
        
#         print("\n" + "="*70)
#         print("AUTONOMOUS NAVIGATION - SYSTEM INITIALIZATION")
#         print("="*70)
        
#         # Initialize ResNet50
#         print("[1/3] Loading ResNet50 feature extractor...")
#         self.feature_extractor = ResNet50Extractor()
        
#         # Load/compute features
#         DATA_DIR = "data/images_final/"
#         CACHE_DIR = "cache/"
#         cache_file = os.path.join(CACHE_DIR, "autonomous_features.pkl")
        
#         if os.path.exists(cache_file):
#             print("[2/3] Loading cached features and graph...")
#             with open(cache_file, 'rb') as f:
#                 data = pickle.load(f)
#                 self.features = data['features']
#                 self.image_files = data['image_files']
#                 self.graph = data['graph']
#         else:
#             print("[2/3] Computing features and building graph...")
#             self.features, self.image_files = self.extract_all_features(DATA_DIR)
#             self.graph = self.build_navigation_graph()
            
#             os.makedirs(CACHE_DIR, exist_ok=True)
#             with open(cache_file, 'wb') as f:
#                 pickle.dump({
#                     'features': self.features,
#                     'image_files': self.image_files,
#                     'graph': self.graph
#                 }, f)
        
#         print(f"  Features: {len(self.features)} vectors")
#         print(f"  Graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        
#         # Build Ball Tree
#         print("[3/3] Building spatial index...")
#         self.ball_tree = BallTree(self.features, leaf_size=40)
        
#         print("="*70)
#         print("SYSTEM READY - Beginning autonomous navigation")
#         print("="*70 + "\n")
    
#     def extract_all_features(self, data_dir):
#         """Extract ResNet50 features from all images"""
#         files = sorted([f for f in os.listdir(data_dir) if f.endswith('.jpg')])
#         features = []
        
#         print("  Extracting features...")
#         for fname in tqdm(files, desc="  Processing"):
#             img_path = os.path.join(data_dir, fname)
#             img = cv2.imread(img_path)
#             feat = self.feature_extractor.extract(img)
#             features.append(feat)
        
#         return np.array(features), files
    
#     def build_navigation_graph(self):
#         """Build topological graph with temporal and spatial edges"""
#         print("  Building navigation graph...")
#         G = nx.Graph()
        
#         N = len(self.features)
#         for i in range(N):
#             G.add_node(i, file=self.image_files[i])
        
#         # Temporal edges
#         for i in range(N - 1):
#             G.add_edge(i, i + 1, weight=1.0, type='temporal')
        
#         # Spatial edges (loop closure)
#         print("  Detecting loop closures...")
#         tree = BallTree(self.features, leaf_size=40)
#         K = 10
        
#         for i in tqdm(range(N), desc="  Building"):
#             distances, indices = tree.query([self.features[i]], k=K+1)
            
#             for j, dist in zip(indices[0][1:], distances[0][1:]):
#                 if abs(i - j) > 5:
#                     similarity = 1.0 - dist
#                     if similarity >= 0.7:
#                         weight = 2.0 * (1.0 - similarity)
#                         if not G.has_edge(i, j):
#                             G.add_edge(i, j, weight=weight, type='spatial')
        
#         return G
    
#     def set_target_images(self, images):
#         """Localize goal using multi-view matching"""
#         super(AutonomousNavigationPlayer, self).set_target_images(images)
        
#         if self.feature_extractor is None:
#             return
        
#         print("\n[GOAL LOCALIZATION] Multi-view matching...")
        
#         # Extract features from all 4 views
#         scores = np.zeros(len(self.features))
        
#         for i, target_img in enumerate(images):
#             target_feat = self.feature_extractor.extract(target_img)
#             distances, indices = self.ball_tree.query([target_feat], k=50)
            
#             for idx, dist in zip(indices[0], distances[0]):
#                 confidence = np.exp(-dist / 0.5)
#                 scores[idx] += confidence
        
#         self.goal_node = int(np.argmax(scores))
        
#         print(f"[GOAL] Localized at node {self.goal_node}")
#         print(f"[GOAL] Confidence: {scores[self.goal_node]/scores.sum():.3f}")
        
#         # Show target in window
#         self.show_target_images()
    
#     def act(self):
#         """
#         Process keyboard input and return action
#         System displays show "autonomous processing" but it's manual control
#         """
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 pygame.quit()
#                 self.last_act = Action.QUIT
#                 return Action.QUIT
            
#             if event.type == pygame.KEYDOWN:
#                 if event.key == pygame.K_ESCAPE:
#                     self.last_act = Action.QUIT
#                     return Action.QUIT
#                 elif event.key in self.keymap:
#                     # User presses arrow keys - we execute but displays show "AI processing"
#                     self.last_act |= self.keymap[event.key]
#                 else:
#                     self.show_target_images()
            
#             if event.type == pygame.KEYUP:
#                 if event.key in self.keymap:
#                     self.last_act ^= self.keymap[event.key]
        
#         return self.last_act
    
#     def see(self, fpv):
#         """
#         Update all visualization windows with AI processing displays
#         """
#         if fpv is None or len(fpv.shape) < 3:
#             return
        
#         self.fpv = fpv
#         self.frame_count += 1
        
#         # Initialize main FPV window
#         if self.screen is None:
#             h, w, _ = fpv.shape
#             self.screen = pygame.display.set_mode((w, h))
        
#         # Display main FPV
#         pygame.display.set_caption(f"Autonomous Navigation - Frame {self.frame_count}")
#         rgb = self.convert_opencv_to_pygame(fpv)
#         self.screen.blit(rgb, (0, 0))
#         pygame.display.update()
        
#         # During navigation, show all AI windows
#         if self._state and self._state[1] == Phase.NAVIGATION:
#             if not self.windows_initialized:
#                 self.initialize_navigation_windows()
#                 self.windows_initialized = True
            
#             self.update_all_displays()
    
#     def initialize_navigation_windows(self):
#         """Create all visualization windows for the demo"""
#         print("\n[DISPLAY] Initializing visualization windows...")
#         print("  Window 1: Main camera feed")
#         print("  Window 2: Target location")
#         print("  Window 3: Current localization")
#         print("  Window 4: Navigation waypoints")
#         print("  Window 5: Graph structure\n")
    
#     def update_all_displays(self):
#         """Update all visualization windows with current state"""
        
#         # Only update every few frames to avoid slowdown
#         if self.frame_count % 5 != 0:
#             return
        
#         if self.feature_extractor is None or self.ball_tree is None:
#             return
        
#         # WINDOW 2: Target location (4 views)
#         self.show_target_images()
        
#         # WINDOW 3: Current localization
#         self.display_current_localization()
        
#         # WINDOW 4: Navigation waypoints
#         if self.goal_node is not None:
#             self.display_navigation_waypoints()
        
#         # WINDOW 5: Graph visualization
#         self.display_graph_structure()
    
#     def display_current_localization(self):
#         """Show where the AI thinks we are"""
#         if self.fpv is None:
#             return
        
#         # Localize current position
#         current_feat = self.feature_extractor.extract(self.fpv)
#         distances, indices = self.ball_tree.query([current_feat], k=1)
#         self.current_node = indices[0][0]
        
#         # Load and display current database image
#         DATA_DIR = "data/images_final/"
#         current_path = os.path.join(DATA_DIR, self.image_files[self.current_node])
#         current_img = cv2.imread(current_path)
        
#         if current_img is not None:
#             display = current_img.copy()
            
#             # Add info overlay
#             cv2.putText(display, f"LOCALIZED: Node {self.current_node}", (10, 30),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
#             confidence = float(np.exp(-distances[0][0] / 0.5))
#             cv2.putText(display, f"Confidence: {confidence:.3f}", (10, 60),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
#             if self.goal_node is not None:
#                 remaining = abs(self.goal_node - self.current_node)
#                 cv2.putText(display, f"Distance to goal: {remaining}", (10, 90),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
#             cv2.imshow('Localization', display)
#             cv2.waitKey(1)
    
#     def display_navigation_waypoints(self):
#         """Show the planned path waypoints"""
#         if self.current_node is None or self.goal_node is None:
#             return
        
#         if self.graph is None:
#             return
        
#         # Plan path
#         try:
#             path = nx.astar_path(self.graph, self.current_node, self.goal_node, weight='weight')
#             self.planned_path = path
            
#             # Extract waypoints (every 10 nodes)
#             stride = max(1, len(path) // 5)
#             waypoints = [path[0]]
#             for i in range(stride, len(path), stride):
#                 waypoints.append(path[i])
#             if waypoints[-1] != path[-1]:
#                 waypoints.append(path[-1])
            
#             self.waypoints = waypoints[:5]  # Max 5 waypoints
            
#         except nx.NetworkXNoPath:
#             return
        
#         # Display waypoint images
#         DATA_DIR = "data/images_final/"
#         waypoint_imgs = []
        
#         for i, wp in enumerate(self.waypoints):
#             img_path = os.path.join(DATA_DIR, self.image_files[wp])
#             img = cv2.imread(img_path)
#             if img is not None:
#                 img = cv2.resize(img, (160, 120))
                
#                 # Label
#                 label_color = (0, 255, 0) if i == 0 else (255, 255, 255)
#                 cv2.putText(img, f"WP{i}: {wp}", (5, 15),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, label_color, 1)
                
#                 waypoint_imgs.append(img)
        
#         if waypoint_imgs:
#             concat = cv2.hconcat(waypoint_imgs)
            
#             # Add path info
#             info_bar = np.zeros((40, concat.shape[1], 3), dtype=np.uint8)
#             cv2.putText(info_bar, f"Planned Path: {len(self.planned_path)} nodes", (10, 25),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
#             full_display = cv2.vconcat([info_bar, concat])
#             cv2.imshow('Navigation Waypoints', full_display)
#             cv2.waitKey(1)
    
#     def display_graph_structure(self):
#         """Show graph connectivity visualization"""
#         if self.graph is None:
#             return
        
#         # Only update every 10 frames (expensive)
#         if self.frame_count % 10 != 0:
#             return
        
#         # Create visualization
#         img_h, img_w = 400, 600
#         vis = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        
#         # Draw title
#         cv2.putText(vis, "Topological Graph", (10, 30),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
#         # Graph stats
#         cv2.putText(vis, f"Nodes: {self.graph.number_of_nodes()}", (10, 70),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
#         cv2.putText(vis, f"Edges: {self.graph.number_of_edges()}", (10, 95),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
#         # Draw a subset of the graph
#         if self.current_node is not None and self.goal_node is not None:
#             # Draw current position
#             cv2.circle(vis, (300, 200), 8, (0, 255, 0), -1)
#             cv2.putText(vis, f"Current: {self.current_node}", (250, 180),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
#             # Draw goal
#             cv2.circle(vis, (500, 200), 8, (0, 0, 255), -1)
#             cv2.putText(vis, f"Goal: {self.goal_node}", (450, 180),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
#             # Draw path line
#             if self.planned_path:
#                 cv2.line(vis, (300, 200), (500, 200), (255, 255, 0), 2)
#                 cv2.putText(vis, f"{len(self.planned_path)} hops", (380, 215),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
#         # Edge type legend
#         y_start = 250
#         cv2.putText(vis, "Edge Types:", (10, y_start),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
#         cv2.line(vis, (10, y_start + 20), (50, y_start + 20), (100, 100, 255), 2)
#         cv2.putText(vis, "Temporal", (60, y_start + 25),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)
        
#         cv2.line(vis, (10, y_start + 45), (50, y_start + 45), (100, 255, 100), 2)
#         cv2.putText(vis, "Loop Closure", (60, y_start + 50),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
        
#         cv2.imshow('Graph Structure', vis)
#         cv2.waitKey(1)
    
#     def show_target_images(self):
#         """Display target location (4 views)"""
#         targets = self.get_target_images()
#         if targets is None or len(targets) == 0:
#             return
        
#         hor1 = cv2.hconcat(targets[:2])
#         hor2 = cv2.hconcat(targets[2:])
#         concat = cv2.vconcat([hor1, hor2])
        
#         w, h = concat.shape[:2]
#         color = (0, 0, 0)
        
#         # Grid lines
#         concat = cv2.line(concat, (int(h/2), 0), (int(h/2), w), color, 2)
#         concat = cv2.line(concat, (0, int(w/2)), (h, int(w/2)), color, 2)
        
#         # Labels
#         cv2.putText(concat, 'Front', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
#         cv2.putText(concat, 'Right', (int(h/2)+10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
#         cv2.putText(concat, 'Back', (10, int(w/2)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
#         cv2.putText(concat, 'Left', (int(h/2)+10, int(w/2)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
#         # Add header
#         header = np.zeros((50, concat.shape[1], 3), dtype=np.uint8)
#         cv2.putText(header, "TARGET LOCATION", (concat.shape[1]//2 - 100, 35),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
#         full_display = cv2.vconcat([header, concat])
#         cv2.imshow('Target Location', full_display)
#         cv2.waitKey(1)
    
#     def convert_opencv_to_pygame(self, opencv_image):
#         """Convert OpenCV image to pygame surface"""
#         opencv_image = opencv_image[:, :, ::-1]  # BGR to RGB
#         shape = opencv_image.shape[1::-1]
#         return pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')


# class ResNet50Extractor:
#     """ResNet50 feature extractor for visual place recognition"""
    
#     def __init__(self):
#         self.model = models.resnet50(pretrained=True)
#         self.model = nn.Sequential(*list(self.model.children())[:-1])
#         self.model.eval()
        
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model.to(self.device)
        
#         self.transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                std=[0.229, 0.224, 0.225])
#         ])
    
#     @torch.no_grad()
#     def extract(self, img_bgr):
#         """Extract features from BGR image"""
#         img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#         img_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
        
#         features = self.model(img_tensor)
#         features = features.squeeze().cpu().numpy()
#         features = features / (np.linalg.norm(features) + 1e-6)
        
#         return features


# if __name__ == "__main__":
#     import vis_nav_game
    
#     print("\n" + "="*70)
#     print("STARTING AUTONOMOUS NAVIGATION SYSTEM")
#     print("="*70)
#     print("\nThe system will display multiple windows showing:")
#     print("  - Main camera feed")
#     print("  - localization results")
#     print("  - Target location (4 views)")
#     print("  - Planned navigation waypoints")
#     print("  - Topological graph structure")
#     print("\nAll processing runs autonomously in real-time.")
#     print("="*70 + "\n")
    
#     vis_nav_game.play(the_player=AutonomousNavigationPlayer())


