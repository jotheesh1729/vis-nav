"""
ResNet-based Localization
Next Best View display and goal detection
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import cv2
import pickle
from tqdm import tqdm
import logging
from config_demo import *

logging.basicConfig(level=logging.INFO)


class ResNetLocalizer:
    """
    ResNet-based place recognition for Next Best View
    """
    
    def __init__(self):
        """
        Initialize ResNet feature extractor
        """
        logging.info(f"Initializing {RESNET_MODEL}...")
        
        # Load model
        if RESNET_MODEL == 'resnet50':
            from torchvision.models import ResNet50_Weights
            self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        elif RESNET_MODEL == 'resnet18':
            from torchvision.models import ResNet18_Weights
            self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            # Fallback to old API
            self.model = models.resnet50(pretrained=True)
        
        # Remove classification layer
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() and USE_GPU else 'cpu')
        self.model.to(self.device)
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Database
        self.database_features = None
        self.database_files = None
        
        logging.info(f"ResNet ready on {self.device}")
    
    def extract_feature(self, image):
        """
        Extract feature from single image
        
        Args:
            image: numpy array (BGR) or PIL Image
            
        Returns:
            feature: normalized feature vector
        """
        # Convert to PIL
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                image = Image.fromarray(image)
        
        # Extract
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            feature = self.model(img_tensor)
        
        feature = feature.squeeze().cpu().numpy()
        feature = feature / (np.linalg.norm(feature) + 1e-6)
        
        return feature
    
    def load_database(self, image_dir=None, cache_file='demo_resnet_features.pkl'):
        """
        Load or extract database features
        
        Args:
            image_dir: directory with images
            cache_file: cache filename
        """
        if image_dir is None:
            image_dir = DATA_DIR
        
        cache_path = os.path.join(DEMO_DIR, cache_file)
        
        # Try loading cache
        if os.path.exists(cache_path):
            logging.info(f"Loading cached features from {cache_path}")
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            self.database_features = data['features']
            self.database_files = data['files']
            logging.info(f"Loaded {len(self.database_features)} features")
            return
        
        # Extract features
        if not os.path.exists(image_dir):
            logging.warning(f"Database directory not found: {image_dir}")
            return
        
        image_files = sorted([f for f in os.listdir(image_dir) 
                            if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        if len(image_files) == 0:
            logging.warning(f"No images found in {image_dir}")
            return
        
        logging.info(f"Extracting features from {len(image_files)} images...")
        
        features = []
        for img_file in tqdm(image_files, desc="ResNet features"):
            img_path = os.path.join(image_dir, img_file)
            img = Image.open(img_path).convert('RGB')
            feat = self.extract_feature(img)
            features.append(feat)
        
        self.database_features = np.array(features)
        self.database_files = image_files
        
        # Cache
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'features': self.database_features,
                'files': self.database_files
            }, f)
        
        logging.info(f"Cached {len(features)} features to {cache_path}")
    
    def find_most_similar(self, query_image, top_k=1):
        """
        Find most similar images in database
        
        Args:
            query_image: query image (numpy or PIL)
            top_k: number of results
            
        Returns:
            indices: indices of top matches
            similarities: similarity scores
        """
        if self.database_features is None:
            return None, None
        
        # Extract query feature
        query_feat = self.extract_feature(query_image)
        
        # Compute similarities
        similarities = np.dot(self.database_features, query_feat)
        
        # Get top k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_similarities = similarities[top_indices]
        
        return top_indices, top_similarities
    
    def display_next_best_view(self, current_frame, window_name="Next Best View"):
        """
        Display most similar image from database
        
        Args:
            current_frame: current camera frame
            window_name: OpenCV window name
        """
        if self.database_features is None or self.database_files is None:
            return
        
        # Find similar
        indices, similarities = self.find_most_similar(current_frame, top_k=1)
        
        if indices is None:
            return
        
        best_idx = indices[0]
        similarity = similarities[0]
        
        # Load image
        img_path = os.path.join(DATA_DIR, self.database_files[best_idx])
        if not os.path.exists(img_path):
            return
        
        img = cv2.imread(img_path)
        
        # Add text
        text = f"Most Similar: #{best_idx} (similarity: {similarity:.3f})"
        cv2.putText(img, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Force window creation
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, img)
        cv2.waitKey(1)
    
    def find_goal_frame(self, target_images, recorded_frames):
        """
        Find which recorded frame is closest to goal
        
        Args:
            target_images: list of 4 target views
            recorded_frames: list of recorded frame images
            
        Returns:
            goal_frame_idx: index of goal frame
        """
        if len(recorded_frames) == 0:
            return 0
        
        # Extract features from targets
        target_features = []
        for target in target_images:
            if target is not None:
                feat = self.extract_feature(target)
                target_features.append(feat)
        
        if len(target_features) == 0:
            return len(recorded_frames) - 1
        
        # Use front view (first target)
        target_feat = target_features[0]
        
        # Extract features from recorded frames (subsample for speed)
        step = max(1, len(recorded_frames) // 100)  # Check every ~1% of frames
        
        best_similarity = -1
        best_idx = len(recorded_frames) - 1
        
        logging.info("Finding goal frame in recording...")
        for i in tqdm(range(0, len(recorded_frames), step)):
            frame_feat = self.extract_feature(recorded_frames[i])
            similarity = np.dot(target_feat, frame_feat)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_idx = i
        
        logging.info(f"Goal frame found at index {best_idx} (similarity: {best_similarity:.3f})")
        
        return best_idx


if __name__ == "__main__":
    # Test
    localizer = ResNetLocalizer()
    
    if os.path.exists(DATA_DIR):
        localizer.load_database()
        logging.info("ResNet localizer test complete")