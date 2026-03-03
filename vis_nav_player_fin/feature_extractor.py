"""
ResNet-based feature extraction for place recognition
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import pickle
import logging
from config import *

logging.basicConfig(level=logging.INFO)

class ResNetExtractor:
    """
    ResNet50 feature extractor for global image features
    """
    
    def __init__(self, model_name='resnet50'):
        """
        Initialize ResNet feature extractor
        
        Args:
            model_name: 'resnet50' or 'resnet18'
        """
        logging.info(f"Initializing {model_name} feature extractor...")
        
        # Load pretrained model
        if model_name == 'resnet50':
            from torchvision.models import ResNet50_Weights
            self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        elif model_name == 'resnet18':
            from torchvision.models import ResNet18_Weights
            self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Remove final classification layer
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        # Set to evaluation mode
        self.model.eval()
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() and USE_GPU else 'cpu')
        self.model.to(self.device)
        logging.info(f"Using device: {self.device}")
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def extract_single(self, image):
        """
        Extract features from a single image
        
        Args:
            image: numpy array (H, W, C) BGR or PIL Image
            
        Returns:
            features: numpy array of shape (FEATURE_DIM,)
        """
        # Convert to PIL if numpy
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = Image.fromarray(image[:, :, ::-1])
            else:
                image = Image.fromarray(image)
        
        # Preprocess
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(img_tensor)
        
        # Flatten and convert to numpy
        features = features.squeeze().cpu().numpy()
        
        # L2 normalize
        features = features / (np.linalg.norm(features) + 1e-6)
        
        return features
    
    def extract_from_path(self, image_path):
        """
        Extract features from image file
        
        Args:
            image_path: path to image file
            
        Returns:
            features: numpy array
        """
        image = Image.open(image_path).convert('RGB')
        return self.extract_single(image)
    
    def extract_database(self, image_dir=None, image_list=None, cache_file='resnet_features.pkl'):
        """
        Extract features for all images in directory or from list
        
        Args:
            image_dir: directory containing images (if not using image_list)
            image_list: list of image arrays (if not using image_dir)
            cache_file: cache filename
            
        Returns:
            features: numpy array of shape (N, FEATURE_DIM)
            image_files: list of filenames (empty if using image_list)
        """
        cache_path = os.path.join(CACHE_DIR, cache_file)
        
        # Check if cached
        if os.path.exists(cache_path):
            logging.info(f"Loading cached features from {cache_path}")
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            return data['features'], data.get('image_files', [])
        
        features = []
        image_files = []
        
        # Extract from directory
        if image_dir is not None:
            image_files = sorted([f for f in os.listdir(image_dir) 
                                if f.endswith(('.png', '.jpg', '.jpeg'))])
            logging.info(f"Found {len(image_files)} images in {image_dir}")
            
            logging.info("Extracting ResNet features...")
            for img_file in tqdm(image_files, desc="Processing"):
                img_path = os.path.join(image_dir, img_file)
                feat = self.extract_from_path(img_path)
                features.append(feat)
        
        # Extract from image list
        elif image_list is not None:
            logging.info(f"Extracting features from {len(image_list)} images...")
            for img in tqdm(image_list, desc="Processing"):
                feat = self.extract_single(img)
                features.append(feat)
        
        else:
            raise ValueError("Must provide either image_dir or image_list")
        
        features = np.array(features)
        logging.info(f"Features shape: {features.shape}")
        
        # Cache results
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'features': features,
                'image_files': image_files
            }, f)
        logging.info(f"Cached features to {cache_path}")
        
        return features, image_files
    
    def find_most_similar(self, query_feature, database_features, top_k=1):
        """
        Find most similar features in database
        
        Args:
            query_feature: feature vector (FEATURE_DIM,)
            database_features: database of features (N, FEATURE_DIM)
            top_k: number of top matches to return
            
        Returns:
            indices: indices of top matches
            similarities: similarity scores
        """
        # Compute cosine similarities
        similarities = np.dot(database_features, query_feature)
        
        # Get top k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_similarities = similarities[top_indices]
        
        return top_indices, top_similarities


if __name__ == "__main__":
    # Test feature extraction
    extractor = ResNetExtractor(RESNET_MODEL)
    
    # Test on exploration data if available
    if os.path.exists(DATA_DIR):
        features, files = extractor.extract_database(DATA_DIR)
        logging.info(f"Extracted {len(features)} feature vectors of dimension {features.shape[1]}")
    else:
        logging.warning(f"Data directory {DATA_DIR} not found")