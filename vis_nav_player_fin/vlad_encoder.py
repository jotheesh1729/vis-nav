"""
SIFT + VLAD encoding for place recognition
"""

import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
from tqdm import tqdm
import pickle
import logging
from config import *

logging.basicConfig(level=logging.INFO)

class VLADEncoder:
    """
    VLAD (Vector of Locally Aggregated Descriptors) encoder
    Uses SIFT features + K-means clustering
    """
    
    def __init__(self):
        """
        Initialize VLAD encoder
        """
        # Initialize SIFT detector
        self.sift = cv2.SIFT_create()
        
        # Storage
        self.codebook = None
        self.database = None
        self.tree = None
        
        logging.info("VLAD encoder initialized")
    
    def compute_sift_features(self, images):
        """
        Compute SIFT descriptors for all images
        
        Args:
            images: list of numpy arrays (BGR images)
            
        Returns:
            sift_descriptors: numpy array of all SIFT descriptors
        """
        logging.info("Computing SIFT features...")
        sift_descriptors = []
        
        for img in tqdm(images, desc="SIFT extraction"):
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
            # Detect and compute
            _, descriptors = self.sift.detectAndCompute(gray, None)
            
            if descriptors is not None:
                sift_descriptors.extend(descriptors)
        
        sift_descriptors = np.array(sift_descriptors)
        logging.info(f"Extracted {len(sift_descriptors)} SIFT descriptors")
        
        return sift_descriptors
    
    def build_codebook(self, descriptors, n_clusters=None, cache_file='codebook.pkl'):
        """
        Build visual vocabulary using K-means clustering
        
        Args:
            descriptors: SIFT descriptors (N, 128)
            n_clusters: number of clusters (visual words)
            cache_file: cache filename
        """
        if n_clusters is None:
            n_clusters = VLAD_CLUSTERS
        
        cache_path = os.path.join(CACHE_DIR, cache_file)
        
        # Check if cached
        if os.path.exists(cache_path):
            logging.info(f"Loading codebook from {cache_path}")
            with open(cache_path, 'rb') as f:
                self.codebook = pickle.load(f)
            return
        
        logging.info(f"Building codebook with {n_clusters} clusters...")
        self.codebook = KMeans(
            n_clusters=n_clusters,
            init=KMEANS_INIT,
            n_init=KMEANS_N_INIT,
            verbose=0,
            random_state=42
        ).fit(descriptors)
        
        # Cache
        with open(cache_path, 'wb') as f:
            pickle.dump(self.codebook, f)
        logging.info(f"Codebook saved to {cache_path}")
    
    def encode_vlad(self, image):
        """
        Compute VLAD descriptor for a single image
        
        Args:
            image: numpy array (BGR or grayscale)
            
        Returns:
            vlad_feature: VLAD descriptor (flattened vector)
        """
        if self.codebook is None:
            raise ValueError("Codebook not built. Call build_codebook() first.")
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Extract SIFT descriptors
        _, descriptors = self.sift.detectAndCompute(gray, None)
        
        if descriptors is None or len(descriptors) == 0:
            # Return zero vector if no features
            k = self.codebook.n_clusters
            return np.zeros(k * 128)
        
        # Predict cluster labels
        labels = self.codebook.predict(descriptors)
        
        # Get cluster centers
        centroids = self.codebook.cluster_centers_
        k = self.codebook.n_clusters
        
        # Initialize VLAD feature
        vlad_feature = np.zeros([k, descriptors.shape[1]])
        
        # Compute residuals for each cluster
        for i in range(k):
            if np.sum(labels == i) > 0:
                # Sum of residuals (descriptor - centroid) for this cluster
                vlad_feature[i] = np.sum(
                    descriptors[labels == i, :] - centroids[i],
                    axis=0
                )
        
        # Flatten
        vlad_feature = vlad_feature.flatten()
        
        # Power normalization
        vlad_feature = np.sign(vlad_feature) * np.sqrt(np.abs(vlad_feature))
        
        # L2 normalization
        vlad_feature = vlad_feature / (np.linalg.norm(vlad_feature) + 1e-12)
        
        return vlad_feature
    
    def build_database(self, images, cache_file='vlad_database.pkl'):
        """
        Build VLAD database for all images
        
        Args:
            images: list of numpy arrays
            cache_file: cache filename
            
        Returns:
            database: numpy array of VLAD descriptors (N, D)
        """
        cache_path = os.path.join(CACHE_DIR, cache_file)
        
        # Check if cached
        if os.path.exists(cache_path):
            logging.info(f"Loading VLAD database from {cache_path}")
            with open(cache_path, 'rb') as f:
                self.database = pickle.load(f)
            return self.database
        
        logging.info("Building VLAD database...")
        self.database = []
        
        for img in tqdm(images, desc="VLAD encoding"):
            vlad = self.encode_vlad(img)
            self.database.append(vlad)
        
        self.database = np.array(self.database)
        logging.info(f"VLAD database shape: {self.database.shape}")
        
        # Cache
        with open(cache_path, 'wb') as f:
            pickle.dump(self.database, f)
        logging.info(f"VLAD database saved to {cache_path}")
        
        return self.database
    
    def build_tree(self, leaf_size=40):
        """
        Build BallTree for fast nearest neighbor search
        
        Args:
            leaf_size: BallTree leaf size parameter
        """
        if self.database is None:
            raise ValueError("Database not built. Call build_database() first.")
        
        logging.info("Building BallTree for nearest neighbor search...")
        self.tree = BallTree(self.database, leaf_size=leaf_size)
        logging.info("BallTree built successfully")
    
    def find_nearest(self, query_image, k=1):
        """
        Find nearest neighbors in database
        
        Args:
            query_image: numpy array (image)
            k: number of neighbors
            
        Returns:
            indices: indices of nearest neighbors
            distances: distances to nearest neighbors
        """
        if self.tree is None:
            raise ValueError("Tree not built. Call build_tree() first.")
        
        # Encode query
        query_vlad = self.encode_vlad(query_image).reshape(1, -1)
        
        # Query tree
        distances, indices = self.tree.query(query_vlad, k=k)
        
        return indices[0], distances[0]
    
    def setup_from_images(self, images):
        """
        Complete setup: SIFT -> codebook -> VLAD database -> tree
        
        Args:
            images: list of numpy arrays
        """
        # Compute SIFT features
        sift_descriptors = self.compute_sift_features(images)
        
        # Build codebook
        self.build_codebook(sift_descriptors)
        
        # Build VLAD database
        self.build_database(images)
        
        # Build search tree
        self.build_tree()
        
        logging.info("VLAD encoder setup complete")


if __name__ == "__main__":
    # Test VLAD encoding
    logging.info("Testing VLAD encoder...")
    
    encoder = VLADEncoder()
    
    # Load some test images
    if os.path.exists(DATA_DIR):
        test_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.jpg')])[:100]
        test_images = [cv2.imread(os.path.join(DATA_DIR, f)) for f in test_files]
        
        encoder.setup_from_images(test_images)
        
        # Test query
        query_img = test_images[50]
        indices, distances = encoder.find_nearest(query_img, k=5)
        logging.info(f"Nearest neighbors: {indices}")
        logging.info(f"Distances: {distances}")
    else:
        logging.warning(f"Data directory {DATA_DIR} not found")