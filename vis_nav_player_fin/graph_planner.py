"""
Graph-based navigation planning with A* pathfinding
"""

import numpy as np
import networkx as nx
import pickle
import os
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import logging
from config import *

logging.basicConfig(level=logging.INFO)


class NavigationGraph:
    """
    Build navigation graph from exploration data and plan paths
    """
    
    def __init__(self):
        """
        Initialize navigation graph
        """
        self.graph = None
        self.features = None
        self.actions = None
        
        logging.info("Navigation graph initialized")
    
    def build_graph(self, features, actions=None, cache_file='navigation_graph.pkl'):
        """
        Build weighted navigation graph
        
        Args:
            features: ResNet features (N, D)
            actions: action sequence (optional, for action retrieval)
            cache_file: cache filename
            
        Returns:
            graph: NetworkX graph
        """
        cache_path = os.path.join(CACHE_DIR, cache_file)
        
        # Check if cached
        if os.path.exists(cache_path):
            logging.info(f"Loading cached graph from {cache_path}")
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
                self.graph = data['graph']
                self.features = data['features']
                self.actions = data.get('actions', None)
            return self.graph
        
        logging.info("Building navigation graph...")
        
        self.features = features
        self.actions = actions
        n_images = len(features)
        
        # Initialize graph
        self.graph = nx.Graph()
        
        # Add nodes
        for i in range(n_images):
            self.graph.add_node(i)
        
        # Add temporal edges (consecutive frames)
        logging.info("Adding temporal edges...")
        for i in tqdm(range(n_images - 1), desc="Temporal edges"):
            weight = TEMPORAL_EDGE_WEIGHT
            self.graph.add_edge(i, i + 1, weight=weight, edge_type='temporal')
        
        # Add spatial edges (similar features)
        logging.info("Adding spatial edges...")
        self._add_spatial_edges(features)
        
        # Cache graph
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'graph': self.graph,
                'features': self.features,
                'actions': self.actions
            }, f)
        logging.info(f"Graph cached to {cache_path}")
        
        logging.info(f"Graph built: {self.graph.number_of_nodes()} nodes, "
                    f"{self.graph.number_of_edges()} edges")
        
        return self.graph
    
    def _add_spatial_edges(self, features):
        """
        Add spatial edges based on feature similarity
        
        Args:
            features: ResNet features (N, D)
        """
        n_images = len(features)
        
        # Compute pairwise similarities
        logging.info("Computing feature similarities...")
        similarities = cosine_similarity(features)
        
        # Add edges for each node
        logging.info("Finding neighbors and adding edges...")
        for i in tqdm(range(n_images), desc="Spatial edges"):
            # Get similarity scores for this node
            sim_scores = similarities[i]
            
            # Find top K neighbors (excluding self)
            top_k_indices = np.argsort(sim_scores)[::-1][1:K_NEIGHBORS + 1]
            
            for neighbor_idx in top_k_indices:
                similarity = sim_scores[neighbor_idx]
                
                # Only add if similarity exceeds threshold
                if similarity < SIMILARITY_THRESHOLD:
                    continue
                
                # Don't connect nearby temporal frames
                if abs(i - neighbor_idx) <= TEMPORAL_WINDOW:
                    continue
                
                # Edge weight: lower similarity = higher weight
                weight = SPATIAL_EDGE_WEIGHT_MULT * (1.0 - similarity)
                
                # Add edge (only if not already exists)
                if not self.graph.has_edge(i, neighbor_idx):
                    self.graph.add_edge(i, neighbor_idx, weight=weight, edge_type='spatial')
    
    def find_path(self, start_idx, goal_idx):
        """
        Find shortest path using A* algorithm
        
        Args:
            start_idx: starting node index
            goal_idx: goal node index
            
        Returns:
            path: list of node indices (or None if no path found)
        """
        if self.graph is None:
            raise ValueError("Graph not built. Call build_graph() first.")
        
        try:
            # A* with euclidean distance heuristic in feature space
            path = nx.astar_path(
                self.graph,
                start_idx,
                goal_idx,
                heuristic=lambda u, v: self._heuristic(u, v),
                weight='weight'
            )
            
            logging.info(f"Path found: {len(path)} steps from {start_idx} to {goal_idx}")
            return path
            
        except nx.NetworkXNoPath:
            logging.warning(f"No path found from {start_idx} to {goal_idx}")
            return None
        except Exception as e:
            logging.error(f"Path planning failed: {e}")
            return None
    
    def _heuristic(self, node1, node2):
        """
        A* heuristic: Euclidean distance in feature space
        
        Args:
            node1: first node index
            node2: second node index
            
        Returns:
            heuristic cost
        """
        if self.features is None:
            return 0
        
        feat1 = self.features[node1]
        feat2 = self.features[node2]
        dist = np.linalg.norm(feat1 - feat2)
        
        return HEURISTIC_WEIGHT * dist
    
    def get_action_sequence(self, path):
        """
        Convert path (list of indices) to action sequence
        
        Args:
            path: list of node indices
            
        Returns:
            actions: list of actions (or None if actions not available)
        """
        if self.actions is None:
            logging.warning("Actions not available")
            return None
        
        if path is None or len(path) < 2:
            return []
        
        # Get actions for each step in path
        action_sequence = []
        for idx in path[:-1]:  # Exclude last node (we're already there)
            if idx < len(self.actions):
                action_sequence.append(self.actions[idx])
        
        return action_sequence
    
    def get_waypoints(self, path, num_waypoints=5):
        """
        Extract evenly spaced waypoints from path
        
        Args:
            path: list of node indices
            num_waypoints: number of waypoints to extract
            
        Returns:
            waypoints: list of waypoint indices
        """
        if path is None or len(path) == 0:
            return []
        
        if len(path) <= num_waypoints:
            return path
        
        # Sample evenly
        indices = np.linspace(0, len(path) - 1, num_waypoints, dtype=int)
        waypoints = [path[i] for i in indices]
        
        return waypoints
    
    def get_next_node(self, current_idx, path):
        """
        Get next node in path from current position
        
        Args:
            current_idx: current node index
            path: planned path
            
        Returns:
            next_idx: next node index (or None if at end)
        """
        if path is None or len(path) < 2:
            return None
        
        try:
            pos = path.index(current_idx)
            if pos >= len(path) - 1:
                return None  # Already at goal
            return path[pos + 1]
        except ValueError:
            # Current position not in path, return first node
            return path[0]


if __name__ == "__main__":
    # Test graph building
    logging.info("Testing navigation graph...")
    
    # Create dummy features for testing
    n_test = 100
    dummy_features = np.random.randn(n_test, 2048)
    dummy_features = dummy_features / np.linalg.norm(dummy_features, axis=1, keepdims=True)
    
    graph_planner = NavigationGraph()
    graph = graph_planner.build_graph(dummy_features)
    
    # Test path planning
    path = graph_planner.find_path(0, 99)
    if path:
        logging.info(f"Test path: {path[:10]}... (length: {len(path)})")
        waypoints = graph_planner.get_waypoints(path)
        logging.info(f"Waypoints: {waypoints}")