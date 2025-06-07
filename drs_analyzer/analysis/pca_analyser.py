"""
Clustering analysis for DRS spectroscopy data
K-means, hierarchical, and DBSCAN clustering implementations
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Any, Optional, Tuple, List, Union

from ..config.settings import AppSettings


class ClusteringAnalyzer:
    """
    Advanced clustering analysis for spectral data
    """
    
    def __init__(self, settings: AppSettings = None):
        self.settings = settings or AppSettings()
        self.logger = logging.getLogger(__name__)
        
        # Available clustering methods
        self.methods = {
            'kmeans': self._kmeans_clustering,
            'hierarchical': self._hierarchical_clustering,
            'dbscan': self._dbscan_clustering
        }
        
        # Results storage
        self.results = {}
        self.scaler = StandardScaler()
        
    def analyze(self, 
               data: np.ndarray,
               method: str = 'kmeans',
               n_clusters: Optional[int] = None,
               scale_data: bool = True,
               **kwargs) -> Dict[str, Any]:
        """
        Perform clustering analysis
        
        Args:
            data: Spectral data (n_samples, n_features)
            method: Clustering method ('kmeans', 'hierarchical', 'dbscan')
            n_clusters: Number of clusters (auto-detect if None)
            scale_data: Whether to standardize data
            **kwargs: Method-specific parameters
            
        Returns:
            Dictionary containing clustering results
        """
        try:
            self.logger.info(f"Performing {method} clustering on {data.shape[0]} samples")
            
            # Validate input
            if data.ndim != 2:
                raise ValueError("Data must be 2D array (samples, features)")
            
            # Scale data if requested
            if scale_data:
                data_scaled = self.scaler.fit_transform(data)
            else:
                data_scaled = data.copy()
            
            # Auto-detect number of clusters if not provided
            if n_clusters is None and method != 'dbscan':
                n_clusters = self._estimate_optimal_clusters(data_scaled)
                self.logger.info(f"Auto-detected optimal clusters: {n_clusters}")
            
            # Perform clustering
            if method not in self.methods:
                raise ValueError(f"Unknown clustering method: {method}")
            
            clustering_method = self.methods[method]
            result = clustering_method(data_scaled, n_clusters, **kwargs)
            
            # Add common metrics
            result.update(self._calculate_metrics(data_scaled, result['labels']))
            result['method'] = method
            result['n_samples'] = data.shape[0]
            result['n_features'] = data.shape[1]
            result['scaled_data'] = scale_data
            
            self.results[method] = result
            
            self.logger.info(f"Clustering completed: {len(np.unique(result['labels']))} clusters found")
            return result
            
        except Exception as e:
            self.logger.error(f"Clustering analysis failed: {e}")
            raise
    
    def _kmeans_clustering(self, data: np.ndarray, n_clusters: int, **kwargs) -> Dict[str, Any]:
        """Perform K-means clustering"""
        random_state = kwargs.get('random_state', 42)
        max_iter = kwargs.get('max_iter', 300)
        n_init = kwargs.get('n_init', 10)
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            max_iter=max_iter,
            n_init=n_init
        )
        
        labels = kmeans.fit_predict(data)
        
        return {
            'labels': labels,
            'cluster_centers': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_,
            'n_iter': kmeans.n_iter_,
            'algorithm_params': {
                'n_clusters': n_clusters,
                'max_iter': max_iter,
                'n_init': n_init
            }
        }
    
    def _hierarchical_clustering(self, data: np.ndarray, n_clusters: int, **kwargs) -> Dict[str, Any]:
        """Perform hierarchical clustering"""
        linkage_method = kwargs.get('linkage', 'ward')
        affinity = kwargs.get('affinity', 'euclidean')
        
        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage_method,
            affinity=affinity
        )
        
        labels = hierarchical.fit_predict(data)
        
        # Calculate linkage matrix for dendrogram
        linkage_matrix = linkage(data, method=linkage_method)
        
        return {
            'labels': labels,
            'linkage_matrix': linkage_matrix,
            'algorithm_params': {
                'n_clusters': n_clusters,
                'linkage': linkage_method,
                'affinity': affinity
            }
        }
    
    def _dbscan_clustering(self, data: np.ndarray, n_clusters: int = None, **kwargs) -> Dict[str, Any]:
        """Perform DBSCAN clustering"""
        eps = kwargs.get('eps', 0.5)
        min_samples = kwargs.get('min_samples', 5)
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(data)
        
        # Count actual clusters (excluding noise points)
        n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        return {
            'labels': labels,
            'core_sample_indices': dbscan.core_sample_indices_,
            'n_clusters_found': n_clusters_found,
            'n_noise_points': n_noise,
            'algorithm_params': {
                'eps': eps,
                'min_samples': min_samples
            }
        }
    
    def _estimate_optimal_clusters(self, data: np.ndarray, max_clusters: int = 10) -> int:
        """Estimate optimal number of clusters using elbow method and silhouette score"""
        max_clusters = min(max_clusters, data.shape[0] - 1)
        inertias = []
        silhouette_scores = []
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(data)
            
            inertias.append(kmeans.inertia_)
            if len(set(labels)) > 1:  # Need at least 2 clusters for silhouette score
                sil_score = silhouette_score(data, labels)
                silhouette_scores.append(sil_score)
            else:
                silhouette_scores.append(0)
        
        # Find elbow using second derivative
        if len(inertias) >= 3:
            second_derivatives = []
            for i in range(1, len(inertias) - 1):
                second_deriv = inertias[i-1] - 2*inertias[i] + inertias[i+1]
                second_derivatives.append(second_deriv)
            
            elbow_k = np.argmax(second_derivatives) + 3  # +3 because we start from k=2 and skip first point
        else:
            elbow_k = 2
        
        # Find best silhouette score
        if silhouette_scores:
            best_sil_k = np.argmax(silhouette_scores) + 2  # +2 because we start from k=2
        else:
            best_sil_k = 2
        
        # Return the average of both methods
        optimal_k = int(np.mean([elbow_k, best_sil_k]))
        return max(2, min(optimal_k, max_clusters))
    
    def _calculate_metrics(self, data: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Calculate clustering quality metrics"""
        metrics = {}
        
        # Number of clusters (excluding noise for DBSCAN)
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        metrics['n_clusters'] = n_clusters
        
        if n_clusters > 1:
            # Silhouette score
            try:
                metrics['silhouette_score'] = silhouette_score(data, labels)
            except:
                metrics['silhouette_score'] = np.nan
            
            # Calinski-Harabasz score
            try:
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(data, labels)
            except:
                metrics['calinski_harabasz_score'] = np.nan
        else:
            metrics['silhouette_score'] = np.nan
            metrics['calinski_harabasz_score'] = np.nan
        
        # Cluster sizes
        unique_labels_list = list(unique_labels)
        cluster_sizes = []
        for label in unique_labels_list:
            if label != -1:  # Exclude noise points
                cluster_sizes.append(np.sum(labels == label))
        
        metrics['cluster_sizes'] = cluster_sizes
        metrics['cluster_size_std'] = np.std(cluster_sizes) if cluster_sizes else 0
        
        return metrics
    
    def compare_methods(self, 
                       data: np.ndarray,
                       methods: List[str] = None,
                       n_clusters_range: List[int] = None) -> Dict[str, Any]:
        """Compare different clustering methods"""
        if methods is None:
            methods = ['kmeans', 'hierarchical', 'dbscan']
        
        if n_clusters_range is None:
            n_clusters_range = list(range(2, 8))
        
        comparison_results = {}
        
        for method in methods:
            method_results = {}
            
            if method == 'dbscan':
                # For DBSCAN, vary eps parameter instead of n_clusters
                eps_values = np.linspace(0.1, 2.0, 6)
                for eps in eps_values:
                    result = self.analyze(data, method=method, eps=eps)
                    method_results[f'eps_{eps:.1f}'] = result
            else:
                # For other methods, vary number of clusters
                for n_clusters in n_clusters_range:
                    if n_clusters < data.shape[0]:
                        result = self.analyze(data, method=method, n_clusters=n_clusters)
                        method_results[f'k_{n_clusters}'] = result
            
            comparison_results[method] = method_results
        
        return comparison_results
    
    def get_cluster_statistics(self, 
                              data: np.ndarray, 
                              labels: np.ndarray,
                              feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Calculate detailed statistics for each cluster"""
        unique_labels = np.unique(labels)
        cluster_stats = {}
        
        for label in unique_labels:
            if label == -1:  # Skip noise points in DBSCAN
                continue
                
            cluster_mask = labels == label
            cluster_data = data[cluster_mask]
            
            stats = {
                'size': np.sum(cluster_mask),
                'percentage': (np.sum(cluster_mask) / len(labels)) * 100,
                'centroid': np.mean(cluster_data, axis=0),
                'std': np.std(cluster_data, axis=0),
                'min_values': np.min(cluster_data, axis=0),
                'max_values': np.max(cluster_data, axis=0),
                'median_values': np.median(cluster_data, axis=0)
            }
            
            # Add feature names if provided
            if feature_names:
                stats['feature_names'] = feature_names
            
            cluster_stats[f'cluster_{label}'] = stats
        
        return cluster_stats
    
    def export_results(self, method: str = None) -> Dict[str, Any]:
        """Export clustering results"""
        if method is None:
            return self.results
        elif method in self.results:
            return self.results[method]
        else:
            raise ValueError(f"No results found for method: {method}")