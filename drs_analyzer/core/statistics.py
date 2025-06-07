"""
Advanced statistical analysis for DRS spectroscopy
Comprehensive statistical methods including PCA, clustering, and correlation analysis
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
import warnings

from ..config.settings import AppSettings

@dataclass
class StatisticalResult:
    """Container for statistical analysis results"""
    name: str
    data: Any
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]

class StatisticsError(Exception):
    """Custom exception for statistical analysis errors"""
    pass

class DRSStatistics:
    """
    Comprehensive statistical analysis for DRS spectroscopy
    Includes dimensionality reduction, clustering, and correlation analysis
    """
    
    def __init__(self, settings: AppSettings = None):
        self.settings = settings or AppSettings()
        self.logger = logging.getLogger(__name__)
        
        # Data storage
        self.data = None
        self.wavelengths = None
        self.metadata = {}
        
        # Results cache
        self.results = {}
        self._cache = {}
        
        # Preprocessing
        self.scaler = StandardScaler()
        self.scaled_data = None
    
    def load_data(self, spectra: np.ndarray, wavelengths: np.ndarray, 
                  metadata: Dict[str, Any] = None):
        """Load spectral data for statistical analysis"""
        try:
            if spectra is None or wavelengths is None:
                raise StatisticsError("Spectra and wavelengths cannot be None")
            
            self.data = np.array(spectra, dtype=np.float64)
            self.wavelengths = np.array(wavelengths, dtype=np.float64)
            self.metadata = metadata or {}
            
            # Clear previous results
            self.results.clear()
            self._cache.clear()
            self.scaled_data = None
            
            self.logger.info(f"Loaded {spectra.shape[0]} spectra for statistical analysis")
            
        except Exception as e:
            raise StatisticsError(f"Failed to load data: {e}")
    
    def preprocess_data(self, method: str = "standard") -> np.ndarray:
        """Preprocess data for statistical analysis"""
        try:
            if self.data is None:
                raise StatisticsError("No data loaded")
            
            if method == "standard":
                self.scaled_data = self.scaler.fit_transform(self.data)
            elif method == "none":
                self.scaled_data = self.data.copy()
            else:
                raise StatisticsError(f"Unknown preprocessing method: {method}")
            
            self.logger.info(f"Preprocessed data using {method} scaling")
            return self.scaled_data
            
        except Exception as e:
            raise StatisticsError(f"Data preprocessing failed: {e}")
    
    def calculate_basic_statistics(self) -> Dict[str, Any]:
        """Calculate basic statistical measures"""
        try:
            if self.data is None:
                raise StatisticsError("No data loaded")
            
            # Global statistics
            global_stats = {
                'mean_spectrum': np.mean(self.data, axis=0),
                'std_spectrum': np.std(self.data, axis=0),
                'median_spectrum': np.median(self.data, axis=0),
                'min_spectrum': np.min(self.data, axis=0),
                'max_spectrum': np.max(self.data, axis=0),
                'percentile_25': np.percentile(self.data, 25, axis=0),
                'percentile_75': np.percentile(self.data, 75, axis=0)
            }
            
            # Per-spectrum statistics
            spectrum_stats = {
                'mean_intensity': np.mean(self.data, axis=1),
                'std_intensity': np.std(self.data, axis=1),
                'total_intensity': np.sum(self.data, axis=1),
                'max_intensity': np.max(self.data, axis=1),
                'min_intensity': np.min(self.data, axis=1),
                'range_intensity': np.ptp(self.data, axis=1),
                'skewness': stats.skew(self.data, axis=1),
                'kurtosis': stats.kurtosis(self.data, axis=1)
            }
            
            # Overall statistics
            overall_stats = {
                'total_spectra': self.data.shape[0],
                'wavelength_points': self.data.shape[1],
                'wavelength_range': (self.wavelengths.min(), self.wavelengths.max()),
                'global_mean': np.mean(self.data),
                'global_std': np.std(self.data),
                'global_min': np.min(self.data),
                'global_max': np.max(self.data),
                'signal_to_noise_ratio': self._calculate_snr()
            }
            
            result = {
                'global_statistics': global_stats,
                'spectrum_statistics': spectrum_stats,
                'overall_statistics': overall_stats
            }
            
            self.results['basic_statistics'] = StatisticalResult(
                name="Basic Statistics",
                data=result,
                parameters={},
                metadata={'calculation_time': 'real-time'}
            )
            
            return result
            
        except Exception as e:
            raise StatisticsError(f"Basic statistics calculation failed: {e}")
    
    def _calculate_snr(self) -> float:
        """Calculate signal-to-noise ratio"""
        try:
            signal_power = np.mean(self.data**2)
            noise_power = np.var(self.data - np.mean(self.data, axis=0))
            return 10 * np.log10(signal_power / (noise_power + 1e-10))
        except:
            return 0.0
    
    def perform_pca(self, n_components: Optional[int] = None, 
                   return_loadings: bool = True) -> Dict[str, Any]:
        """Perform Principal Component Analysis"""
        try:
            if self.data is None:
                raise StatisticsError("No data loaded")
            
            # Preprocess data if not done
            if self.scaled_data is None:
                self.preprocess_data()
            
            # Determine number of components
            if n_components is None:
                n_components = min(10, self.data.shape[0] - 1, self.data.shape[1])
            
            # Perform PCA
            pca = PCA(n_components=n_components)
            scores = pca.fit_transform(self.scaled_data)
            
            # Calculate additional metrics
            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)
            
            result = {
                'scores': scores,
                'explained_variance_ratio': explained_variance_ratio,
                'cumulative_variance': cumulative_variance,
                'singular_values': pca.singular_values_,
                'n_components': n_components,
                'total_variance_explained': cumulative_variance[-1]
            }
            
            if return_loadings:
                result['loadings'] = pca.components_
                result['loadings_wavelengths'] = self.wavelengths
            
            self.results['pca'] = StatisticalResult(
                name="Principal Component Analysis",
                data=result,
                parameters={'n_components': n_components},
                metadata={'method': 'sklearn.PCA'}
            )
            
            self.logger.info(f"PCA completed: {n_components} components, "
                           f"{result['total_variance_explained']:.2%} variance explained")
            
            return result
            
        except Exception as e:
            raise StatisticsError(f"PCA analysis failed: {e}")
    
    def perform_ica(self, n_components: Optional[int] = None) -> Dict[str, Any]:
        """Perform Independent Component Analysis"""
        try:
            if self.data is None:
                raise StatisticsError("No data loaded")
            
            if self.scaled_data is None:
                self.preprocess_data()
            
            if n_components is None:
                n_components = min(5, self.data.shape[0] - 1, self.data.shape[1])
            
            # Perform ICA
            ica = FastICA(n_components=n_components, random_state=42, max_iter=1000)
            sources = ica.fit_transform(self.scaled_data)
            
            result = {
                'sources': sources,
                'mixing_matrix': ica.mixing_,
                'components': ica.components_,
                'n_components': n_components,
                'n_iter': ica.n_iter_
            }
            
            self.results['ica'] = StatisticalResult(
                name="Independent Component Analysis",
                data=result,
                parameters={'n_components': n_components},
                metadata={'method': 'sklearn.FastICA', 'max_iter': 1000}
            )
            
            self.logger.info(f"ICA completed: {n_components} components, {ica.n_iter_} iterations")
            
            return result
            
        except Exception as e:
            raise StatisticsError(f"ICA analysis failed: {e}")
    
    def perform_clustering(self, method: str = "kmeans", **kwargs) -> Dict[str, Any]:
        """Perform clustering analysis"""
        try:
            if self.data is None:
                raise StatisticsError("No data loaded")
            
            if self.scaled_data is None:
                self.preprocess_data()
            
            if method == "kmeans":
                n_clusters = kwargs.get('n_clusters', 3)
                clusterer = KMeans(n_clusters=n_clusters, random_state=42)
                
            elif method == "dbscan":
                eps = kwargs.get('eps', 0.5)
                min_samples = kwargs.get('min_samples', 5)
                clusterer = DBSCAN(eps=eps, min_samples=min_samples)
                
            elif method == "hierarchical":
                n_clusters = kwargs.get('n_clusters', 3)
                linkage_method = kwargs.get('linkage', 'ward')
                clusterer = AgglomerativeClustering(
                    n_clusters=n_clusters, 
                    linkage=linkage_method
                )
                
            else:
                raise StatisticsError(f"Unknown clustering method: {method}")
            
            # Fit clustering
            labels = clusterer.fit_predict(self.scaled_data)
            
            # Calculate metrics
            if len(set(labels)) > 1:
                silhouette_avg = silhouette_score(self.scaled_data, labels)
                calinski_score = calinski_harabasz_score(self.scaled_data, labels)
            else:
                silhouette_avg = 0
                calinski_score = 0
            
            # Cluster statistics
            unique_labels = np.unique(labels)
            cluster_stats = {}
            for label in unique_labels:
                mask = labels == label
                cluster_data = self.data[mask]
                cluster_stats[f'cluster_{label}'] = {
                    'size': np.sum(mask),
                    'mean_spectrum': np.mean(cluster_data, axis=0),
                    'std_spectrum': np.std(cluster_data, axis=0),
                    'centroid': np.mean(cluster_data, axis=0)
                }
            
            result = {
                'labels': labels,
                'n_clusters': len(unique_labels),
                'cluster_centers': getattr(clusterer, 'cluster_centers_', None),
                'silhouette_score': silhouette_avg,
                'calinski_harabasz_score': calinski_score,
                'cluster_statistics': cluster_stats,
                'method': method
            }
            
            # Add hierarchical-specific results
            if method == "hierarchical":
                linkage_matrix = linkage(self.scaled_data, method=kwargs.get('linkage', 'ward'))
                result['linkage_matrix'] = linkage_matrix
            
            self.results['clustering'] = StatisticalResult(
                name=f"Clustering ({method})",
                data=result,
                parameters=kwargs,
                metadata={'method': method}
            )
            
            self.logger.info(f"Clustering completed: {method}, {len(unique_labels)} clusters, "
                           f"silhouette score: {silhouette_avg:.3f}")
            
            return result
            
        except Exception as e:
            raise StatisticsError(f"Clustering analysis failed: {e}")
    
    def calculate_correlation_matrix(self, method: str = "pearson") -> Dict[str, Any]:
        """Calculate correlation matrix between spectra"""
        try:
            if self.data is None:
                raise StatisticsError("No data loaded")
            
            if method == "pearson":
                corr_matrix = np.corrcoef(self.data)
            elif method == "spearman":
                corr_matrix = stats.spearmanr(self.data, axis=1)[0]
            elif method == "kendall":
                # Note: This is computationally expensive for large datasets
                n_spectra = self.data.shape[0]
                corr_matrix = np.zeros((n_spectra, n_spectra))
                for i in range(n_spectra):
                    for j in range(i, n_spectra):
                        tau, _ = stats.kendalltau(self.data[i], self.data[j])
                        corr_matrix[i, j] = tau
                        corr_matrix[j, i] = tau
            else:
                raise StatisticsError(f"Unknown correlation method: {method}")
            
            # Calculate summary statistics
            upper_triangle = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
            
            result = {
                'correlation_matrix': corr_matrix,
                'method': method,
                'mean_correlation': np.mean(upper_triangle),
                'std_correlation': np.std(upper_triangle),
                'min_correlation': np.min(upper_triangle),
                'max_correlation': np.max(upper_triangle),
                'highly_correlated_pairs': self._find_highly_correlated_pairs(corr_matrix, 0.9)
            }
            
            self.results['correlation'] = StatisticalResult(
                name=f"Correlation Analysis ({method})",
                data=result,
                parameters={'method': method},
                metadata={'method': method}
            )
            
            self.logger.info(f"Correlation analysis completed: {method}, "
                           f"mean correlation: {result['mean_correlation']:.3f}")
            
            return result
            
        except Exception as e:
            raise StatisticsError(f"Correlation analysis failed: {e}")
    
    def _find_highly_correlated_pairs(self, corr_matrix: np.ndarray, 
                                     threshold: float) -> List[Tuple[int, int, float]]:
        """Find pairs of spectra with high correlation"""
        pairs = []
        n = corr_matrix.shape[0]
        
        for i in range(n):
            for j in range(i+1, n):
                if abs(corr_matrix[i, j]) >= threshold:
                    pairs.append((i, j, corr_matrix[i, j]))
        
        return sorted(pairs, key=lambda x: abs(x[2]), reverse=True)
    
    def perform_tsne(self, n_components: int = 2, **kwargs) -> Dict[str, Any]:
        """Perform t-SNE dimensionality reduction"""
        try:
            if self.data is None:
                raise StatisticsError("No data loaded")
            
            if self.scaled_data is None:
                self.preprocess_data()
            
            # t-SNE parameters
            perplexity = kwargs.get('perplexity', min(30, self.data.shape[0] - 1))
            learning_rate = kwargs.get('learning_rate', 200)
            n_iter = kwargs.get('n_iter', 1000)
            
            tsne = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                learning_rate=learning_rate,
                n_iter=n_iter,
                random_state=42
            )
            
            embedding = tsne.fit_transform(self.scaled_data)
            
            result = {
                'embedding': embedding,
                'n_components': n_components,
                'perplexity': perplexity,
                'learning_rate': learning_rate,
                'n_iter': n_iter,
                'kl_divergence': tsne.kl_divergence_
            }
            
            self.results['tsne'] = StatisticalResult(
                name="t-SNE",
                data=result,
                parameters=kwargs,
                metadata={'method': 't-SNE'}
            )
            
            self.logger.info(f"t-SNE completed: {n_components}D embedding, "
                           f"KL divergence: {tsne.kl_divergence_:.3f}")
            
            return result
            
        except Exception as e:
            raise StatisticsError(f"t-SNE analysis failed: {e}")
    
    def calculate_spectral_distances(self, metric: str = "euclidean") -> Dict[str, Any]:
        """Calculate pairwise distances between spectra"""
        try:
            if self.data is None:
                raise StatisticsError("No data loaded")
            
            from scipy.spatial.distance import pdist, squareform
            
            # Calculate pairwise distances
            distances = pdist(self.data, metric=metric)
            distance_matrix = squareform(distances)
            
            result = {
                'distance_matrix': distance_matrix,
                'pairwise_distances': distances,
                'metric': metric,
                'mean_distance': np.mean(distances),
                'std_distance': np.std(distances),
                'min_distance': np.min(distances),
                'max_distance': np.max(distances)
            }
            
            self.results['distances'] = StatisticalResult(
                name=f"Spectral Distances ({metric})",
                data=result,
                parameters={'metric': metric},
                metadata={'method': metric}
            )
            
            self.logger.info(f"Distance calculation completed: {metric}, "
                           f"mean distance: {result['mean_distance']:.3f}")
            
            return result
            
        except Exception as e:
            raise StatisticsError(f"Distance calculation failed: {e}")
    
    def detect_outliers(self, method: str = "isolation_forest", **kwargs) -> Dict[str, Any]:
        """Detect outlier spectra"""
        try:
            if self.data is None:
                raise StatisticsError("No data loaded")
            
            if self.scaled_data is None:
                self.preprocess_data()
            
            if method == "isolation_forest":
                from sklearn.ensemble import IsolationForest
                
                contamination = kwargs.get('contamination', 0.1)
                detector = IsolationForest(contamination=contamination, random_state=42)
                outlier_labels = detector.fit_predict(self.scaled_data)
                outlier_scores = detector.score_samples(self.scaled_data)
                
            elif method == "zscore":
                threshold = kwargs.get('threshold', 3.0)
                z_scores = np.abs(stats.zscore(self.data, axis=1))
                outlier_mask = np.any(z_scores > threshold, axis=1)
                outlier_labels = np.where(outlier_mask, -1, 1)
                outlier_scores = np.max(z_scores, axis=1)
                
            elif method == "iqr":
                # Interquartile range method
                q1 = np.percentile(self.data, 25, axis=1)
                q3 = np.percentile(self.data, 75, axis=1)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outlier_mask = np.any(
                    (self.data < lower_bound[:, np.newaxis]) |
                    (self.data > upper_bound[:, np.newaxis]), axis=1
                )
                outlier_labels = np.where(outlier_mask, -1, 1)
                outlier_scores = np.max(np.abs(self.data - np.median(self.data, axis=1, keepdims=True)), axis=1)
                
            else:
                raise StatisticsError(f"Unknown outlier detection method: {method}")
            
            outlier_indices = np.where(outlier_labels == -1)[0]
            
            result = {
                'outlier_labels': outlier_labels,
                'outlier_scores': outlier_scores,
                'outlier_indices': outlier_indices,
                'n_outliers': len(outlier_indices),
                'outlier_percentage': len(outlier_indices) / len(outlier_labels) * 100,
                'method': method
            }
            
            self.results['outliers'] = StatisticalResult(
                name=f"Outlier Detection ({method})",
                data=result,
                parameters=kwargs,
                metadata={'method': method}
            )
            
            self.logger.info(f"Outlier detection completed: {method}, "
                           f"{len(outlier_indices)} outliers ({result['outlier_percentage']:.1f}%)")
            
            return result
            
        except Exception as e:
            raise StatisticsError(f"Outlier detection failed: {e}")
    
    def generate_statistical_report(self, include_all: bool = True) -> Dict[str, Any]:
        """Generate comprehensive statistical report"""
        try:
            if self.data is None:
                raise StatisticsError("No data loaded")
            
            report = {
                'data_summary': {
                    'n_spectra': self.data.shape[0],
                    'n_wavelengths': self.data.shape[1],
                    'wavelength_range': (self.wavelengths.min(), self.wavelengths.max()),
                    'data_shape': self.data.shape
                }
            }
            
            if include_all:
                # Run all analyses
                self.logger.info("Generating comprehensive statistical report...")
                
                report['basic_statistics'] = self.calculate_basic_statistics()
                report['pca'] = self.perform_pca()
                report['correlation'] = self.calculate_correlation_matrix()
                report['clustering'] = self.perform_clustering()
                
                # Optional analyses (may be computationally expensive)
                try:
                    report['outliers'] = self.detect_outliers()
                except Exception as e:
                    self.logger.warning(f"Outlier detection failed: {e}")
                
                try:
                    if self.data.shape[0] <= 1000:  # Limit t-SNE for large datasets
                        report['tsne'] = self.perform_tsne()
                except Exception as e:
                    self.logger.warning(f"t-SNE analysis failed: {e}")
            
            # Add metadata
            report['metadata'] = {
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'data_metadata': self.metadata,
                'preprocessing': 'standard_scaling'
            }
            
            self.logger.info("Statistical report generation completed")
            return report
            
        except Exception as e:
            raise StatisticsError(f"Report generation failed: {e}")
    
    def export_results(self) -> Dict[str, Any]:
        """Export all statistical results"""
        return {
            'results': {name: result.data for name, result in self.results.items()},
            'parameters': {name: result.parameters for name, result in self.results.items()},
            'metadata': {name: result.metadata for name, result in self.results.items()},
            'data_info': {
                'shape': self.data.shape if self.data is not None else None,
                'wavelength_range': (self.wavelengths.min(), self.wavelengths.max()) if self.wavelengths is not None else None
            }
        }
    
    def clear_results(self):
        """Clear all statistical results"""
        self.results.clear()
        self._cache.clear()
        self.scaled_data = None