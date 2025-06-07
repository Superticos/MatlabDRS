"""
Comprehensive statistical analysis for DRS spectroscopy data
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import normaltest, shapiro, kstest, anderson
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif, mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Any, Optional, Tuple, List, Union

from ..config.settings import AppSettings


class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis for spectral data
    """
    
    def __init__(self, settings: AppSettings = None):
        self.settings = settings or AppSettings()
        self.logger = logging.getLogger(__name__)
        
    def compute_descriptive_statistics(self, 
                                     data: np.ndarray,
                                     wavelengths: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Compute comprehensive descriptive statistics
        
        Args:
            data: Spectral data (n_samples, n_wavelengths)
            wavelengths: Wavelength array
            
        Returns:
            Dictionary containing statistical measures
        """
        try:
            self.logger.info(f"Computing descriptive statistics for {data.shape[0]} spectra")
            
            stats_dict = {
                'mean_spectrum': np.mean(data, axis=0),
                'median_spectrum': np.median(data, axis=0),
                'std_spectrum': np.std(data, axis=0),
                'var_spectrum': np.var(data, axis=0),
                'min_spectrum': np.min(data, axis=0),
                'max_spectrum': np.max(data, axis=0),
                'q25_spectrum': np.percentile(data, 25, axis=0),
                'q75_spectrum': np.percentile(data, 75, axis=0),
                'iqr_spectrum': np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0),
                'skewness_spectrum': stats.skew(data, axis=0),
                'kurtosis_spectrum': stats.kurtosis(data, axis=0),
                'cv_spectrum': np.std(data, axis=0) / np.mean(data, axis=0)  # Coefficient of variation
            }
            
            # Overall statistics
            stats_dict.update({
                'n_spectra': data.shape[0],
                'n_wavelengths': data.shape[1],
                'overall_mean': np.mean(data),
                'overall_std': np.std(data),
                'overall_min': np.min(data),
                'overall_max': np.max(data),
                'overall_range': np.max(data) - np.min(data),
                'overall_snr': self._calculate_snr(data),
                'spectral_similarity': self._calculate_spectral_similarity(data)
            })
            
            if wavelengths is not None:
                stats_dict['wavelengths'] = wavelengths
                stats_dict['wavelength_range'] = (wavelengths.min(), wavelengths.max())
                stats_dict['wavelength_resolution'] = np.mean(np.diff(wavelengths))
            
            return stats_dict
            
        except Exception as e:
            self.logger.error(f"Failed to compute descriptive statistics: {e}")
            raise
    
    def test_normality(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Test normality of spectral data using multiple tests
        
        Args:
            data: Spectral data
            
        Returns:
            Dictionary containing normality test results
        """
        try:
            # Flatten data for overall normality test
            flat_data = data.flatten()
            
            # Sample for large datasets to avoid computational issues
            if len(flat_data) > 5000:
                sample_indices = np.random.choice(len(flat_data), 5000, replace=False)
                sample_data = flat_data[sample_indices]
            else:
                sample_data = flat_data
            
            normality_results = {}
            
            # Shapiro-Wilk test (good for small samples)
            if len(sample_data) <= 5000:
                try:
                    shapiro_stat, shapiro_p = shapiro(sample_data)
                    normality_results['shapiro_wilk'] = {
                        'statistic': shapiro_stat,
                        'p_value': shapiro_p,
                        'is_normal': shapiro_p > 0.05
                    }
                except:
                    normality_results['shapiro_wilk'] = {'error': 'Test failed'}
            
            # D'Agostino-Pearson test
            try:
                dagostino_stat, dagostino_p = normaltest(sample_data)
                normality_results['dagostino_pearson'] = {
                    'statistic': dagostino_stat,
                    'p_value': dagostino_p,
                    'is_normal': dagostino_p > 0.05
                }
            except:
                normality_results['dagostino_pearson'] = {'error': 'Test failed'}
            
            # Kolmogorov-Smirnov test
            try:
                # Test against normal distribution with same mean and std
                ks_stat, ks_p = kstest(sample_data, 'norm', 
                                      args=(np.mean(sample_data), np.std(sample_data)))
                normality_results['kolmogorov_smirnov'] = {
                    'statistic': ks_stat,
                    'p_value': ks_p,
                    'is_normal': ks_p > 0.05
                }
            except:
                normality_results['kolmogorov_smirnov'] = {'error': 'Test failed'}
            
            # Anderson-Darling test
            try:
                anderson_result = anderson(sample_data, dist='norm')
                # Use 5% significance level (index 2)
                is_normal = anderson_result.statistic < anderson_result.critical_values[2]
                normality_results['anderson_darling'] = {
                    'statistic': anderson_result.statistic,
                    'critical_values': anderson_result.critical_values,
                    'significance_levels': anderson_result.significance_level,
                    'is_normal': is_normal
                }
            except:
                normality_results['anderson_darling'] = {'error': 'Test failed'}
            
            # Overall assessment
            normal_count = sum(1 for test in normality_results.values() 
                             if isinstance(test, dict) and test.get('is_normal', False))
            total_tests = sum(1 for test in normality_results.values() 
                            if isinstance(test, dict) and 'is_normal' in test)
            
            normality_results['summary'] = {
                'tests_passed': normal_count,
                'total_tests': total_tests,
                'overall_normal': normal_count > total_tests / 2 if total_tests > 0 else False
            }
            
            return normality_results
            
        except Exception as e:
            self.logger.error(f"Normality testing failed: {e}")
            raise
    
    def perform_correlation_analysis(self, 
                                   data: np.ndarray,
                                   method: str = 'pearson') -> Dict[str, Any]:
        """
        Perform correlation analysis between wavelengths
        
        Args:
            data: Spectral data
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Dictionary containing correlation results
        """
        try:
            self.logger.info(f"Computing {method} correlation matrix")
            
            # Compute correlation matrix
            if method == 'pearson':
                corr_matrix = np.corrcoef(data.T)
            elif method == 'spearman':
                corr_matrix, _ = stats.spearmanr(data, axis=0)
            elif method == 'kendall':
                # Kendall tau is computationally expensive for large datasets
                if data.shape[1] > 1000:
                    self.logger.warning("Large dataset - sampling for Kendall correlation")
                    sample_indices = np.random.choice(data.shape[1], 1000, replace=False)
                    sample_data = data[:, sample_indices]
                    sample_corr, _ = stats.kendalltau(sample_data.T)
                    # Create full matrix with NaN and fill sampled positions
                    corr_matrix = np.full((data.shape[1], data.shape[1]), np.nan)
                    np.fill_diagonal(corr_matrix, 1.0)
                    for i, idx1 in enumerate(sample_indices):
                        for j, idx2 in enumerate(sample_indices):
                            corr_matrix[idx1, idx2] = sample_corr[i, j]
                else:
                    corr_matrix, _ = stats.kendalltau(data.T)
            else:
                raise ValueError(f"Unknown correlation method: {method}")
            
            # Calculate correlation statistics
            # Remove diagonal elements for statistics
            off_diagonal = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
            off_diagonal = off_diagonal[~np.isnan(off_diagonal)]
            
            correlation_stats = {
                'correlation_matrix': corr_matrix,
                'method': method,
                'mean_correlation': np.mean(off_diagonal),
                'std_correlation': np.std(off_diagonal),
                'min_correlation': np.min(off_diagonal),
                'max_correlation': np.max(off_diagonal),
                'high_correlation_pairs': self._find_high_correlations(corr_matrix, threshold=0.9)
            }
            
            return correlation_stats
            
        except Exception as e:
            self.logger.error(f"Correlation analysis failed: {e}")
            raise
    
    def detect_outliers(self, 
                       data: np.ndarray,
                       method: str = 'iqr') -> Dict[str, Any]:
        """
        Detect outliers in spectral data
        
        Args:
            data: Spectral data
            method: Outlier detection method ('iqr', 'zscore', 'modified_zscore', 'isolation_forest')
            
        Returns:
            Dictionary containing outlier detection results
        """
        try:
            self.logger.info(f"Detecting outliers using {method} method")
            
            if method == 'iqr':
                outliers = self._detect_outliers_iqr(data)
            elif method == 'zscore':
                outliers = self._detect_outliers_zscore(data)
            elif method == 'modified_zscore':
                outliers = self._detect_outliers_modified_zscore(data)
            elif method == 'isolation_forest':
                outliers = self._detect_outliers_isolation_forest(data)
            else:
                raise ValueError(f"Unknown outlier detection method: {method}")
            
            # Calculate outlier statistics
            n_outliers = np.sum(outliers['is_outlier'])
            outlier_percentage = (n_outliers / data.shape[0]) * 100
            
            result = {
                'method': method,
                'outlier_indices': np.where(outliers['is_outlier'])[0],
                'outlier_scores': outliers.get('scores', None),
                'n_outliers': n_outliers,
                'outlier_percentage': outlier_percentage,
                'is_outlier': outliers['is_outlier']
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Outlier detection failed: {e}")
            raise
    
    def _calculate_snr(self, data: np.ndarray) -> float:
        """Calculate signal-to-noise ratio"""
        # Simple SNR calculation: mean signal / std noise
        signal = np.mean(data)
        noise = np.std(data)
        return signal / noise if noise > 0 else np.inf
    
    def _calculate_spectral_similarity(self, data: np.ndarray) -> Dict[str, float]:
        """Calculate spectral similarity metrics"""
        n_spectra = data.shape[0]
        similarities = []
        
        # Calculate pairwise correlations
        for i in range(n_spectra):
            for j in range(i + 1, n_spectra):
                corr = np.corrcoef(data[i], data[j])[0, 1]
                if not np.isnan(corr):
                    similarities.append(corr)
        
        if similarities:
            return {
                'mean_similarity': np.mean(similarities),
                'std_similarity': np.std(similarities),
                'min_similarity': np.min(similarities),
                'max_similarity': np.max(similarities)
            }
        else:
            return {
                'mean_similarity': np.nan,
                'std_similarity': np.nan,
                'min_similarity': np.nan,
                'max_similarity': np.nan
            }
    
    def _find_high_correlations(self, corr_matrix: np.ndarray, threshold: float = 0.9) -> List[Tuple]:
        """Find pairs of wavelengths with high correlation"""
        high_corr_pairs = []
        
        for i in range(corr_matrix.shape[0]):
            for j in range(i + 1, corr_matrix.shape[1]):
                if not np.isnan(corr_matrix[i, j]) and abs(corr_matrix[i, j]) > threshold:
                    high_corr_pairs.append((i, j, corr_matrix[i, j]))
        
        return high_corr_pairs
    
    def _detect_outliers_iqr(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect outliers using IQR method"""
        # Calculate IQR for each spectrum
        outlier_flags = np.zeros(data.shape[0], dtype=bool)
        
        for i in range(data.shape[0]):
            spectrum = data[i]
            q1 = np.percentile(spectrum, 25)
            q3 = np.percentile(spectrum, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Check if spectrum has outlier points
            outlier_points = (spectrum < lower_bound) | (spectrum > upper_bound)
            
            # Consider spectrum outlier if > 5% of points are outliers
            if np.sum(outlier_points) > 0.05 * len(spectrum):
                outlier_flags[i] = True
        
        return {'is_outlier': outlier_flags}
    
    def _detect_outliers_zscore(self, data: np.ndarray, threshold: float = 3.0) -> Dict[str, Any]:
        """Detect outliers using Z-score method"""
        # Calculate Z-scores for mean intensity of each spectrum
        mean_intensities = np.mean(data, axis=1)
        z_scores = np.abs(stats.zscore(mean_intensities))
        
        outlier_flags = z_scores > threshold
        
        return {
            'is_outlier': outlier_flags,
            'scores': z_scores
        }
    
    def _detect_outliers_modified_zscore(self, data: np.ndarray, threshold: float = 3.5) -> Dict[str, Any]:
        """Detect outliers using modified Z-score method"""
        # Calculate modified Z-scores using median
        mean_intensities = np.mean(data, axis=1)
        median = np.median(mean_intensities)
        mad = np.median(np.abs(mean_intensities - median))
        
        modified_z_scores = 0.6745 * (mean_intensities - median) / mad
        outlier_flags = np.abs(modified_z_scores) > threshold
        
        return {
            'is_outlier': outlier_flags,
            'scores': np.abs(modified_z_scores)
        }
    
    def _detect_outliers_isolation_forest(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect outliers using Isolation Forest"""
        try:
            from sklearn.ensemble import IsolationForest
            
            # Use PCA to reduce dimensionality for Isolation Forest
            if data.shape[1] > 50:
                pca = PCA(n_components=50)
                data_reduced = pca.fit_transform(data)
            else:
                data_reduced = data
            
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(data_reduced)
            
            # -1 indicates outlier, 1 indicates normal
            outlier_flags = outlier_labels == -1
            
            # Get anomaly scores
            scores = iso_forest.decision_function(data_reduced)
            
            return {
                'is_outlier': outlier_flags,
                'scores': -scores  # Negative scores for outliers
            }
            
        except ImportError:
            self.logger.warning("sklearn not available, falling back to Z-score method")
            return self._detect_outliers_zscore(data)
    
    def compare_groups(self, 
                      group1: np.ndarray,
                      group2: np.ndarray,
                      test: str = 'ttest') -> Dict[str, Any]:
        """
        Compare two groups of spectra statistically
        
        Args:
            group1: First group of spectra
            group2: Second group of spectra
            test: Statistical test ('ttest', 'mannwhitney', 'ks')
            
        Returns:
            Dictionary containing comparison results
        """
        try:
            self.logger.info(f"Comparing groups using {test} test")
            
            if test == 'ttest':
                # Independent t-test for each wavelength
                statistics, p_values = stats.ttest_ind(group1, group2, axis=0)
            elif test == 'mannwhitney':
                # Mann-Whitney U test for each wavelength
                statistics = np.zeros(group1.shape[1])
                p_values = np.zeros(group1.shape[1])
                
                for i in range(group1.shape[1]):
                    stat, p_val = stats.mannwhitneyu(group1[:, i], group2[:, i])
                    statistics[i] = stat
                    p_values[i] = p_val
            elif test == 'ks':
                # Kolmogorov-Smirnov test for each wavelength
                statistics = np.zeros(group1.shape[1])
                p_values = np.zeros(group1.shape[1])
                
                for i in range(group1.shape[1]):
                    stat, p_val = stats.ks_2samp(group1[:, i], group2[:, i])
                    statistics[i] = stat
                    p_values[i] = p_val
            else:
                raise ValueError(f"Unknown test: {test}")
            
            # Multiple testing correction (Bonferroni)
            p_values_corrected = p_values * len(p_values)
            p_values_corrected[p_values_corrected > 1] = 1
            
            # Find significant differences
            significant_indices = np.where(p_values_corrected < 0.05)[0]
            
            result = {
                'test': test,
                'statistics': statistics,
                'p_values': p_values,
                'p_values_corrected': p_values_corrected,
                'significant_indices': significant_indices,
                'n_significant': len(significant_indices),
                'group1_mean': np.mean(group1, axis=0),
                'group2_mean': np.mean(group2, axis=0),
                'effect_size': np.mean(group1, axis=0) - np.mean(group2, axis=0)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Group comparison failed: {e}")
            raise