"""
Advanced peak detection for DRS spectroscopy
Multiple algorithms with intelligent clustering
"""

import numpy as np
from scipy import signal
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN, KMeans
import logging
from typing import List, Tuple, Dict, Any, Optional, NamedTuple
from dataclasses import dataclass

from ..config.settings import AppSettings, PeakDetectionSettings

class Peak(NamedTuple):
    """Peak data structure"""
    wavelength: float
    intensity: float
    width: float
    prominence: float
    area: float
    spectrum_index: int

class PeakCluster:
    """Peak cluster data structure"""
    def __init__(self, peaks: List[Peak]):
        self.peaks = peaks
        self.center_wavelength = np.mean([p.wavelength for p in peaks])
        self.mean_intensity = np.mean([p.intensity for p in peaks])
        self.std_wavelength = np.std([p.wavelength for p in peaks])
        self.size = len(peaks)

class PeakDetectionError(Exception):
    """Custom exception for peak detection errors"""
    pass

class PeakDetector:
    """
    Advanced peak detector for DRS spectroscopy
    Supports multiple detection algorithms and clustering
    """
    
    def __init__(self, settings: AppSettings = None):
        self.settings = settings or AppSettings()
        self.logger = logging.getLogger(__name__)
        
        # Detection results
        self.peaks = []
        self.clustered_peaks = []
        self.detection_metadata = {}
        
        # Caching
        self._cache = {}
    
    def detect_peaks(self, 
                    spectra: np.ndarray, 
                    wavelengths: np.ndarray,
                    method: str = "scipy_peaks",
                    **kwargs) -> List[List[Peak]]:
        """
        Detect peaks in spectral data
        Returns list of peak lists (one per spectrum)
        """
        try:
            if spectra is None or wavelengths is None:
                raise PeakDetectionError("Spectra and wavelengths cannot be None")
            
            # Get detection settings
            peak_settings = self.settings.get_peak_detection_settings()
            
            # Override with kwargs
            height = kwargs.get('height', peak_settings.height_threshold)
            prominence = kwargs.get('prominence', peak_settings.prominence_threshold)
            distance = kwargs.get('distance', peak_settings.distance_threshold)
            width_range = kwargs.get('width_range', peak_settings.width_range)
            
            self.peaks = []
            
            for i, spectrum in enumerate(spectra):
                if method == "scipy_peaks":
                    spectrum_peaks = self._detect_scipy_peaks(
                        spectrum, wavelengths, i, height, prominence, distance, width_range
                    )
                elif method == "cwt":
                    spectrum_peaks = self._detect_cwt_peaks(
                        spectrum, wavelengths, i, **kwargs
                    )
                elif method == "derivative":
                    spectrum_peaks = self._detect_derivative_peaks(
                        spectrum, wavelengths, i, **kwargs
                    )
                elif method == "threshold":
                    spectrum_peaks = self._detect_threshold_peaks(
                        spectrum, wavelengths, i, height
                    )
                else:
                    raise PeakDetectionError(f"Unknown detection method: {method}")
                
                self.peaks.append(spectrum_peaks)
            
            # Store metadata
            self.detection_metadata = {
                'method': method,
                'parameters': kwargs,
                'total_peaks': sum(len(peaks) for peaks in self.peaks),
                'spectra_count': len(spectra)
            }
            
            self.logger.info(f"Detected {self.detection_metadata['total_peaks']} peaks using {method}")
            return self.peaks
            
        except Exception as e:
            raise PeakDetectionError(f"Peak detection failed: {e}")
    
    def _detect_scipy_peaks(self, 
                           spectrum: np.ndarray, 
                           wavelengths: np.ndarray,
                           spectrum_index: int,
                           height: float,
                           prominence: float, 
                           distance: float,
                           width_range: Tuple[float, float]) -> List[Peak]:
        """Detect peaks using scipy.signal.find_peaks"""
        
        # Convert distance from wavelength to index units
        distance_idx = int(distance / np.mean(np.diff(wavelengths)))
        
        # Find peaks
        peak_indices, properties = signal.find_peaks(
            spectrum,
            height=height,
            prominence=prominence,
            distance=max(1, distance_idx),
            width=width_range
        )
        
        peaks = []
        for i, peak_idx in enumerate(peak_indices):
            peak = Peak(
                wavelength=wavelengths[peak_idx],
                intensity=spectrum[peak_idx],
                width=properties.get('widths', [0])[i] if 'widths' in properties else 0,
                prominence=properties.get('prominences', [0])[i] if 'prominences' in properties else 0,
                area=self._calculate_peak_area(spectrum, peak_idx, properties.get('widths', [5])[i]),
                spectrum_index=spectrum_index
            )
            peaks.append(peak)
        
        return peaks
    
    def _detect_cwt_peaks(self, 
                         spectrum: np.ndarray, 
                         wavelengths: np.ndarray,
                         spectrum_index: int,
                         **kwargs) -> List[Peak]:
        """Detect peaks using Continuous Wavelet Transform"""
        
        widths = kwargs.get('widths', np.arange(1, 20))
        min_snr = kwargs.get('min_snr', 1)
        noise_perc = kwargs.get('noise_perc', 10)
        
        peak_indices = signal.find_peaks_cwt(
            spectrum, 
            widths, 
            min_snr=min_snr,
            noise_perc=noise_perc
        )
        
        peaks = []
        for peak_idx in peak_indices:
            if 0 <= peak_idx < len(spectrum):
                peak = Peak(
                    wavelength=wavelengths[peak_idx],
                    intensity=spectrum[peak_idx],
                    width=np.mean(widths),
                    prominence=self._calculate_prominence(spectrum, peak_idx),
                    area=self._calculate_peak_area(spectrum, peak_idx, np.mean(widths)),
                    spectrum_index=spectrum_index
                )
                peaks.append(peak)
        
        return peaks
    
    def _detect_derivative_peaks(self, 
                               spectrum: np.ndarray, 
                               wavelengths: np.ndarray,
                               spectrum_index: int,
                               **kwargs) -> List[Peak]:
        """Detect peaks using derivative method"""
        
        # Calculate first and second derivatives
        first_deriv = np.gradient(spectrum)
        second_deriv = np.gradient(first_deriv)
        
        # Find zero crossings in first derivative
        zero_crossings = np.where(np.diff(np.sign(first_deriv)))[0]
        
        peaks = []
        for crossing in zero_crossings:
            # Check if it's a maximum (second derivative < 0)
            if crossing < len(second_deriv) and second_deriv[crossing] < 0:
                # Additional filtering
                if spectrum[crossing] > kwargs.get('min_height', 0.1):
                    peak = Peak(
                        wavelength=wavelengths[crossing],
                        intensity=spectrum[crossing],
                        width=self._estimate_peak_width(spectrum, crossing),
                        prominence=self._calculate_prominence(spectrum, crossing),
                        area=self._calculate_peak_area(spectrum, crossing, 5),
                        spectrum_index=spectrum_index
                    )
                    peaks.append(peak)
        
        return peaks
    
    def _detect_threshold_peaks(self, 
                              spectrum: np.ndarray, 
                              wavelengths: np.ndarray,
                              spectrum_index: int,
                              threshold: float) -> List[Peak]:
        """Simple threshold-based peak detection"""
        
        # Find points above threshold
        above_threshold = spectrum > threshold
        
        # Find groups of consecutive points
        diff = np.diff(np.concatenate(([False], above_threshold, [False])).astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        peaks = []
        for start, end in zip(starts, ends):
            if end > start:
                # Find maximum in this region
                region = spectrum[start:end]
                max_idx = np.argmax(region) + start
                
                peak = Peak(
                    wavelength=wavelengths[max_idx],
                    intensity=spectrum[max_idx],
                    width=end - start,
                    prominence=spectrum[max_idx] - threshold,
                    area=np.sum(spectrum[start:end] - threshold),
                    spectrum_index=spectrum_index
                )
                peaks.append(peak)
        
        return peaks
    
    def _calculate_prominence(self, spectrum: np.ndarray, peak_idx: int) -> float:
        """Calculate peak prominence"""
        try:
            # Simple prominence calculation
            left_min = np.min(spectrum[max(0, peak_idx-10):peak_idx]) if peak_idx > 0 else spectrum[peak_idx]
            right_min = np.min(spectrum[peak_idx:min(len(spectrum), peak_idx+10)]) if peak_idx < len(spectrum)-1 else spectrum[peak_idx]
            baseline = max(left_min, right_min)
            return spectrum[peak_idx] - baseline
        except:
            return 0.0
    
    def _calculate_peak_area(self, spectrum: np.ndarray, peak_idx: int, width: float) -> float:
        """Calculate peak area"""
        try:
            half_width = int(width / 2)
            start = max(0, peak_idx - half_width)
            end = min(len(spectrum), peak_idx + half_width)
            
            # Simple baseline subtraction
            baseline = (spectrum[start] + spectrum[end-1]) / 2
            area = np.sum(spectrum[start:end] - baseline)
            return max(0, area)
        except:
            return 0.0
    
    def _estimate_peak_width(self, spectrum: np.ndarray, peak_idx: int) -> float:
        """Estimate peak width at half maximum"""
        try:
            half_max = spectrum[peak_idx] / 2
            
            # Find left side
            left_idx = peak_idx
            while left_idx > 0 and spectrum[left_idx] > half_max:
                left_idx -= 1
            
            # Find right side
            right_idx = peak_idx
            while right_idx < len(spectrum) - 1 and spectrum[right_idx] > half_max:
                right_idx += 1
            
            return right_idx - left_idx
        except:
            return 1.0
    
    def cluster_peaks(self, 
                     method: str = "dbscan",
                     **kwargs) -> List[PeakCluster]:
        """
        Cluster detected peaks based on wavelength proximity
        """
        try:
            if not self.peaks or not any(self.peaks):
                raise PeakDetectionError("No peaks detected yet")
            
            # Flatten all peaks
            all_peaks = []
            for spectrum_peaks in self.peaks:
                all_peaks.extend(spectrum_peaks)
            
            if len(all_peaks) < 2:
                self.clustered_peaks = [PeakCluster([peak]) for peak in all_peaks]
                return self.clustered_peaks
            
            # Extract wavelengths for clustering
            wavelengths = np.array([peak.wavelength for peak in all_peaks])
            
            if method == "dbscan":
                eps = kwargs.get('eps', self.settings.get_peak_detection_settings().clustering_tolerance)
                min_samples = kwargs.get('min_samples', self.settings.get_peak_detection_settings().min_cluster_size)
                
                clustering = DBSCAN(eps=eps, min_samples=min_samples)
                labels = clustering.fit_predict(wavelengths.reshape(-1, 1))
                
            elif method == "hierarchical":
                threshold = kwargs.get('threshold', self.settings.get_peak_detection_settings().clustering_tolerance)
                
                # Calculate linkage matrix
                distances = pdist(wavelengths.reshape(-1, 1))
                linkage_matrix = linkage(distances, method='ward')
                labels = fcluster(linkage_matrix, threshold, criterion='distance')
                labels -= 1  # Convert to 0-based indexing
                
            elif method == "kmeans":
                n_clusters = kwargs.get('n_clusters', min(10, len(all_peaks)//2))
                
                clustering = KMeans(n_clusters=n_clusters, random_state=42)
                labels = clustering.fit_predict(wavelengths.reshape(-1, 1))
                
            else:
                raise PeakDetectionError(f"Unknown clustering method: {method}")
            
            # Group peaks by cluster
            clusters = {}
            for peak, label in zip(all_peaks, labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(peak)
            
            # Create cluster objects
            self.clustered_peaks = []
            for cluster_peaks in clusters.values():
                if len(cluster_peaks) >= 1:  # Include single-peak clusters
                    self.clustered_peaks.append(PeakCluster(cluster_peaks))
            
            self.logger.info(f"Created {len(self.clustered_peaks)} peak clusters using {method}")
            return self.clustered_peaks
            
        except Exception as e:
            raise PeakDetectionError(f"Peak clustering failed: {e}")
    
    def filter_peaks(self, 
                    min_intensity: Optional[float] = None,
                    min_prominence: Optional[float] = None,
                    min_width: Optional[float] = None,
                    wavelength_range: Optional[Tuple[float, float]] = None) -> List[List[Peak]]:
        """Filter detected peaks based on criteria"""
        
        filtered_peaks = []
        
        for spectrum_peaks in self.peaks:
            filtered_spectrum_peaks = []
            
            for peak in spectrum_peaks:
                # Apply filters
                if min_intensity is not None and peak.intensity < min_intensity:
                    continue
                if min_prominence is not None and peak.prominence < min_prominence:
                    continue
                if min_width is not None and peak.width < min_width:
                    continue
                if wavelength_range is not None:
                    if peak.wavelength < wavelength_range[0] or peak.wavelength > wavelength_range[1]:
                        continue
                
                filtered_spectrum_peaks.append(peak)
            
            filtered_peaks.append(filtered_spectrum_peaks)
        
        self.peaks = filtered_peaks
        return self.peaks
    
    def get_peak_statistics(self) -> Dict[str, Any]:
        """Calculate statistics for detected peaks"""
        
        if not self.peaks:
            return {}
        
        all_peaks = []
        for spectrum_peaks in self.peaks:
            all_peaks.extend(spectrum_peaks)
        
        if not all_peaks:
            return {}
        
        wavelengths = [peak.wavelength for peak in all_peaks]
        intensities = [peak.intensity for peak in all_peaks]
        widths = [peak.width for peak in all_peaks]
        prominences = [peak.prominence for peak in all_peaks]
        
        stats = {
            'total_peaks': len(all_peaks),
            'peaks_per_spectrum': {
                'mean': np.mean([len(peaks) for peaks in self.peaks]),
                'std': np.std([len(peaks) for peaks in self.peaks]),
                'min': min(len(peaks) for peaks in self.peaks),
                'max': max(len(peaks) for peaks in self.peaks)
            },
            'wavelength_stats': {
                'mean': np.mean(wavelengths),
                'std': np.std(wavelengths),
                'min': min(wavelengths),
                'max': max(wavelengths)
            },
            'intensity_stats': {
                'mean': np.mean(intensities),
                'std': np.std(intensities),
                'min': min(intensities),
                'max': max(intensities)
            },
            'width_stats': {
                'mean': np.mean(widths),
                'std': np.std(widths),
                'min': min(widths),
                'max': max(widths)
            },
            'prominence_stats': {
                'mean': np.mean(prominences),
                'std': np.std(prominences),
                'min': min(prominences),
                'max': max(prominences)
            }
        }
        
        if self.clustered_peaks:
            cluster_sizes = [cluster.size for cluster in self.clustered_peaks]
            stats['cluster_stats'] = {
                'num_clusters': len(self.clustered_peaks),
                'cluster_size_mean': np.mean(cluster_sizes),
                'cluster_size_std': np.std(cluster_sizes),
                'largest_cluster': max(cluster_sizes),
                'smallest_cluster': min(cluster_sizes)
            }
        
        return stats
    
    def export_peaks(self) -> Dict[str, Any]:
        """Export peak detection results"""
        
        return {
            'peaks': self.peaks,
            'clustered_peaks': self.clustered_peaks,
            'detection_metadata': self.detection_metadata,
            'statistics': self.get_peak_statistics()
        }
    
    def clear_results(self):
        """Clear all detection results"""
        self.peaks = []
        self.clustered_peaks = []
        self.detection_metadata = {}
        self._cache.clear()