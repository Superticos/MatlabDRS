"""
Advanced plotting capabilities for DRS spectroscopy
Publication-quality plots with matplotlib and plotly backends
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.style as mplstyle
import seaborn as sns
from scipy import interpolate
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass

from ..config.settings import AppSettings

@dataclass
class PlotTheme:
    """Configuration for plot appearance"""
    background_color: str = 'white'
    figure_color: str = 'white'
    text_color: str = 'black'
    grid_color: str = '#CCCCCC'
    grid_alpha: float = 0.7
    line_width: float = 1.5
    marker_size: float = 6.0
    font_size: int = 12
    title_size: int = 14
    label_size: int = 12
    legend_size: int = 10
    dpi: int = 100
    figure_size: Tuple[float, float] = (10, 8)
    color_palette: str = 'tab10'
    style: str = 'default'

class DRSPlotter:
    """
    Advanced plotting class for DRS spectroscopy data
    Supports both matplotlib and plotly backends for different use cases
    """
    
    def __init__(self, settings: AppSettings = None, backend: str = 'matplotlib'):
        self.settings = settings or AppSettings()
        self.backend = backend
        self.logger = logging.getLogger(__name__)
        
        # Plot configuration
        self.theme = PlotTheme()
        self.setup_theme()
        
        # Color palettes
        self.color_palettes = {
            'default': plt.cm.tab10,
            'viridis': plt.cm.viridis,
            'plasma': plt.cm.plasma,
            'spectral': plt.cm.Spectral,
            'rainbow': plt.cm.rainbow,
            'custom_drs': self._create_custom_drs_palette()
        }
        
        # Current figure and axes
        self.current_fig = None
        self.current_axes = None
        
        # Plot data storage
        self.plot_data = {}
        
        self.logger.info(f"DRS Plotter initialized with {backend} backend")
    
    def setup_theme(self):
        """Setup plotting theme based on settings"""
        try:
            # Get theme from settings
            theme_name = self.settings.get('theme', 'default')
            
            if 'dark' in theme_name.lower():
                self.theme.background_color = '#2E2E2E'
                self.theme.figure_color = '#2E2E2E'
                self.theme.text_color = 'white'
                self.theme.grid_color = '#555555'
            else:
                self.theme.background_color = 'white'
                self.theme.figure_color = 'white'
                self.theme.text_color = 'black'
                self.theme.grid_color = '#CCCCCC'
            
            # Apply matplotlib style
            if self.backend == 'matplotlib':
                plt.style.use('seaborn-v0_8' if 'seaborn' in plt.style.available else 'default')
                
                # Custom rcParams
                plt.rcParams.update({
                    'figure.facecolor': self.theme.figure_color,
                    'axes.facecolor': self.theme.background_color,
                    'axes.edgecolor': self.theme.text_color,
                    'axes.labelcolor': self.theme.text_color,
                    'text.color': self.theme.text_color,
                    'xtick.color': self.theme.text_color,
                    'ytick.color': self.theme.text_color,
                    'grid.color': self.theme.grid_color,
                    'grid.alpha': self.theme.grid_alpha,
                    'lines.linewidth': self.theme.line_width,
                    'font.size': self.theme.font_size,
                    'figure.dpi': self.theme.dpi,
                    'savefig.dpi': self.theme.dpi * 2
                })
                
        except Exception as e:
            self.logger.warning(f"Failed to setup theme: {e}")
    
    def _create_custom_drs_palette(self):
        """Create custom color palette for DRS data"""
        colors = [
            '#1f77b4',  # Blue
            '#ff7f0e',  # Orange
            '#2ca02c',  # Green
            '#d62728',  # Red
            '#9467bd',  # Purple
            '#8c564b',  # Brown
            '#e377c2',  # Pink
            '#7f7f7f',  # Gray
            '#bcbd22',  # Olive
            '#17becf'   # Cyan
        ]
        return LinearSegmentedColormap.from_list('custom_drs', colors)
    
    def plot_spectra(self, 
                    spectra: np.ndarray, 
                    wavelengths: np.ndarray,
                    labels: Optional[List[str]] = None,
                    title: str = "Spectral Data",
                    normalize: bool = False,
                    offset: float = 0.0,
                    highlight_indices: Optional[List[int]] = None,
                    show_mean: bool = False,
                    show_std: bool = False,
                    interactive: bool = False) -> Union[Figure, go.Figure]:
        """
        Plot spectral data with various visualization options
        
        Args:
            spectra: 2D array of spectral data (n_spectra, n_wavelengths)
            wavelengths: 1D array of wavelength values
            labels: Optional labels for each spectrum
            title: Plot title
            normalize: Whether to normalize spectra
            offset: Vertical offset between spectra for stacked display
            highlight_indices: Indices of spectra to highlight
            show_mean: Whether to show mean spectrum
            show_std: Whether to show standard deviation band
            interactive: Whether to create interactive plotly plot
            
        Returns:
            Figure object (matplotlib or plotly)
        """
        try:
            if interactive and self.backend == 'matplotlib':
                return self._plot_spectra_plotly(
                    spectra, wavelengths, labels, title, normalize, 
                    offset, highlight_indices, show_mean, show_std
                )
            else:
                return self._plot_spectra_matplotlib(
                    spectra, wavelengths, labels, title, normalize,
                    offset, highlight_indices, show_mean, show_std
                )
                
        except Exception as e:
            self.logger.error(f"Failed to plot spectra: {e}")
            raise
    
    def _plot_spectra_matplotlib(self, 
                                spectra: np.ndarray, 
                                wavelengths: np.ndarray,
                                labels: Optional[List[str]] = None,
                                title: str = "Spectral Data",
                                normalize: bool = False,
                                offset: float = 0.0,
                                highlight_indices: Optional[List[int]] = None,
                                show_mean: bool = False,
                                show_std: bool = False) -> Figure:
        """Plot spectra using matplotlib"""
        
        # Prepare data
        plot_data = spectra.copy()
        if normalize:
            plot_data = self._normalize_spectra(plot_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.theme.figure_size)
        self.current_fig = fig
        self.current_axes = ax
        
        # Get colors
        colors = plt.cm.get_cmap(self.theme.color_palette)
        n_spectra = len(plot_data)
        
        # Plot individual spectra
        for i, spectrum in enumerate(plot_data):
            y_data = spectrum + (i * offset if offset > 0 else 0)
            
            # Determine line style and color
            if highlight_indices and i in highlight_indices:
                color = 'red'
                linewidth = self.theme.line_width * 1.5
                alpha = 1.0
                zorder = 10
            else:
                color = colors(i / max(n_spectra - 1, 1))
                linewidth = self.theme.line_width
                alpha = 0.7 if highlight_indices else 1.0
                zorder = 1
            
            label = labels[i] if labels else f'Spectrum {i+1}'
            
            ax.plot(wavelengths, y_data, 
                   color=color, linewidth=linewidth, alpha=alpha,
                   label=label, zorder=zorder)
        
        # Plot mean and std if requested
        if show_mean:
            mean_spectrum = np.mean(plot_data, axis=0)
            ax.plot(wavelengths, mean_spectrum, 
                   color='black', linewidth=self.theme.line_width * 2,
                   linestyle='--', label='Mean', zorder=15)
        
        if show_std:
            mean_spectrum = np.mean(plot_data, axis=0)
            std_spectrum = np.std(plot_data, axis=0)
            ax.fill_between(wavelengths, 
                          mean_spectrum - std_spectrum,
                          mean_spectrum + std_spectrum,
                          alpha=0.3, color='gray', label='±1 STD')
        
        # Customize plot
        ax.set_xlabel('Wavelength (nm)', fontsize=self.theme.label_size)
        ax.set_ylabel('Intensity', fontsize=self.theme.label_size)
        ax.set_title(title, fontsize=self.theme.title_size)
        ax.grid(True, alpha=self.theme.grid_alpha)
        
        # Add legend if not too many spectra
        if n_spectra <= 10 or highlight_indices or show_mean or show_std:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                     fontsize=self.theme.legend_size)
        
        plt.tight_layout()
        
        # Store plot data
        self.plot_data['spectra'] = {
            'wavelengths': wavelengths,
            'spectra': plot_data,
            'labels': labels,
            'title': title
        }
        
        return fig
    
    def _plot_spectra_plotly(self, 
                            spectra: np.ndarray, 
                            wavelengths: np.ndarray,
                            labels: Optional[List[str]] = None,
                            title: str = "Spectral Data",
                            normalize: bool = False,
                            offset: float = 0.0,
                            highlight_indices: Optional[List[int]] = None,
                            show_mean: bool = False,
                            show_std: bool = False) -> go.Figure:
        """Plot spectra using plotly for interactive visualization"""
        
        # Prepare data
        plot_data = spectra.copy()
        if normalize:
            plot_data = self._normalize_spectra(plot_data)
        
        # Create plotly figure
        fig = go.Figure()
        
        # Color scale
        colors = px.colors.qualitative.Set1
        n_spectra = len(plot_data)
        
        # Plot individual spectra
        for i, spectrum in enumerate(plot_data):
            y_data = spectrum + (i * offset if offset > 0 else 0)
            
            # Determine line style
            if highlight_indices and i in highlight_indices:
                line_color = 'red'
                line_width = 3
                opacity = 1.0
            else:
                line_color = colors[i % len(colors)]
                line_width = 2
                opacity = 0.7 if highlight_indices else 1.0
            
            label = labels[i] if labels else f'Spectrum {i+1}'
            
            fig.add_trace(go.Scatter(
                x=wavelengths,
                y=y_data,
                mode='lines',
                name=label,
                line=dict(color=line_color, width=line_width),
                opacity=opacity,
                hovertemplate=f'<b>{label}</b><br>' +
                             'Wavelength: %{x:.1f} nm<br>' +
                             'Intensity: %{y:.4f}<br>' +
                             '<extra></extra>'
            ))
        
        # Add mean and std if requested
        if show_mean:
            mean_spectrum = np.mean(plot_data, axis=0)
            fig.add_trace(go.Scatter(
                x=wavelengths,
                y=mean_spectrum,
                mode='lines',
                name='Mean',
                line=dict(color='black', width=3, dash='dash'),
                hovertemplate='<b>Mean Spectrum</b><br>' +
                             'Wavelength: %{x:.1f} nm<br>' +
                             'Intensity: %{y:.4f}<br>' +
                             '<extra></extra>'
            ))
        
        if show_std:
            mean_spectrum = np.mean(plot_data, axis=0)
            std_spectrum = np.std(plot_data, axis=0)
            
            # Add upper and lower bounds
            fig.add_trace(go.Scatter(
                x=np.concatenate([wavelengths, wavelengths[::-1]]),
                y=np.concatenate([mean_spectrum + std_spectrum, 
                                (mean_spectrum - std_spectrum)[::-1]]),
                fill='toself',
                fillcolor='rgba(128,128,128,0.3)',
                line=dict(color='rgba(255,255,255,0)'),
                name='±1 STD',
                showlegend=True,
                hoverinfo='skip'
            ))
        
        # Update layout
        fig.update_layout(
            title=dict(text=title, font=dict(size=self.theme.title_size)),
            xaxis=dict(
                title='Wavelength (nm)',
                titlefont=dict(size=self.theme.label_size),
                tickfont=dict(size=self.theme.font_size)
            ),
            yaxis=dict(
                title='Intensity',
                titlefont=dict(size=self.theme.label_size),
                tickfont=dict(size=self.theme.font_size)
            ),
            hovermode='x unified',
            template='plotly_white' if 'dark' not in self.theme.background_color else 'plotly_dark',
            showlegend=n_spectra <= 10 or highlight_indices or show_mean or show_std,
            width=self.theme.figure_size[0] * 100,
            height=self.theme.figure_size[1] * 100
        )
        
        return fig
    
    def plot_pca_results(self, 
                        pca_data: Dict[str, Any],
                        wavelengths: np.ndarray,
                        n_components: int = 3,
                        interactive: bool = False) -> Union[Figure, go.Figure]:
        """
        Plot PCA analysis results including scores, loadings, and variance
        
        Args:
            pca_data: Dictionary containing PCA results
            wavelengths: Wavelength array for loadings plot
            n_components: Number of components to plot
            interactive: Whether to create interactive plot
            
        Returns:
            Figure object with subplots
        """
        try:
            if interactive:
                return self._plot_pca_plotly(pca_data, wavelengths, n_components)
            else:
                return self._plot_pca_matplotlib(pca_data, wavelengths, n_components)
                
        except Exception as e:
            self.logger.error(f"Failed to plot PCA results: {e}")
            raise
    
    def _plot_pca_matplotlib(self, 
                            pca_data: Dict[str, Any],
                            wavelengths: np.ndarray,
                            n_components: int = 3) -> Figure:
        """Plot PCA results using matplotlib"""
        
        scores = pca_data['scores']
        loadings = pca_data.get('loadings')
        variance_ratio = pca_data['explained_variance_ratio']
        
        # Create subplots
        fig = plt.figure(figsize=(15, 10))
        
        # Scores plot (PC1 vs PC2)
        ax1 = plt.subplot(2, 3, 1)
        scatter = ax1.scatter(scores[:, 0], scores[:, 1], 
                            c=range(len(scores)), cmap='viridis', 
                            alpha=0.7, s=self.theme.marker_size * 10)
        ax1.set_xlabel(f'PC1 ({variance_ratio[0]:.1%})')
        ax1.set_ylabel(f'PC2 ({variance_ratio[1]:.1%})')
        ax1.set_title('PCA Scores Plot')
        ax1.grid(True, alpha=self.theme.grid_alpha)
        plt.colorbar(scatter, ax=ax1, label='Sample Index')
        
        # 3D scores plot if enough components
        if scores.shape[1] >= 3:
            ax2 = plt.subplot(2, 3, 2, projection='3d')
            scatter_3d = ax2.scatter(scores[:, 0], scores[:, 1], scores[:, 2],
                                   c=range(len(scores)), cmap='viridis',
                                   alpha=0.7, s=self.theme.marker_size * 10)
            ax2.set_xlabel(f'PC1 ({variance_ratio[0]:.1%})')
            ax2.set_ylabel(f'PC2 ({variance_ratio[1]:.1%})')
            ax2.set_zlabel(f'PC3 ({variance_ratio[2]:.1%})')
            ax2.set_title('3D PCA Scores')
        
        # Loadings plot
        if loadings is not None:
            ax3 = plt.subplot(2, 3, 3)
            for i in range(min(n_components, loadings.shape[0])):
                ax3.plot(wavelengths, loadings[i], 
                        label=f'PC{i+1} ({variance_ratio[i]:.1%})',
                        linewidth=self.theme.line_width)
            ax3.set_xlabel('Wavelength (nm)')
            ax3.set_ylabel('Loadings')
            ax3.set_title('PCA Loadings')
            ax3.legend()
            ax3.grid(True, alpha=self.theme.grid_alpha)
        
        # Explained variance plot
        ax4 = plt.subplot(2, 3, 4)
        x_vals = range(1, len(variance_ratio) + 1)
        bars = ax4.bar(x_vals, variance_ratio * 100, 
                      alpha=0.7, color='steelblue')
        ax4.set_xlabel('Principal Component')
        ax4.set_ylabel('Explained Variance (%)')
        ax4.set_title('Explained Variance Ratio')
        ax4.grid(True, alpha=self.theme.grid_alpha)
        
        # Add percentage labels on bars
        for bar, pct in zip(bars, variance_ratio * 100):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{pct:.1f}%', ha='center', va='bottom')
        
        # Cumulative variance plot
        ax5 = plt.subplot(2, 3, 5)
        cumulative_var = np.cumsum(variance_ratio) * 100
        ax5.plot(x_vals, cumulative_var, 'ro-', linewidth=self.theme.line_width)
        ax5.set_xlabel('Principal Component')
        ax5.set_ylabel('Cumulative Variance (%)')
        ax5.set_title('Cumulative Explained Variance')
        ax5.grid(True, alpha=self.theme.grid_alpha)
        
        # Add horizontal line at 95%
        ax5.axhline(y=95, color='red', linestyle='--', alpha=0.7, label='95%')
        ax5.legend()
        
        # Biplot (if 2D)
        if loadings is not None and scores.shape[1] >= 2:
            ax6 = plt.subplot(2, 3, 6)
            
            # Plot scores
            ax6.scatter(scores[:, 0], scores[:, 1], alpha=0.6, s=30)
            
            # Plot loadings vectors (scaled)
            scale_factor = 3.0
            for i in range(min(10, len(wavelengths))):  # Plot every nth loading
                idx = i * len(wavelengths) // 10
                if idx < len(wavelengths):
                    ax6.arrow(0, 0, 
                            loadings[0, idx] * scale_factor,
                            loadings[1, idx] * scale_factor,
                            head_width=0.1, head_length=0.1, 
                            fc='red', ec='red', alpha=0.7)
                    ax6.text(loadings[0, idx] * scale_factor * 1.1,
                           loadings[1, idx] * scale_factor * 1.1,
                           f'{wavelengths[idx]:.0f}',
                           fontsize=8, ha='center')
            
            ax6.set_xlabel(f'PC1 ({variance_ratio[0]:.1%})')
            ax6.set_ylabel(f'PC2 ({variance_ratio[1]:.1%})')
            ax6.set_title('PCA Biplot')
            ax6.grid(True, alpha=self.theme.grid_alpha)
        
        plt.tight_layout()
        self.current_fig = fig
        
        return fig
    
    def _plot_pca_plotly(self, 
                        pca_data: Dict[str, Any],
                        wavelengths: np.ndarray,
                        n_components: int = 3) -> go.Figure:
        """Plot PCA results using plotly for interactive visualization"""
        
        scores = pca_data['scores']
        loadings = pca_data.get('loadings')
        variance_ratio = pca_data['explained_variance_ratio']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('PCA Scores (2D)', 'PCA Scores (3D)', 'PCA Loadings',
                          'Explained Variance', 'Cumulative Variance', 'Biplot'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter3d'}, {'type': 'scatter'}],
                   [{'type': 'bar'}, {'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        # 2D Scores plot
        fig.add_trace(
            go.Scatter(
                x=scores[:, 0], y=scores[:, 1],
                mode='markers',
                marker=dict(
                    color=range(len(scores)),
                    colorscale='viridis',
                    size=8,
                    showscale=True,
                    colorbar=dict(title="Sample Index", x=0.35)
                ),
                text=[f'Sample {i+1}' for i in range(len(scores))],
                hovertemplate='<b>%{text}</b><br>' +
                             f'PC1: %{{x:.3f}}<br>' +
                             f'PC2: %{{y:.3f}}<br>' +
                             '<extra></extra>',
                name='Scores'
            ),
            row=1, col=1
        )
        
        # 3D Scores plot
        if scores.shape[1] >= 3:
            fig.add_trace(
                go.Scatter3d(
                    x=scores[:, 0], y=scores[:, 1], z=scores[:, 2],
                    mode='markers',
                    marker=dict(
                        color=range(len(scores)),
                        colorscale='viridis',
                        size=5
                    ),
                    text=[f'Sample {i+1}' for i in range(len(scores))],
                    hovertemplate='<b>%{text}</b><br>' +
                                 f'PC1: %{{x:.3f}}<br>' +
                                 f'PC2: %{{y:.3f}}<br>' +
                                 f'PC3: %{{z:.3f}}<br>' +
                                 '<extra></extra>',
                    name='3D Scores'
                ),
                row=1, col=2
            )
        
        # Loadings plot
        if loadings is not None:
            colors = px.colors.qualitative.Set1
            for i in range(min(n_components, loadings.shape[0])):
                fig.add_trace(
                    go.Scatter(
                        x=wavelengths, y=loadings[i],
                        mode='lines',
                        name=f'PC{i+1} ({variance_ratio[i]:.1%})',
                        line=dict(color=colors[i % len(colors)], width=2),
                        hovertemplate='<b>PC%{i+1}</b><br>' +
                                     'Wavelength: %{x:.1f} nm<br>' +
                                     'Loading: %{y:.4f}<br>' +
                                     '<extra></extra>'
                    ),
                    row=1, col=3
                )
        
        # Explained variance bar plot
        fig.add_trace(
            go.Bar(
                x=list(range(1, len(variance_ratio) + 1)),
                y=variance_ratio * 100,
                name='Explained Variance',
                marker=dict(color='steelblue'),
                text=[f'{v:.1f}%' for v in variance_ratio * 100],
                textposition='outside',
                hovertemplate='<b>PC%{x}</b><br>' +
                             'Explained Variance: %{y:.1f}%<br>' +
                             '<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Cumulative variance plot
        cumulative_var = np.cumsum(variance_ratio) * 100
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(variance_ratio) + 1)),
                y=cumulative_var,
                mode='lines+markers',
                name='Cumulative Variance',
                line=dict(color='red', width=3),
                marker=dict(size=8),
                hovertemplate='<b>Up to PC%{x}</b><br>' +
                             'Cumulative Variance: %{y:.1f}%<br>' +
                             '<extra></extra>'
            ),
            row=2, col=2
        )
        
        # Add 95% line
        fig.add_hline(y=95, line_dash="dash", line_color="red", 
                     row=2, col=2, annotation_text="95%")
        
        # Update layout
        fig.update_layout(
            title="PCA Analysis Results",
            showlegend=True,
            height=800,
            width=1400
        )
        
        # Update axis labels
        fig.update_xaxes(title_text=f"PC1 ({variance_ratio[0]:.1%})", row=1, col=1)
        fig.update_yaxes(title_text=f"PC2 ({variance_ratio[1]:.1%})", row=1, col=1)
        
        if scores.shape[1] >= 3:
            fig.update_scenes(
                xaxis_title=f"PC1 ({variance_ratio[0]:.1%})",
                yaxis_title=f"PC2 ({variance_ratio[1]:.1%})",
                zaxis_title=f"PC3 ({variance_ratio[2]:.1%})",
                row=1, col=2
            )
        
        fig.update_xaxes(title_text="Wavelength (nm)", row=1, col=3)
        fig.update_yaxes(title_text="Loadings", row=1, col=3)
        
        fig.update_xaxes(title_text="Principal Component", row=2, col=1)
        fig.update_yaxes(title_text="Explained Variance (%)", row=2, col=1)
        
        fig.update_xaxes(title_text="Principal Component", row=2, col=2)
        fig.update_yaxes(title_text="Cumulative Variance (%)", row=2, col=2)
        
        return fig
    
    def plot_peak_analysis(self, 
                          spectra: np.ndarray,
                          wavelengths: np.ndarray,
                          peaks: List[List],
                          labels: Optional[List[str]] = None,
                          highlight_peaks: bool = True,
                          show_peak_info: bool = True,
                          interactive: bool = False) -> Union[Figure, go.Figure]:
        """
        Plot spectral data with detected peaks highlighted
        
        Args:
            spectra: Spectral data array
            wavelengths: Wavelength array
            peaks: List of peak lists for each spectrum
            labels: Optional spectrum labels
            highlight_peaks: Whether to highlight peaks
            show_peak_info: Whether to show peak information
            interactive: Whether to create interactive plot
            
        Returns:
            Figure object with peak annotations
        """
        try:
            if interactive:
                return self._plot_peaks_plotly(
                    spectra, wavelengths, peaks, labels, 
                    highlight_peaks, show_peak_info
                )
            else:
                return self._plot_peaks_matplotlib(
                    spectra, wavelengths, peaks, labels,
                    highlight_peaks, show_peak_info
                )
                
        except Exception as e:
            self.logger.error(f"Failed to plot peak analysis: {e}")
            raise
    
    def _plot_peaks_matplotlib(self, 
                              spectra: np.ndarray,
                              wavelengths: np.ndarray,
                              peaks: List[List],
                              labels: Optional[List[str]] = None,
                              highlight_peaks: bool = True,
                              show_peak_info: bool = True) -> Figure:
        """Plot peak analysis using matplotlib"""
        
        fig, ax = plt.subplots(figsize=self.theme.figure_size)
        
        # Plot spectra
        colors = plt.cm.get_cmap(self.theme.color_palette)
        n_spectra = len(spectra)
        
        for i, spectrum in enumerate(spectra):
            color = colors(i / max(n_spectra - 1, 1))
            label = labels[i] if labels else f'Spectrum {i+1}'
            
            ax.plot(wavelengths, spectrum, color=color, 
                   linewidth=self.theme.line_width, label=label, alpha=0.8)
            
            # Plot peaks for this spectrum
            if i < len(peaks) and highlight_peaks:
                spectrum_peaks = peaks[i]
                
                for peak in spectrum_peaks:
                    # Peak marker
                    ax.plot(peak.wavelength, peak.intensity, 
                           'o', color=color, markersize=self.theme.marker_size * 1.5,
                           markeredgecolor='white', markeredgewidth=1)
                    
                    # Peak annotation
                    if show_peak_info:
                        ax.annotate(f'{peak.wavelength:.1f}',
                                  (peak.wavelength, peak.intensity),
                                  xytext=(5, 5), textcoords='offset points',
                                  fontsize=8, ha='left',
                                  bbox=dict(boxstyle='round,pad=0.2', 
                                          facecolor=color, alpha=0.7))
        
        # Customize plot
        ax.set_xlabel('Wavelength (nm)', fontsize=self.theme.label_size)
        ax.set_ylabel('Intensity', fontsize=self.theme.label_size)
        ax.set_title('Spectral Data with Detected Peaks', fontsize=self.theme.title_size)
        ax.grid(True, alpha=self.theme.grid_alpha)
        
        if n_spectra <= 10:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        self.current_fig = fig
        
        return fig
    
    def _plot_peaks_plotly(self, 
                          spectra: np.ndarray,
                          wavelengths: np.ndarray,
                          peaks: List[List],
                          labels: Optional[List[str]] = None,
                          highlight_peaks: bool = True,
                          show_peak_info: bool = True) -> go.Figure:
        """Plot peak analysis using plotly"""
        
        fig = go.Figure()
        colors = px.colors.qualitative.Set1
        
        # Plot spectra
        for i, spectrum in enumerate(spectra):
            color = colors[i % len(colors)]
            label = labels[i] if labels else f'Spectrum {i+1}'
            
            fig.add_trace(go.Scatter(
                x=wavelengths,
                y=spectrum,
                mode='lines',
                name=label,
                line=dict(color=color, width=2),
                hovertemplate=f'<b>{label}</b><br>' +
                             'Wavelength: %{x:.1f} nm<br>' +
                             'Intensity: %{y:.4f}<br>' +
                             '<extra></extra>'
            ))
            
            # Add peaks
            if i < len(peaks) and highlight_peaks:
                spectrum_peaks = peaks[i]
                
                if spectrum_peaks:
                    peak_wavelengths = [p.wavelength for p in spectrum_peaks]
                    peak_intensities = [p.intensity for p in spectrum_peaks]
                    peak_info = [f'Peak at {p.wavelength:.1f} nm<br>' +
                               f'Intensity: {p.intensity:.4f}<br>' +
                               f'Width: {p.width:.2f}' for p in spectrum_peaks]
                    
                    fig.add_trace(go.Scatter(
                        x=peak_wavelengths,
                        y=peak_intensities,
                        mode='markers',
                        name=f'{label} Peaks',
                        marker=dict(
                            color=color,
                            size=10,
                            symbol='circle',
                            line=dict(color='white', width=2)
                        ),
                        text=peak_info,
                        hovertemplate='<b>%{text}</b><extra></extra>'
                    ))
        
        # Update layout
        fig.update_layout(
            title='Spectral Data with Detected Peaks',
            xaxis_title='Wavelength (nm)',
            yaxis_title='Intensity',
            hovermode='x unified',
            template='plotly_white',
            showlegend=True,
            width=self.theme.figure_size[0] * 100,
            height=self.theme.figure_size[1] * 100
        )
        
        return fig
    
    def plot_correlation_matrix(self, 
                               correlation_matrix: np.ndarray,
                               labels: Optional[List[str]] = None,
                               title: str = "Correlation Matrix",
                               interactive: bool = False) -> Union[Figure, go.Figure]:
        """
        Plot correlation matrix as heatmap
        
        Args:
            correlation_matrix: 2D correlation matrix
            labels: Optional labels for axes
            title: Plot title
            interactive: Whether to create interactive plot
            
        Returns:
            Figure object with correlation heatmap
        """
        try:
            if interactive:
                return self._plot_correlation_plotly(correlation_matrix, labels, title)
            else:
                return self._plot_correlation_matplotlib(correlation_matrix, labels, title)
                
        except Exception as e:
            self.logger.error(f"Failed to plot correlation matrix: {e}")
            raise
    
    def _plot_correlation_matplotlib(self, 
                                    correlation_matrix: np.ndarray,
                                    labels: Optional[List[str]] = None,
                                    title: str = "Correlation Matrix") -> Figure:
        """Plot correlation matrix using matplotlib"""
        
        fig, ax = plt.subplots(figsize=self.theme.figure_size)
        
        # Create heatmap
        im = ax.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto',
                      vmin=-1, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation Coefficient', fontsize=self.theme.label_size)
        
        # Set ticks and labels
        if labels:
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_yticklabels(labels)
        
        # Add correlation values
        for i in range(correlation_matrix.shape[0]):
            for j in range(correlation_matrix.shape[1]):
                text = ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black" if abs(correlation_matrix[i, j]) < 0.5 else "white")
        
        ax.set_title(title, fontsize=self.theme.title_size)
        plt.tight_layout()
        self.current_fig = fig
        
        return fig
    
    def _plot_correlation_plotly(self, 
                                correlation_matrix: np.ndarray,
                                labels: Optional[List[str]] = None,
                                title: str = "Correlation Matrix") -> go.Figure:
        """Plot correlation matrix using plotly"""
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=labels,
            y=labels,
            colorscale='RdBu',
            zmid=0,
            text=[[f'{val:.2f}' for val in row] for row in correlation_matrix],
            texttemplate="%{text}",
            textfont={"size": 10},
            hovetemplate='<b>%{x} vs %{y}</b><br>' +
                        'Correlation: %{z:.3f}<br>' +
                        '<extra></extra>',
            colorbar=dict(title="Correlation Coefficient")
        ))
        
        fig.update_layout(
            title=title,
            xaxis_nticks=len(labels) if labels else 10,
            width=self.theme.figure_size[0] * 100,
            height=self.theme.figure_size[1] * 100,
            template='plotly_white'
        )
        
        return fig
    
    def _normalize_spectra(self, spectra: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """Normalize spectra for visualization"""
        if method == 'minmax':
            return (spectra - np.min(spectra, axis=1, keepdims=True)) / \
                   (np.max(spectra, axis=1, keepdims=True) - np.min(spectra, axis=1, keepdims=True) + 1e-8)
        elif method == 'standard':
            return (spectra - np.mean(spectra, axis=1, keepdims=True)) / \
                   (np.std(spectra, axis=1, keepdims=True) + 1e-8)
        else:
            return spectra
    
    def save_figure(self, 
                   filename: str, 
                   figure: Optional[Union[Figure, go.Figure]] = None,
                   dpi: int = 300,
                   format: str = 'png',
                   transparent: bool = False) -> bool:
        """
        Save figure to file
        
        Args:
            filename: Output filename
            figure: Figure to save (uses current if None)
            dpi: Resolution for raster formats
            format: Output format ('png', 'pdf', 'svg', 'html')
            transparent: Whether to use transparent background
            
        Returns:
            Success status
        """
        try:
            figure = figure or self.current_fig
            
            if figure is None:
                self.logger.error("No figure to save")
                return False
            
            if isinstance(figure, go.Figure):
                # Plotly figure
                if format.lower() == 'html':
                    figure.write_html(filename)
                else:
                    figure.write_image(filename, format=format, width=dpi*4, height=dpi*3)
            else:
                # Matplotlib figure
                figure.savefig(filename, dpi=dpi, format=format, 
                             transparent=transparent, bbox_inches='tight')
            
            self.logger.info(f"Figure saved to {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save figure: {e}")
            return False
    
    def create_dashboard(self, 
                        data: Dict[str, Any],
                        output_file: str = "drs_dashboard.html") -> str:
        """
        Create interactive dashboard with multiple plots
        
        Args:
            data: Dictionary containing analysis data
            output_file: Output HTML filename
            
        Returns:
            Path to created dashboard
        """
        try:
            # Create subplot layout
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('Raw Spectra', 'PCA Scores', 'Peak Analysis',
                              'Correlation Matrix', 'Statistics', 'Summary'),
                specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                       [{'type': 'scatter'}, {'type': 'heatmap'}],
                       [{'type': 'bar'}, {'type': 'table'}]]
            )
            
            # Add plots based on available data
            if 'spectra' in data:
                # Raw spectra
                spectra = data['spectra']
                wavelengths = data['wavelengths']
                
                for i, spectrum in enumerate(spectra[:10]):  # Limit for performance
                    fig.add_trace(
                        go.Scatter(
                            x=wavelengths, y=spectrum,
                            mode='lines', name=f'Spectrum {i+1}',
                            showlegend=False
                        ),
                        row=1, col=1
                    )
            
            if 'pca' in data:
                # PCA scores
                pca_data = data['pca']
                scores = pca_data['scores']
                
                fig.add_trace(
                    go.Scatter(
                        x=scores[:, 0], y=scores[:, 1],
                        mode='markers', name='PCA Scores',
                        marker=dict(color=range(len(scores)), colorscale='viridis'),
                        showlegend=False
                    ),
                    row=1, col=2
                )
            
            # Add more plots as needed...
            
            # Update layout
            fig.update_layout(
                title="DRS Analysis Dashboard",
                height=1200,
                showlegend=True
            )
            
            # Save dashboard
            fig.write_html(output_file)
            
            self.logger.info(f"Dashboard created: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Failed to create dashboard: {e}")
            return ""
    
    def set_theme(self, theme_name: str):
        """Set plotting theme"""
        if theme_name in ['dark', 'light', 'custom']:
            # Update theme settings
            self.setup_theme()
            self.logger.info(f"Theme changed to {theme_name}")
    
    def get_available_themes(self) -> List[str]:
        """Get list of available themes"""
        return ['default', 'dark', 'light', 'seaborn', 'ggplot', 'bmh']