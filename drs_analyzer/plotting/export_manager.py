"""
Export and animation management for DRS spectroscopy plots
Handles various export formats and creates animations for time-series data
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_pdf import PdfPages
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.offline as pyo

import cv2
import imageio
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import logging
from datetime import datetime
import json
import zipfile
import tempfile
import os

from ..config.settings import AppSettings

class ExportManager:
    """
    Comprehensive export manager for DRS analysis results
    Supports multiple formats and batch operations
    """
    
    def __init__(self, settings: AppSettings = None):
        self.settings = settings or AppSettings()
        self.logger = logging.getLogger(__name__)
        
        # Export configuration
        self.default_dpi = 300
        self.default_format = 'png'
        self.temp_dir = Path(tempfile.gettempdir()) / 'drs_exports'
        self.temp_dir.mkdir(exist_ok=True)
        
        # Supported formats
        self.supported_formats = {
            'image': ['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'svg', 'pdf'],
            'interactive': ['html', 'json'],
            'data': ['csv', 'xlsx', 'h5', 'mat', 'json'],
            'report': ['html', 'pdf', 'docx'],
            'animation': ['gif', 'mp4', 'avi']
        }
        
        # Initialize plotly settings
        self._setup_plotly_export()
        
        self.logger.info("Export Manager initialized")
    
    def _setup_plotly_export(self):
        """Setup plotly export configuration"""
        try:
            # Set default image export engine
            pio.kaleido.scope.default_format = "png"
            pio.kaleido.scope.default_width = 1200
            pio.kaleido.scope.default_height = 800
            
        except Exception as e:
            self.logger.warning(f"Failed to setup plotly export: {e}")
    
    def export_figure(self, 
                     figure,
                     filename: str,
                     format: str = None,
                     dpi: int = None,
                     width: int = None,
                     height: int = None,
                     transparent: bool = False,
                     metadata: Dict[str, Any] = None) -> bool:
        """
        Export figure to file with specified format and options
        
        Args:
            figure: Matplotlib or Plotly figure object
            filename: Output filename (with or without extension)
            format: Export format (auto-detected if None)
            dpi: Resolution for raster formats
            width: Width in pixels (for interactive formats)
            height: Height in pixels (for interactive formats)
            transparent: Use transparent background
            metadata: Additional metadata to include
            
        Returns:
            Success status
        """
        try:
            # Auto-detect format from filename
            file_path = Path(filename)
            if format is None:
                format = file_path.suffix.lower().lstrip('.')
            
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Set defaults
            dpi = dpi or self.default_dpi
            
            # Determine figure type and export accordingly
            if hasattr(figure, 'write_image'):  # Plotly figure
                return self._export_plotly_figure(
                    figure, file_path, format, width, height, metadata
                )
            else:  # Matplotlib figure
                return self._export_matplotlib_figure(
                    figure, file_path, format, dpi, transparent, metadata
                )
                
        except Exception as e:
            self.logger.error(f"Failed to export figure: {e}")
            return False
    
    def _export_matplotlib_figure(self, 
                                 figure,
                                 file_path: Path,
                                 format: str,
                                 dpi: int,
                                 transparent: bool,
                                 metadata: Dict[str, Any] = None) -> bool:
        """Export matplotlib figure"""
        try:
            # Prepare save arguments
            save_kwargs = {
                'dpi': dpi,
                'format': format,
                'bbox_inches': 'tight',
                'transparent': transparent,
                'facecolor': 'white' if not transparent else 'none'
            }
            
            # Add metadata if supported
            if metadata and format in ['png', 'pdf']:
                if format == 'png':
                    save_kwargs['metadata'] = metadata
                elif format == 'pdf':
                    save_kwargs['metadata'] = metadata
            
            # Save figure
            figure.savefig(str(file_path), **save_kwargs)
            
            self.logger.info(f"Matplotlib figure exported to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export matplotlib figure: {e}")
            return False
    
    def _export_plotly_figure(self, 
                             figure,
                             file_path: Path,
                             format: str,
                             width: int = None,
                             height: int = None,
                             metadata: Dict[str, Any] = None) -> bool:
        """Export plotly figure"""
        try:
            width = width or 1200
            height = height or 800
            
            if format == 'html':
                # Export as interactive HTML
                config = {
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
                }
                
                figure.write_html(
                    str(file_path),
                    include_plotlyjs=True,
                    config=config,
                    div_id="drs_plot"
                )
                
            elif format == 'json':
                # Export as JSON
                figure.write_json(str(file_path))
                
            else:
                # Export as static image
                figure.write_image(
                    str(file_path),
                    format=format,
                    width=width,
                    height=height,
                    scale=2  # For higher quality
                )
            
            self.logger.info(f"Plotly figure exported to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export plotly figure: {e}")
            return False
    
    def export_data(self, 
                   data: Dict[str, Any],
                   filename: str,
                   format: str = None,
                   include_metadata: bool = True) -> bool:
        """
        Export analysis data to various formats
        
        Args:
            data: Dictionary containing analysis data
            filename: Output filename
            format: Export format (csv, xlsx, h5, mat, json)
            include_metadata: Whether to include metadata
            
        Returns:
            Success status
        """
        try:
            file_path = Path(filename)
            if format is None:
                format = file_path.suffix.lower().lstrip('.')
            
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format == 'csv':
                return self._export_csv(data, file_path, include_metadata)
            elif format == 'xlsx':
                return self._export_excel(data, file_path, include_metadata)
            elif format == 'h5':
                return self._export_hdf5(data, file_path, include_metadata)
            elif format == 'mat':
                return self._export_matlab(data, file_path, include_metadata)
            elif format == 'json':
                return self._export_json(data, file_path, include_metadata)
            else:
                self.logger.error(f"Unsupported data format: {format}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to export data: {e}")
            return False
    
    def _export_csv(self, data: Dict[str, Any], file_path: Path, include_metadata: bool) -> bool:
        """Export data to CSV format"""
        import pandas as pd
        
        try:
            # Create main data directory
            data_dir = file_path.parent / file_path.stem
            data_dir.mkdir(exist_ok=True)
            
            # Export spectral data
            if 'spectra' in data and 'wavelengths' in data:
                spectra_df = pd.DataFrame(
                    data['spectra'].T,
                    index=data['wavelengths'],
                    columns=[f'Spectrum_{i+1}' for i in range(data['spectra'].shape[0])]
                )
                spectra_df.index.name = 'Wavelength_nm'
                spectra_df.to_csv(data_dir / 'spectra.csv')
            
            # Export PCA results
            if 'pca' in data:
                pca_data = data['pca']
                
                # Scores
                if 'scores' in pca_data:
                    scores_df = pd.DataFrame(
                        pca_data['scores'],
                        columns=[f'PC{i+1}' for i in range(pca_data['scores'].shape[1])]
                    )
                    scores_df.to_csv(data_dir / 'pca_scores.csv', index_label='Sample')
                
                # Loadings
                if 'loadings' in pca_data and 'wavelengths' in data:
                    loadings_df = pd.DataFrame(
                        pca_data['loadings'].T,
                        index=data['wavelengths'],
                        columns=[f'PC{i+1}' for i in range(pca_data['loadings'].shape[0])]
                    )
                    loadings_df.index.name = 'Wavelength_nm'
                    loadings_df.to_csv(data_dir / 'pca_loadings.csv')
                
                # Explained variance
                variance_df = pd.DataFrame({
                    'Component': [f'PC{i+1}' for i in range(len(pca_data['explained_variance_ratio']))],
                    'Explained_Variance_Ratio': pca_data['explained_variance_ratio'],
                    'Cumulative_Variance': np.cumsum(pca_data['explained_variance_ratio'])
                })
                variance_df.to_csv(data_dir / 'pca_variance.csv', index=False)
            
            # Export peak data
            if 'peaks' in data:
                peaks_data = []
                for spectrum_idx, spectrum_peaks in enumerate(data['peaks']):
                    for peak_idx, peak in enumerate(spectrum_peaks):
                        peaks_data.append({
                            'Spectrum': spectrum_idx + 1,
                            'Peak': peak_idx + 1,
                            'Wavelength': peak.wavelength,
                            'Intensity': peak.intensity,
                            'Width': getattr(peak, 'width', None),
                            'Height': getattr(peak, 'height', None),
                            'Prominence': getattr(peak, 'prominence', None)
                        })
                
                if peaks_data:
                    peaks_df = pd.DataFrame(peaks_data)
                    peaks_df.to_csv(data_dir / 'peaks.csv', index=False)
            
            # Export metadata
            if include_metadata and 'metadata' in data:
                metadata_df = pd.DataFrame([data['metadata']])
                metadata_df.to_csv(data_dir / 'metadata.csv', index=False)
            
            self.logger.info(f"Data exported to CSV in {data_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export CSV: {e}")
            return False
    
    def _export_excel(self, data: Dict[str, Any], file_path: Path, include_metadata: bool) -> bool:
        """Export data to Excel format"""
        import pandas as pd
        
        try:
            with pd.ExcelWriter(str(file_path), engine='openpyxl') as writer:
                # Export spectral data
                if 'spectra' in data and 'wavelengths' in data:
                    spectra_df = pd.DataFrame(
                        data['spectra'].T,
                        index=data['wavelengths'],
                        columns=[f'Spectrum_{i+1}' for i in range(data['spectra'].shape[0])]
                    )
                    spectra_df.index.name = 'Wavelength_nm'
                    spectra_df.to_excel(writer, sheet_name='Spectra')
                
                # Export PCA results
                if 'pca' in data:
                    pca_data = data['pca']
                    
                    if 'scores' in pca_data:
                        scores_df = pd.DataFrame(
                            pca_data['scores'],
                            columns=[f'PC{i+1}' for i in range(pca_data['scores'].shape[1])]
                        )
                        scores_df.to_excel(writer, sheet_name='PCA_Scores', index_label='Sample')
                    
                    if 'loadings' in pca_data and 'wavelengths' in data:
                        loadings_df = pd.DataFrame(
                            pca_data['loadings'].T,
                            index=data['wavelengths'],
                            columns=[f'PC{i+1}' for i in range(pca_data['loadings'].shape[0])]
                        )
                        loadings_df.index.name = 'Wavelength_nm'
                        loadings_df.to_excel(writer, sheet_name='PCA_Loadings')
                
                # Export clustering results
                if 'clustering' in data:
                    clustering_data = data['clustering']
                    if 'labels' in clustering_data:
                        cluster_df = pd.DataFrame({
                            'Sample': range(1, len(clustering_data['labels']) + 1),
                            'Cluster': clustering_data['labels']
                        })
                        cluster_df.to_excel(writer, sheet_name='Clustering', index=False)
                
                # Export peak data
                if 'peaks' in data:
                    peaks_data = []
                    for spectrum_idx, spectrum_peaks in enumerate(data['peaks']):
                        for peak_idx, peak in enumerate(spectrum_peaks):
                            peaks_data.append({
                                'Spectrum': spectrum_idx + 1,
                                'Peak': peak_idx + 1,
                                'Wavelength': peak.wavelength,
                                'Intensity': peak.intensity,
                                'Width': getattr(peak, 'width', None),
                                'Height': getattr(peak, 'height', None),
                                'Prominence': getattr(peak, 'prominence', None)
                            })
                    
                    if peaks_data:
                        peaks_df = pd.DataFrame(peaks_data)
                        peaks_df.to_excel(writer, sheet_name='Peaks', index=False)
                
                # Export metadata
                if include_metadata and 'metadata' in data:
                    metadata_df = pd.DataFrame([data['metadata']])
                    metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
            
            self.logger.info(f"Data exported to Excel: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export Excel: {e}")
            return False
    
    def _export_hdf5(self, data: Dict[str, Any], file_path: Path, include_metadata: bool) -> bool:
        """Export data to HDF5 format"""
        import h5py
        
        try:
            with h5py.File(str(file_path), 'w') as f:
                # Export spectral data
                if 'spectra' in data and 'wavelengths' in data:
                    spectra_group = f.create_group('spectra')
                    spectra_group.create_dataset('data', data=data['spectra'])
                    spectra_group.create_dataset('wavelengths', data=data['wavelengths'])
                
                # Export PCA results
                if 'pca' in data:
                    pca_group = f.create_group('pca')
                    pca_data = data['pca']
                    
                    for key, value in pca_data.items():
                        if isinstance(value, np.ndarray):
                            pca_group.create_dataset(key, data=value)
                        else:
                            pca_group.attrs[key] = value
                
                # Export clustering results
                if 'clustering' in data:
                    clustering_group = f.create_group('clustering')
                    clustering_data = data['clustering']
                    
                    for key, value in clustering_data.items():
                        if isinstance(value, np.ndarray):
                            clustering_group.create_dataset(key, data=value)
                        elif isinstance(value, (list, tuple)):
                            clustering_group.create_dataset(key, data=np.array(value))
                        else:
                            clustering_group.attrs[key] = value
                
                # Export metadata
                if include_metadata and 'metadata' in data:
                    metadata_group = f.create_group('metadata')
                    for key, value in data['metadata'].items():
                        if isinstance(value, (str, int, float)):
                            metadata_group.attrs[key] = value
            
            self.logger.info(f"Data exported to HDF5: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export HDF5: {e}")
            return False
    
    def _export_matlab(self, data: Dict[str, Any], file_path: Path, include_metadata: bool) -> bool:
        """Export data to MATLAB format"""
        from scipy.io import savemat
        
        try:
            # Prepare data for MATLAB
            matlab_data = {}
            
            # Spectral data
            if 'spectra' in data and 'wavelengths' in data:
                matlab_data['spectra'] = data['spectra']
                matlab_data['wavelengths'] = data['wavelengths']
            
            # PCA results
            if 'pca' in data:
                pca_data = data['pca']
                for key, value in pca_data.items():
                    if isinstance(value, np.ndarray):
                        matlab_data[f'pca_{key}'] = value
            
            # Clustering results
            if 'clustering' in data:
                clustering_data = data['clustering']
                for key, value in clustering_data.items():
                    if isinstance(value, np.ndarray):
                        matlab_data[f'clustering_{key}'] = value
                    elif isinstance(value, list):
                        matlab_data[f'clustering_{key}'] = np.array(value)
            
            # Save to MATLAB file
            savemat(str(file_path), matlab_data, oned_as='row')
            
            self.logger.info(f"Data exported to MATLAB: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export MATLAB: {e}")
            return False
    
    def _export_json(self, data: Dict[str, Any], file_path: Path, include_metadata: bool) -> bool:
        """Export data to JSON format"""
        try:
            # Convert numpy arrays to lists for JSON serialization
            json_data = self._prepare_json_data(data)
            
            with open(file_path, 'w') as f:
                json.dump(json_data, f, indent=2, default=self._json_serializer)
            
            self.logger.info(f"Data exported to JSON: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export JSON: {e}")
            return False
    
    def _prepare_json_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for JSON serialization"""
        json_data = {}
        
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                json_data[key] = value.tolist()
            elif isinstance(value, dict):
                json_data[key] = self._prepare_json_data(value)
            elif hasattr(value, '__dict__'):
                # Convert objects to dictionaries
                json_data[key] = value.__dict__
            else:
                json_data[key] = value
        
        return json_data
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return str(obj)
    
    def create_report(self, 
                     data: Dict[str, Any],
                     figures: List[Union[plt.Figure, go.Figure]],
                     filename: str,
                     format: str = 'html',
                     template: str = 'default') -> bool:
        """
        Create comprehensive analysis report
        
        Args:
            data: Analysis data
            figures: List of figures to include
            filename: Output filename
            format: Report format (html, pdf)
            template: Report template
            
        Returns:
            Success status
        """
        try:
            file_path = Path(filename)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == 'html':
                return self._create_html_report(data, figures, file_path, template)
            elif format.lower() == 'pdf':
                return self._create_pdf_report(data, figures, file_path, template)
            else:
                self.logger.error(f"Unsupported report format: {format}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to create report: {e}")
            return False
    
    def _create_html_report(self, 
                           data: Dict[str, Any],
                           figures: List,
                           file_path: Path,
                           template: str) -> bool:
        """Create HTML report"""
        try:
            html_content = self._generate_html_content(data, figures, template)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"HTML report created: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create HTML report: {e}")
            return False
    
    def _generate_html_content(self, 
                              data: Dict[str, Any],
                              figures: List,
                              template: str) -> str:
        """Generate HTML content for report"""
        
        # Base HTML template
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>DRS Analysis Report</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    background-color: #f8f9fa;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }
                h1 {
                    color: #2c3e50;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }
                h2 {
                    color: #34495e;
                    border-bottom: 1px solid #bdc3c7;
                    padding-bottom: 5px;
                }
                .summary {
                    background-color: #ecf0f1;
                    padding: 20px;
                    border-radius: 5px;
                    margin: 20px 0;
                }
                .figure {
                    text-align: center;
                    margin: 30px 0;
                    padding: 20px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }
                .metadata {
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-left: 4px solid #3498db;
                    margin: 20px 0;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }
                th, td {
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }
                th {
                    background-color: #f2f2f2;
                    font-weight: bold;
                }
                .timestamp {
                    color: #7f8c8d;
                    font-style: italic;
                }
            </style>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <div class="container">
                {content}
            </div>
        </body>
        </html>
        """
        
        # Generate content sections
        content_sections = []
        
        # Header
        content_sections.append(f"""
        <h1>🔬 DRS Spectroscopy Analysis Report</h1>
        <p class="timestamp">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        """)
        
        # Summary section
        if 'summary' in data or 'metadata' in data:
            content_sections.append(self._generate_summary_section(data))
        
        # Data overview
        content_sections.append(self._generate_data_overview(data))
        
        # Figures section
        content_sections.append(self._generate_figures_section(figures))
        
        # Results sections
        if 'pca' in data:
            content_sections.append(self._generate_pca_section(data['pca']))
        
        if 'clustering' in data:
            content_sections.append(self._generate_clustering_section(data['clustering']))
        
        if 'peaks' in data:
            content_sections.append(self._generate_peaks_section(data['peaks']))
        
        # Footer
        content_sections.append("""
        <hr>
        <p class="timestamp">Report generated by DRS Analyzer</p>
        """)
        
        # Combine all sections
        content = '\n'.join(content_sections)
        
        return html_template.format(content=content)
    
    def _generate_summary_section(self, data: Dict[str, Any]) -> str:
        """Generate summary section of HTML report"""
        summary_html = '<h2>📊 Analysis Summary</h2><div class="summary">'
        
        if 'metadata' in data:
            metadata = data['metadata']
            summary_html += f"""
            <h3>Dataset Information</h3>
            <ul>
                <li><strong>Number of Spectra:</strong> {metadata.get('n_spectra', 'N/A')}</li>
                <li><strong>Wavelength Points:</strong> {metadata.get('n_wavelengths', 'N/A')}</li>
                <li><strong>Wavelength Range:</strong> {metadata.get('wavelength_range', 'N/A')}</li>
                <li><strong>Analysis Date:</strong> {metadata.get('analysis_date', 'N/A')}</li>
            </ul>
            """
        
        if 'processing_info' in data:
            proc_info = data['processing_info']
            summary_html += f"""
            <h3>Processing Applied</h3>
            <ul>
                <li><strong>Baseline Correction:</strong> {proc_info.get('baseline', 'None')}</li>
                <li><strong>Smoothing:</strong> {proc_info.get('smoothing', 'None')}</li>
                <li><strong>Normalization:</strong> {proc_info.get('normalization', 'None')}</li>
            </ul>
            """
        
        summary_html += '</div>'
        return summary_html
    
    def _generate_data_overview(self, data: Dict[str, Any]) -> str:
        """Generate data overview section"""
        overview_html = '<h2>📈 Data Overview</h2>'
        
        if 'spectra' in data:
            spectra = data['spectra']
            overview_html += f"""
            <div class="metadata">
                <h3>Spectral Data Statistics</h3>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Number of Spectra</td><td>{spectra.shape[0]}</td></tr>
                    <tr><td>Spectral Points</td><td>{spectra.shape[1]}</td></tr>
                    <tr><td>Mean Intensity</td><td>{np.mean(spectra):.4f}</td></tr>
                    <tr><td>Std Deviation</td><td>{np.std(spectra):.4f}</td></tr>
                    <tr><td>Min Intensity</td><td>{np.min(spectra):.4f}</td></tr>
                    <tr><td>Max Intensity</td><td>{np.max(spectra):.4f}</td></tr>
                </table>
            </div>
            """
        
        return overview_html
    
    def _generate_figures_section(self, figures: List) -> str:
        """Generate figures section of HTML report"""
        figures_html = '<h2>📊 Analysis Plots</h2>'
        
        for i, figure in enumerate(figures):
            figures_html += f'<div class="figure">'
            figures_html += f'<h3>Figure {i+1}</h3>'
            
            if hasattr(figure, 'to_html'):  # Plotly figure
                # Convert plotly figure to HTML
                figure_html = figure.to_html(
                    include_plotlyjs=False,
                    div_id=f"plotly_div_{i}"
                )
                figures_html += figure_html
            else:  # Matplotlib figure
                # Convert matplotlib figure to base64 image
                img_data = self._figure_to_base64(figure)
                if img_data:
                    figures_html += f'<img src="data:image/png;base64,{img_data}" style="max-width: 100%; height: auto;">'
            
            figures_html += '</div>'
        
        return figures_html
    
    def _figure_to_base64(self, figure) -> str:
        """Convert matplotlib figure to base64 string"""
        try:
            buffer = io.BytesIO()
            figure.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            img_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            buffer.close()
            return img_data
        except Exception as e:
            self.logger.error(f"Failed to convert figure to base64: {e}")
            return ""
    
    def _generate_pca_section(self, pca_data: Dict[str, Any]) -> str:
        """Generate PCA results section"""
        pca_html = '<h2>🔍 Principal Component Analysis</h2>'
        
        variance_ratio = pca_data.get('explained_variance_ratio', [])
        
        pca_html += f"""
        <div class="metadata">
            <h3>PCA Results Summary</h3>
            <table>
                <tr><th>Component</th><th>Explained Variance (%)</th><th>Cumulative (%)</th></tr>
        """
        
        cumulative = 0
        for i, var in enumerate(variance_ratio[:5]):  # Show first 5 components
            cumulative += var
            pca_html += f"""
                <tr>
                    <td>PC{i+1}</td>
                    <td>{var*100:.2f}%</td>
                    <td>{cumulative*100:.2f}%</td>
                </tr>
            """
        
        pca_html += """
            </table>
        </div>
        """
        
        return pca_html
    
    def _generate_clustering_section(self, clustering_data: Dict[str, Any]) -> str:
        """Generate clustering results section"""
        clustering_html = '<h2>🎯 Clustering Analysis</h2>'
        
        method = clustering_data.get('method', 'Unknown')
        labels = clustering_data.get('labels', [])
        
        if labels:
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels)
            
            clustering_html += f"""
            <div class="metadata">
                <h3>Clustering Results</h3>
                <p><strong>Method:</strong> {method.title()}</p>
                <p><strong>Number of Clusters:</strong> {n_clusters}</p>
                
                <table>
                    <tr><th>Cluster</th><th>Size</th><th>Percentage</th></tr>
            """
            
            for label in unique_labels:
                count = np.sum(labels == label)
                percentage = count / len(labels) * 100
                cluster_name = f"Cluster {label}" if label >= 0 else "Noise"
                
                clustering_html += f"""
                    <tr>
                        <td>{cluster_name}</td>
                        <td>{count}</td>
                        <td>{percentage:.1f}%</td>
                    </tr>
                """
            
            clustering_html += """
                </table>
            </div>
            """
        
        return clustering_html
    
    def _generate_peaks_section(self, peaks_data: List[List]) -> str:
        """Generate peaks analysis section"""
        peaks_html = '<h2>⛰️ Peak Analysis</h2>'
        
        total_peaks = sum(len(spectrum_peaks) for spectrum_peaks in peaks_data)
        avg_peaks = total_peaks / len(peaks_data) if peaks_data else 0
        
        peaks_html += f"""
        <div class="metadata">
            <h3>Peak Detection Summary</h3>
            <p><strong>Total Peaks Detected:</strong> {total_peaks}</p>
            <p><strong>Average Peaks per Spectrum:</strong> {avg_peaks:.1f}</p>
            <p><strong>Spectra with Peaks:</strong> {sum(1 for p in peaks_data if p)}</p>
        </div>
        """
        
        return peaks_html
    
    def export_batch(self, 
                    data_list: List[Dict[str, Any]],
                    output_dir: str,
                    formats: List[str],
                    prefix: str = "analysis") -> bool:
        """
        Export multiple analyses in batch
        
        Args:
            data_list: List of analysis data dictionaries
            output_dir: Output directory
            formats: List of formats to export
            prefix: Filename prefix
            
        Returns:
            Success status
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            success_count = 0
            
            for i, data in enumerate(data_list):
                for format in formats:
                    filename = output_path / f"{prefix}_{i+1:03d}.{format}"
                    
                    if format in self.supported_formats['data']:
                        if self.export_data(data, str(filename), format):
                            success_count += 1
                    # Add more format types as needed
            
            self.logger.info(f"Batch export completed: {success_count} files exported")
            return success_count > 0
            
        except Exception as e:
            self.logger.error(f"Failed batch export: {e}")
            return False
    
    def create_archive(self, 
                      files: List[str],
                      archive_path: str,
                      format: str = 'zip') -> bool:
        """
        Create archive of exported files
        
        Args:
            files: List of file paths to archive
            archive_path: Output archive path
            format: Archive format ('zip', 'tar')
            
        Returns:
            Success status
        """
        try:
            if format.lower() == 'zip':
                with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file_path in files:
                        if Path(file_path).exists():
                            zipf.write(file_path, Path(file_path).name)
            else:
                import tarfile
                with tarfile.open(archive_path, 'w:gz') as tarf:
                    for file_path in files:
                        if Path(file_path).exists():
                            tarf.add(file_path, Path(file_path).name)
            
            self.logger.info(f"Archive created: {archive_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create archive: {e}")
            return False

class AnimationCreator:
    """
    Create animations for time-series spectral data
    """
    
    def __init__(self, settings: AppSettings = None):
        self.settings = settings or AppSettings()
        self.logger = logging.getLogger(__name__)
        
        # Animation parameters
        self.default_fps = 10
        self.default_duration = 5.0  # seconds
        self.temp_dir = Path(tempfile.gettempdir()) / 'drs_animations'
        self.temp_dir.mkdir(exist_ok=True)
    
    def create_spectral_animation(self, 
                                 spectra_sequence: List[np.ndarray],
                                 wavelengths: np.ndarray,
                                 output_path: str,
                                 fps: int = None,
                                 duration: float = None,
                                 titles: List[str] = None) -> bool:
        """
        Create animation of spectral evolution
        
        Args:
            spectra_sequence: List of spectral data arrays
            wavelengths: Wavelength array
            output_path: Output file path
            fps: Frames per second
            duration: Total duration in seconds
            titles: Optional titles for each frame
            
        Returns:
            Success status
        """
        try:
            fps = fps or self.default_fps
            duration = duration or self.default_duration
            
            # Calculate frame interval
            n_frames = len(spectra_sequence)
            interval = (duration * 1000) / n_frames  # milliseconds
            
            # Create matplotlib animation
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Setup plot limits
            all_data = np.concatenate(spectra_sequence)
            y_min, y_max = np.min(all_data), np.max(all_data)
            ax.set_xlim(wavelengths.min(), wavelengths.max())
            ax.set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))
            
            line, = ax.plot([], [], 'b-', linewidth=2)
            title_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
            
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('Intensity')
            ax.grid(True, alpha=0.3)
            
            def animate(frame):
                line.set_data(wavelengths, spectra_sequence[frame])
                if titles:
                    title_text.set_text(titles[frame])
                else:
                    title_text.set_text(f'Frame {frame + 1}/{n_frames}')
                return line, title_text
            
            # Create animation
            anim = animation.FuncAnimation(
                fig, animate, frames=n_frames,
                interval=interval, blit=True, repeat=True
            )
            
            # Save animation
            file_ext = Path(output_path).suffix.lower()
            if file_ext == '.gif':
                anim.save(output_path, writer='pillow', fps=fps)
            elif file_ext in ['.mp4', '.avi']:
                anim.save(output_path, writer='ffmpeg', fps=fps)
            else:
                self.logger.error(f"Unsupported animation format: {file_ext}")
                return False
            
            plt.close(fig)
            
            self.logger.info(f"Spectral animation created: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create spectral animation: {e}")
            return False
    
    def create_pca_evolution_animation(self, 
                                     pca_sequence: List[Dict[str, Any]],
                                     output_path: str,
                                     fps: int = None) -> bool:
        """
        Create animation showing PCA evolution
        
        Args:
            pca_sequence: List of PCA results over time
            output_path: Output file path
            fps: Frames per second
            
        Returns:
            Success status
        """
        try:
            fps = fps or self.default_fps
            
            # Create frames
            frames = []
            for i, pca_data in enumerate(pca_sequence):
                scores = pca_data['scores']
                
                fig, ax = plt.subplots(figsize=(8, 6))
                scatter = ax.scatter(scores[:, 0], scores[:, 1], 
                                   c=range(len(scores)), cmap='viridis', alpha=0.7)
                
                ax.set_xlabel(f"PC1 ({pca_data['explained_variance_ratio'][0]:.1%})")
                ax.set_ylabel(f"PC2 ({pca_data['explained_variance_ratio'][1]:.1%})")
                ax.set_title(f'PCA Scores - Frame {i+1}')
                ax.grid(True, alpha=0.3)
                
                # Save frame
                frame_path = self.temp_dir / f'pca_frame_{i:03d}.png'
                fig.savefig(frame_path, dpi=150, bbox_inches='tight')
                frames.append(str(frame_path))
                plt.close(fig)
            
            # Create GIF from frames
            images = [Image.open(frame) for frame in frames]
            images[0].save(
                output_path,
                save_all=True,
                append_images=images[1:],
                duration=1000//fps,
                loop=0
            )
            
            # Clean up temporary frames
            for frame in frames:
                Path(frame).unlink()
            
            self.logger.info(f"PCA evolution animation created: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create PCA animation: {e}")
            return False
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                self.temp_dir.mkdir(exist_ok=True)
            self.logger.info("Temporary files cleaned up")
        except Exception as e:
            self.logger.warning(f"Failed to clean up temp files: {e}")