"""
Advanced plotting widgets for DRS spectroscopy
Interactive plots with PyQtGraph for high performance
"""

import numpy as np
import pyqtgraph as pg
from pyqtgraph import PlotWidget, ImageView, GraphicsLayoutWidget
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, 
    QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox, QLabel, 
    QSlider, QGroupBox, QSplitter, QTabWidget, QFileDialog,
    QColorDialog, QMessageBox, QProgressBar, QTextEdit
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QColor, QPen, QBrush, QFont
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import json

from ..config.settings import AppSettings

class DRSPlotWidget(QWidget):
    """Base class for DRS plotting widgets"""
    
    # Signals
    spectrum_selected = pyqtSignal(int)  # spectrum index
    peak_selected = pyqtSignal(int, int)  # spectrum index, peak index
    region_selected = pyqtSignal(float, float)  # wavelength range
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = AppSettings()
        self.logger = logging.getLogger(__name__)
        
        # Data storage
        self.data = None
        self.wavelengths = None
        self.current_peaks = []
        
        # Plot configuration
        self.colors = self._generate_color_palette()
        self.current_theme = 'dark'
        
        # Initialize UI
        self.init_ui()
        self.setup_plot()
        self.apply_theme()
    
    def init_ui(self):
        """Initialize the user interface"""
        self.layout = QVBoxLayout(self)
        
        # Create plot widget
        self.plot_widget = PlotWidget()
        self.plot_item = self.plot_widget.plotItem
        
        # Controls layout
        self.controls_layout = QHBoxLayout()
        
        self.layout.addWidget(self.plot_widget)
        self.layout.addLayout(self.controls_layout)
    
    def setup_plot(self):
        """Setup plot appearance and behavior"""
        # Configure plot
        self.plot_item.setLabel('left', 'Intensity', units='a.u.')
        self.plot_item.setLabel('bottom', 'Wavelength', units='nm')
        self.plot_item.showGrid(x=True, y=True, alpha=0.3)
        self.plot_item.setMouseEnabled(x=True, y=True)
        self.plot_item.enableAutoRange()
        
        # Add crosshair
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.plot_item.addItem(self.vLine, ignoreBounds=True)
        self.plot_item.addItem(self.hLine, ignoreBounds=True)
        
        # Connect mouse events
        self.plot_widget.scene().sigMouseMoved.connect(self.mouse_moved)
        self.plot_widget.scene().sigMouseClicked.connect(self.mouse_clicked)
    
    def _generate_color_palette(self) -> List[QColor]:
        """Generate distinct colors for plotting"""
        colors = [
            QColor(255, 0, 0),      # Red
            QColor(0, 255, 0),      # Green
            QColor(0, 0, 255),      # Blue
            QColor(255, 255, 0),    # Yellow
            QColor(255, 0, 255),    # Magenta
            QColor(0, 255, 255),    # Cyan
            QColor(255, 128, 0),    # Orange
            QColor(128, 0, 255),    # Purple
            QColor(255, 128, 128),  # Light Red
            QColor(128, 255, 128),  # Light Green
        ]
        
        # Extend with generated colors
        for i in range(20):
            hue = (i * 360 / 20) % 360
            color = QColor()
            color.setHsv(hue, 200, 255)
            colors.append(color)
        
        return colors
    
    def apply_theme(self):
        """Apply visual theme to the plot"""
        if self.current_theme == 'dark':
            pg.setConfigOption('background', 'k')
            pg.setConfigOption('foreground', 'w')
            self.plot_widget.setBackground('k')
        else:
            pg.setConfigOption('background', 'w')
            pg.setConfigOption('foreground', 'k')
            self.plot_widget.setBackground('w')
    
    def mouse_moved(self, pos):
        """Handle mouse movement for crosshair"""
        if self.plot_item.sceneBoundingRect().contains(pos):
            mouse_point = self.plot_item.vb.mapSceneToView(pos)
            self.vLine.setPos(mouse_point.x())
            self.hLine.setPos(mouse_point.y())
    
    def mouse_clicked(self, event):
        """Handle mouse clicks"""
        if event.double():
            self.plot_item.autoRange()
    
    def export_plot(self):
        """Export plot to file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Plot",
            "",
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)"
        )
        
        if file_path:
            try:
                exporter = pg.exporters.ImageExporter(self.plot_item)
                exporter.parameters()['width'] = 1920
                exporter.export(file_path)
                self.logger.info(f"Plot exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export plot: {e}")

class SpectraPlotWidget(DRSPlotWidget):
    """Widget for plotting spectral data with interactive features"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Spectra-specific attributes
        self.raw_data = None
        self.processed_data = None
        self.plot_curves = []
        self.peak_items = []
        self.selected_spectra = set()
        
        # Processing comparison
        self.show_raw = True
        self.show_processed = True
        
        # Plot options
        self.offset_mode = False
        self.offset_value = 0.1
        
        self.setup_controls()
    
    def setup_controls(self):
        """Setup control widgets specific to spectra plotting"""
        # Display options group
        display_group = QGroupBox("Display Options")
        display_layout = QGridLayout(display_group)
        
        # Raw/processed toggles
        self.raw_check = QCheckBox("Show Raw")
        self.raw_check.setChecked(True)
        self.raw_check.toggled.connect(self.update_display)
        
        self.processed_check = QCheckBox("Show Processed")
        self.processed_check.setChecked(True)
        self.processed_check.toggled.connect(self.update_display)
        
        # Offset mode
        self.offset_check = QCheckBox("Offset Mode")
        self.offset_check.toggled.connect(self.toggle_offset_mode)
        
        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setRange(0.01, 1.0)
        self.offset_spin.setValue(0.1)
        self.offset_spin.setSingleStep(0.01)
        self.offset_spin.valueChanged.connect(self.set_offset_value)
        
        # Peak display
        self.peaks_check = QCheckBox("Show Peaks")
        self.peaks_check.setChecked(True)
        self.peaks_check.toggled.connect(self.update_peak_display)
        
        # Layout controls
        display_layout.addWidget(self.raw_check, 0, 0)
        display_layout.addWidget(self.processed_check, 0, 1)
        display_layout.addWidget(self.offset_check, 1, 0)
        display_layout.addWidget(self.offset_spin, 1, 1)
        display_layout.addWidget(self.peaks_check, 2, 0)
        
        # Selection group
        selection_group = QGroupBox("Selection")
        selection_layout = QGridLayout(selection_group)
        
        # Spectrum selection
        self.spectrum_combo = QComboBox()
        self.spectrum_combo.currentIndexChanged.connect(self.select_spectrum)
        
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self.select_all_spectra)
        
        self.clear_selection_btn = QPushButton("Clear Selection")
        self.clear_selection_btn.clicked.connect(self.clear_selection)
        
        selection_layout.addWidget(QLabel("Spectrum:"), 0, 0)
        selection_layout.addWidget(self.spectrum_combo, 0, 1)
        selection_layout.addWidget(self.select_all_btn, 1, 0)
        selection_layout.addWidget(self.clear_selection_btn, 1, 1)
        
        # Add groups to controls
        self.controls_layout.addWidget(display_group)
        self.controls_layout.addWidget(selection_group)
        self.controls_layout.addStretch()
    
    def update_data(self, spectra: np.ndarray, wavelengths: np.ndarray):
        """Update plot with new spectral data"""
        try:
            self.raw_data = spectra.copy()
            self.wavelengths = wavelengths.copy()
            
            # Update spectrum selector
            self.spectrum_combo.clear()
            for i in range(len(spectra)):
                self.spectrum_combo.addItem(f"Spectrum {i+1}")
            
            # Plot data
            self.plot_spectra()
            
            self.logger.info(f"Updated plot with {len(spectra)} spectra")
            
        except Exception as e:
            self.logger.error(f"Failed to update spectra data: {e}")
    
    def update_processed_data(self, processed_spectra: np.ndarray, wavelengths: np.ndarray):
        """Update plot with processed spectral data"""
        try:
            self.processed_data = processed_spectra.copy()
            self.wavelengths = wavelengths.copy()
            
            # Replot if showing processed data
            if self.show_processed:
                self.plot_spectra()
            
            self.logger.info("Updated plot with processed data")
            
        except Exception as e:
            self.logger.error(f"Failed to update processed data: {e}")
    
    def plot_spectra(self):
        """Plot spectral data"""
        try:
            # Clear existing plots
            self.clear_plot()
            
            # Determine what to plot
            data_to_plot = []
            labels = []
            
            if self.show_raw and self.raw_data is not None:
                data_to_plot.append(('raw', self.raw_data))
                
            if self.show_processed and self.processed_data is not None:
                data_to_plot.append(('processed', self.processed_data))
            elif self.raw_data is not None:
                data_to_plot.append(('raw', self.raw_data))
            
            # Plot each dataset
            for data_type, data in data_to_plot:
                self._plot_dataset(data, data_type)
            
            # Update plot range
            self.plot_item.autoRange()
            
        except Exception as e:
            self.logger.error(f"Failed to plot spectra: {e}")
    
    def _plot_dataset(self, data: np.ndarray, data_type: str):
        """Plot a single dataset"""
        n_spectra = len(data)
        
        for i, spectrum in enumerate(data):
            # Apply offset if enabled
            if self.offset_mode:
                offset = i * self.offset_value * np.max(spectrum)
                y_data = spectrum + offset
            else:
                y_data = spectrum
            
            # Select color and style
            color_idx = i % len(self.colors)
            color = self.colors[color_idx]
            
            if data_type == 'processed':
                pen = pg.mkPen(color=color, width=2)
            else:
                pen = pg.mkPen(color=color, width=1, style=Qt.DashLine)
            
            # Create curve
            curve = self.plot_item.plot(
                self.wavelengths, 
                y_data, 
                pen=pen,
                name=f"{data_type.title()} {i+1}"
            )
            
            # Make curve clickable
            curve.setClickable(True)
            curve.sigClicked.connect(lambda curve, idx=i: self.spectrum_clicked(idx))
            
            self.plot_curves.append(curve)
    
    def update_peaks(self, peak_lists: List[List]):
        """Update plot with detected peaks"""
        try:
            self.current_peaks = peak_lists
            
            if self.peaks_check.isChecked():
                self.plot_peaks()
            
        except Exception as e:
            self.logger.error(f"Failed to update peaks: {e}")
    
    def plot_peaks(self):
        """Plot detected peaks as markers"""
        try:
            # Clear existing peak items
            self.clear_peaks()
            
            if not self.current_peaks:
                return
            
            # Use processed data if available, otherwise raw data
            data_to_use = self.processed_data if self.processed_data is not None else self.raw_data
            
            if data_to_use is None:
                return
            
            for i, spectrum_peaks in enumerate(self.current_peaks):
                if not spectrum_peaks:
                    continue
                
                color = self.colors[i % len(self.colors)]
                
                for j, peak in enumerate(spectrum_peaks):
                    # Apply offset if enabled
                    if self.offset_mode:
                        offset = i * self.offset_value * np.max(data_to_use[i])
                        y_pos = peak.intensity + offset
                    else:
                        y_pos = peak.intensity
                    
                    # Create peak marker
                    peak_item = pg.ScatterPlotItem(
                        [peak.wavelength], 
                        [y_pos],
                        symbol='o',
                        size=8,
                        brush=pg.mkBrush(color),
                        pen=pg.mkPen('w', width=1)
                    )
                    
                    peak_item.setClickable(True)
                    peak_item.sigClicked.connect(
                        lambda item, s_idx=i, p_idx=j: self.peak_clicked(s_idx, p_idx)
                    )
                    
                    self.plot_item.addItem(peak_item)
                    self.peak_items.append(peak_item)
            
            self.logger.info(f"Plotted peaks for {len(self.current_peaks)} spectra")
            
        except Exception as e:
            self.logger.error(f"Failed to plot peaks: {e}")
    
    def clear_plot(self):
        """Clear all plotted data"""
        for curve in self.plot_curves:
            self.plot_item.removeItem(curve)
        self.plot_curves.clear()
        self.clear_peaks()
    
    def clear_peaks(self):
        """Clear peak markers"""
        for item in self.peak_items:
            self.plot_item.removeItem(item)
        self.peak_items.clear()
    
    def spectrum_clicked(self, index: int):
        """Handle spectrum selection"""
        if index in self.selected_spectra:
            self.selected_spectra.remove(index)
        else:
            self.selected_spectra.add(index)
        
        self.spectrum_selected.emit(index)
        self.highlight_selected_spectra()
    
    def peak_clicked(self, spectrum_idx: int, peak_idx: int):
        """Handle peak selection"""
        self.peak_selected.emit(spectrum_idx, peak_idx)
    
    def highlight_selected_spectra(self):
        """Highlight selected spectra"""
        # This would modify the appearance of selected curves
        # Implementation depends on specific requirements
        pass
    
    def update_display(self):
        """Update display based on checkboxes"""
        self.show_raw = self.raw_check.isChecked()
        self.show_processed = self.processed_check.isChecked()
        self.plot_spectra()
    
    def update_peak_display(self):
        """Toggle peak display"""
        if self.peaks_check.isChecked():
            self.plot_peaks()
        else:
            self.clear_peaks()
    
    def toggle_offset_mode(self, enabled: bool):
        """Toggle offset mode for better visualization"""
        self.offset_mode = enabled
        self.offset_spin.setEnabled(enabled)
        self.plot_spectra()
        
        if enabled and self.peaks_check.isChecked():
            self.plot_peaks()
    
    def set_offset_value(self, value: float):
        """Set offset value for stacked display"""
        self.offset_value = value
        if self.offset_mode:
            self.plot_spectra()
            if self.peaks_check.isChecked():
                self.plot_peaks()
    
    def select_spectrum(self, index: int):
        """Select specific spectrum"""
        if index >= 0:
            self.selected_spectra.clear()
            self.selected_spectra.add(index)
            self.spectrum_selected.emit(index)
            self.highlight_selected_spectra()
    
    def select_all_spectra(self):
        """Select all spectra"""
        if self.raw_data is not None:
            self.selected_spectra = set(range(len(self.raw_data)))
            self.highlight_selected_spectra()
    
    def clear_selection(self):
        """Clear spectrum selection"""
        self.selected_spectra.clear()
        self.highlight_selected_spectra()

class StatisticsPlotWidget(DRSPlotWidget):
    """Widget for plotting statistical analysis results"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Replace single plot with tab widget for multiple views
        self.layout.removeWidget(self.plot_widget)
        self.plot_widget.setParent(None)
        
        # Create tab widget for different analysis views
        self.plot_tabs = QTabWidget()
        self.layout.insertWidget(0, self.plot_tabs)
        
        # Create different plot widgets
        self.setup_analysis_plots()
        self.setup_controls()
        
        # Data storage for different analyses
        self.pca_data = None
        self.clustering_data = None
        self.correlation_data = None
        self.report_data = None
    
    def setup_analysis_plots(self):
        """Setup different analysis plot widgets"""
        # PCA plot
        self.pca_widget = GraphicsLayoutWidget()
        self.pca_scores_plot = self.pca_widget.addPlot(row=0, col=0, title="PCA Scores")
        self.pca_loadings_plot = self.pca_widget.addPlot(row=0, col=1, title="PCA Loadings")
        self.pca_variance_plot = self.pca_widget.addPlot(row=1, col=0, colspan=2, title="Explained Variance")
        
        # Clustering plot
        self.clustering_widget = GraphicsLayoutWidget()
        self.cluster_plot = self.clustering_widget.addPlot(row=0, col=0, title="Cluster Visualization")
        self.cluster_centers_plot = self.clustering_widget.addPlot(row=0, col=1, title="Cluster Centers")
        
        # Correlation plot
        self.correlation_widget = PlotWidget()
        self.correlation_plot = self.correlation_widget.plotItem
        self.correlation_plot.setTitle("Correlation Matrix")
        
        # Report widget
        self.report_widget = QTextEdit()
        self.report_widget.setReadOnly(True)
        
        # Add tabs
        self.plot_tabs.addTab(self.pca_widget, "📊 PCA")
        self.plot_tabs.addTab(self.clustering_widget, "🔍 Clustering") 
        self.plot_tabs.addTab(self.correlation_widget, "🔗 Correlation")
        self.plot_tabs.addTab(self.report_widget, "📋 Report")
    
    def setup_controls(self):
        """Setup controls for statistics plots"""
        # Analysis options
        analysis_group = QGroupBox("Analysis Options")
        analysis_layout = QGridLayout(analysis_group)
        
        # PCA components
        self.pca_comp_spin = QSpinBox()
        self.pca_comp_spin.setRange(2, 20)
        self.pca_comp_spin.setValue(5)
        
        # Clustering method
        self.cluster_method_combo = QComboBox()
        self.cluster_method_combo.addItems(['K-Means', 'DBSCAN', 'Hierarchical'])
        
        # Number of clusters
        self.n_clusters_spin = QSpinBox()
        self.n_clusters_spin.setRange(2, 20)
        self.n_clusters_spin.setValue(3)
        
        analysis_layout.addWidget(QLabel("PCA Components:"), 0, 0)
        analysis_layout.addWidget(self.pca_comp_spin, 0, 1)
        analysis_layout.addWidget(QLabel("Clustering:"), 1, 0)
        analysis_layout.addWidget(self.cluster_method_combo, 1, 1)
        analysis_layout.addWidget(QLabel("N Clusters:"), 2, 0)
        analysis_layout.addWidget(self.n_clusters_spin, 2, 1)
        
        self.controls_layout.addWidget(analysis_group)
        self.controls_layout.addStretch()
    
    def update_pca_results(self, pca_data: Dict[str, Any]):
        """Update PCA plots with results"""
        try:
            self.pca_data = pca_data
            
            # Clear existing plots
            self.pca_scores_plot.clear()
            self.pca_loadings_plot.clear()
            self.pca_variance_plot.clear()
            
            scores = pca_data['scores']
            loadings = pca_data.get('loadings')
            variance_ratio = pca_data['explained_variance_ratio']
            
            # Plot scores (first two components)
            if scores.shape[1] >= 2:
                scatter = pg.ScatterPlotItem(
                    scores[:, 0], 
                    scores[:, 1],
                    brush=pg.mkBrush(255, 0, 0, 120),
                    size=8
                )
                self.pca_scores_plot.addItem(scatter)
                self.pca_scores_plot.setLabel('left', 'PC2')
                self.pca_scores_plot.setLabel('bottom', 'PC1')
            
            # Plot loadings if available
            if loadings is not None and hasattr(self, 'wavelengths') and self.wavelengths is not None:
                for i in range(min(3, loadings.shape[0])):  # Plot first 3 components
                    color = self.colors[i % len(self.colors)]
                    pen = pg.mkPen(color=color, width=2)
                    self.pca_loadings_plot.plot(
                        self.wavelengths, 
                        loadings[i], 
                        pen=pen,
                        name=f'PC{i+1}'
                    )
                
                self.pca_loadings_plot.setLabel('left', 'Loadings')
                self.pca_loadings_plot.setLabel('bottom', 'Wavelength (nm)')
                self.pca_loadings_plot.addLegend()
            
            # Plot explained variance
            x_vals = np.arange(1, len(variance_ratio) + 1)
            bars = pg.BarGraphItem(
                x=x_vals, 
                height=variance_ratio * 100,
                width=0.8,
                brush='b'
            )
            self.pca_variance_plot.addItem(bars)
            self.pca_variance_plot.setLabel('left', 'Explained Variance (%)')
            self.pca_variance_plot.setLabel('bottom', 'Principal Component')
            
            # Switch to PCA tab
            self.plot_tabs.setCurrentIndex(0)
            
            self.logger.info("Updated PCA plots")
            
        except Exception as e:
            self.logger.error(f"Failed to update PCA plots: {e}")
    
    def update_clustering_results(self, clustering_data: Dict[str, Any]):
        """Update clustering plots with results"""
        try:
            self.clustering_data = clustering_data
            
            # Clear existing plots
            self.cluster_plot.clear()
            self.cluster_centers_plot.clear()
            
            labels = clustering_data['labels']
            unique_labels = np.unique(labels)
            
            # Use PCA data for visualization if available
            if self.pca_data is not None:
                scores = self.pca_data['scores']
                
                # Plot clusters in PC space
                for label in unique_labels:
                    mask = labels == label
                    if label == -1:  # Noise points (for DBSCAN)
                        color = QColor(128, 128, 128)  # Gray
                        symbol = 'x'
                    else:
                        color = self.colors[label % len(self.colors)]
                        symbol = 'o'
                    
                    if scores.shape[1] >= 2:
                        scatter = pg.ScatterPlotItem(
                            scores[mask, 0],
                            scores[mask, 1],
                            brush=pg.mkBrush(color),
                            symbol=symbol,
                            size=8
                        )
                        self.cluster_plot.addItem(scatter)
                
                self.cluster_plot.setLabel('left', 'PC2')
                self.cluster_plot.setLabel('bottom', 'PC1')
                self.cluster_plot.setTitle(f"Clusters in PC Space ({clustering_data['method']})")
            
            # Plot cluster centers if available
            cluster_stats = clustering_data.get('cluster_statistics', {})
            if cluster_stats and hasattr(self, 'wavelengths') and self.wavelengths is not None:
                for i, (cluster_name, stats) in enumerate(cluster_stats.items()):
                    if 'mean_spectrum' in stats:
                        color = self.colors[i % len(self.colors)]
                        pen = pg.mkPen(color=color, width=2)
                        self.cluster_centers_plot.plot(
                            self.wavelengths,
                            stats['mean_spectrum'],
                            pen=pen,
                            name=cluster_name
                        )
                
                self.cluster_centers_plot.setLabel('left', 'Intensity')
                self.cluster_centers_plot.setLabel('bottom', 'Wavelength (nm)')
                self.cluster_centers_plot.addLegend()
            
            # Switch to clustering tab
            self.plot_tabs.setCurrentIndex(1)
            
            self.logger.info("Updated clustering plots")
            
        except Exception as e:
            self.logger.error(f"Failed to update clustering plots: {e}")
    
    def update_correlation_results(self, correlation_data: Dict[str, Any]):
        """Update correlation plot with results"""
        try:
            self.correlation_data = correlation_data
            
            # Clear existing plot
            self.correlation_plot.clear()
            
            corr_matrix = correlation_data['correlation_matrix']
            
            # Create image item for correlation matrix
            img_item = pg.ImageItem()
            img_item.setImage(corr_matrix)
            
            # Set colormap
            colormap = pg.colormap.get('viridis')
            img_item.setLookupTable(colormap.getLookupTable())
            
            self.correlation_plot.addItem(img_item)
            self.correlation_plot.setAspectLocked(True)
            self.correlation_plot.setTitle(f"Correlation Matrix ({correlation_data['method']})")
            
            # Add colorbar
            colorbar = pg.ColorBarItem(
                values=(corr_matrix.min(), corr_matrix.max()),
                colorMap=colormap
            )
            
            # Switch to correlation tab
            self.plot_tabs.setCurrentIndex(2)
            
            self.logger.info("Updated correlation plot")
            
        except Exception as e:
            self.logger.error(f"Failed to update correlation plot: {e}")
    
    def update_report(self, report_data: Dict[str, Any]):
        """Update report tab with comprehensive analysis"""
        try:
            self.report_data = report_data
            
            # Generate HTML report
            html_report = self._generate_html_report(report_data)
            self.report_widget.setHtml(html_report)
            
            # Switch to report tab
            self.plot_tabs.setCurrentIndex(3)
            
            self.logger.info("Updated analysis report")
            
        except Exception as e:
            self.logger.error(f"Failed to update report: {e}")
    
    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML report from analysis data"""
        html = """
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #2E86C1; border-bottom: 2px solid #2E86C1; }
                h2 { color: #5D6D7E; border-bottom: 1px solid #D5DBDB; }
                .summary { background-color: #EBF5FB; padding: 10px; border-radius: 5px; }
                .metric { margin: 5px 0; }
                .value { font-weight: bold; color: #E74C3C; }
                table { border-collapse: collapse; width: 100%; margin: 10px 0; }
                th, td { border: 1px solid #D5DBDB; padding: 8px; text-align: left; }
                th { background-color: #F8F9FA; }
            </style>
        </head>
        <body>
        """
        
        html += "<h1>🔬 DRS Analysis Report</h1>"
        
        # Data summary
        if 'data_summary' in report_data:
            summary = report_data['data_summary']
            html += "<h2>📊 Data Summary</h2>"
            html += '<div class="summary">'
            html += f'<div class="metric">Number of Spectra: <span class="value">{summary["n_spectra"]}</span></div>'
            html += f'<div class="metric">Wavelength Points: <span class="value">{summary["n_wavelengths"]}</span></div>'
            html += f'<div class="metric">Wavelength Range: <span class="value">{summary["wavelength_range"][0]:.1f} - {summary["wavelength_range"][1]:.1f} nm</span></div>'
            html += '</div>'
        
        # PCA Results
        if 'pca' in report_data:
            pca = report_data['pca']
            html += "<h2>📈 Principal Component Analysis</h2>"
            html += f'<div class="metric">Components Analyzed: <span class="value">{pca["n_components"]}</span></div>'
            html += f'<div class="metric">Total Variance Explained: <span class="value">{pca["total_variance_explained"]:.1%}</span></div>'
            
            # Variance table
            html += "<h3>Explained Variance by Component</h3>"
            html += "<table><tr><th>Component</th><th>Variance (%)</th><th>Cumulative (%)</th></tr>"
            for i, (var, cum_var) in enumerate(zip(pca['explained_variance_ratio'], pca['cumulative_variance'])):
                html += f"<tr><td>PC{i+1}</td><td>{var:.2%}</td><td>{cum_var:.2%}</td></tr>"
            html += "</table>"
        
        # Clustering Results
        if 'clustering' in report_data:
            clustering = report_data['clustering']
            html += "<h2>🔍 Clustering Analysis</h2>"
            html += f'<div class="metric">Method: <span class="value">{clustering["method"].title()}</span></div>'
            html += f'<div class="metric">Number of Clusters: <span class="value">{clustering["n_clusters"]}</span></div>'
            
            if 'silhouette_score' in clustering:
                html += f'<div class="metric">Silhouette Score: <span class="value">{clustering["silhouette_score"]:.3f}</span></div>'
            
            # Cluster sizes
            if 'cluster_statistics' in clustering:
                html += "<h3>Cluster Information</h3>"
                html += "<table><tr><th>Cluster</th><th>Size</th><th>Percentage</th></tr>"
                total_size = sum(stats['size'] for stats in clustering['cluster_statistics'].values())
                for cluster_name, stats in clustering['cluster_statistics'].items():
                    percentage = stats['size'] / total_size * 100
                    html += f"<tr><td>{cluster_name}</td><td>{stats['size']}</td><td>{percentage:.1f}%</td></tr>"
                html += "</table>"
        
        # Basic Statistics
        if 'basic_statistics' in report_data:
            stats = report_data['basic_statistics']
            html += "<h2>📋 Basic Statistics</h2>"
            
            overall = stats['overall_statistics']
            html += f'<div class="metric">Global Mean Intensity: <span class="value">{overall["global_mean"]:.3f}</span></div>'
            html += f'<div class="metric">Global Std Deviation: <span class="value">{overall["global_std"]:.3f}</span></div>'
            html += f'<div class="metric">Signal-to-Noise Ratio: <span class="value">{overall["signal_to_noise_ratio"]:.1f} dB</span></div>'
        
        # Metadata
        if 'metadata' in report_data:
            metadata = report_data['metadata']
            html += "<h2>ℹ️ Analysis Metadata</h2>"
            html += f'<div class="metric">Analysis Time: <span class="value">{metadata.get("analysis_timestamp", "Unknown")}</span></div>'
            html += f'<div class="metric">Preprocessing: <span class="value">{metadata.get("preprocessing", "None")}</span></div>'
        
        html += "</body></html>"
        return html
    
    def export_plot(self):
        """Export current statistical plot"""
        current_index = self.plot_tabs.currentIndex()
        current_widget = self.plot_tabs.currentWidget()
        
        if current_index < 3:  # Plot widgets
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Statistical Plot",
                "",
                "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)"
            )
            
            if file_path:
                try:
                    if hasattr(current_widget, 'scene'):
                        scene = current_widget.scene()
                        exporter = pg.exporters.ImageExporter(scene)
                        exporter.parameters()['width'] = 1920
                        exporter.export(file_path)
                    self.logger.info(f"Statistical plot exported to {file_path}")
                except Exception as e:
                    QMessageBox.critical(self, "Export Error", f"Failed to export plot: {e}")
        
        elif current_index == 3:  # Report
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Report",
                "",
                "HTML Files (*.html);;PDF Files (*.pdf)"
            )
            
            if file_path:
                try:
                    if file_path.endswith('.html'):
                        with open(file_path, 'w') as f:
                            f.write(self.report_widget.toHtml())
                    self.logger.info(f"Report exported to {file_path}")
                except Exception as e:
                    QMessageBox.critical(self, "Export Error", f"Failed to export report: {e}")