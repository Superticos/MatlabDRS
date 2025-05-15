import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QLineEdit, QFileDialog, QTableWidget, QTableWidgetItem,
                             QSlider, QSpinBox, QComboBox, QDoubleSpinBox, QGroupBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFontDatabase  # Add this import
from scipy.signal import find_peaks
import imageio
import qt_material


class DRSAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DRS Analyzer - Python Version")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize variables
        self.raw_data = None
        self.wavelengths = None
        self.energy_ev = None
        self.processed_data = None
        self.delta_data = None
        self.folder_path = ""
        self.file_name = ""
        self.color_maps = {
            'Hot Red': 'hot',
            'Prism': 'prism',
            'Black & White': 'gray',
            'Blue Laurent': 'Blues'
        }
        self.current_color_map = 'Blues'
        self.plot_speed = 0.2
        self.delta_n = 1
        self.num_avg = 1
        self.dynamic_plotting = False

        # Initialize UI
        self.init_ui()

        # Load default wavelength and energy data
        self.load_default_spectral_data()

    def init_ui(self):
        # Create main tab widget
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # Create tabs
        self.create_load_tab()
        self.create_process_tab()
        self.create_plot_tab()
        self.create_peak_tab()

    def create_load_tab(self):
        """Create the data loading tab"""
        self.load_tab = QWidget()
        layout = QVBoxLayout()

        # File selection controls
        file_group = QGroupBox("File Selection")
        file_layout = QHBoxLayout()

        self.file_label = QLabel("File:")
        self.file_edit = QLineEdit()
        self.file_edit.setReadOnly(True)
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.load_data)

        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.file_edit, 1)
        file_layout.addWidget(self.browse_btn)
        file_group.setLayout(file_layout)

        # Data table
        self.data_table = QTableWidget()
        self.data_table.setAlternatingRowColors(True)

        layout.addWidget(file_group)
        layout.addWidget(self.data_table, 1)

        self.load_tab.setLayout(layout)
        self.tab_widget.addTab(self.load_tab, "Load Data")

    def create_process_tab(self):
        """Create the data processing tab"""
        self.process_tab = QWidget()
        layout = QVBoxLayout()

        # Processing controls
        process_group = QGroupBox("Processing Parameters")
        process_layout = QHBoxLayout()

        self.avg_label = QLabel("Averaging:")
        self.avg_spin = QSpinBox()
        self.avg_spin.setMinimum(1)
        self.avg_spin.setValue(1)

        self.process_btn = QPushButton("Process Data")
        self.process_btn.clicked.connect(self.process_data)

        # Add in create_process_tab, after self.processed_table:
        self.export_btn = QPushButton("Export Averaged DRS")
        self.export_btn.clicked.connect(self.export_averaged_drs)
        layout.addWidget(self.export_btn)

        process_layout.addWidget(self.avg_label)
        process_layout.addWidget(self.avg_spin)
        process_layout.addStretch(1)
        process_layout.addWidget(self.process_btn)
        process_group.setLayout(process_layout)

        # Processed data table
        self.processed_table = QTableWidget()
        self.processed_table.setAlternatingRowColors(True)

        layout.addWidget(process_group)
        layout.addWidget(self.processed_table, 1)

        self.process_tab.setLayout(layout)
        self.tab_widget.addTab(self.process_tab, "Processing")

    def create_plot_tab(self):
        """Create the plotting tab"""
        self.plot_tab = QWidget()
        main_layout = QHBoxLayout()

        # Left side: controls and DRS plot
        left_layout = QVBoxLayout()

        # DRS plot
        self.drs_figure = Figure(figsize=(6, 4), dpi=100)
        self.drs_canvas = FigureCanvas(self.drs_figure)
        self.drs_ax = self.drs_figure.add_subplot(111)
        self.drs_ax.set_title("DRS Spectra")
        self.drs_ax.set_xlabel("Wavelength (nm)")
        self.drs_ax.set_ylabel("DRS")

        # Controls
        controls_group = QGroupBox("Plot Controls")
        controls_layout = QVBoxLayout()

        # Color selection
        color_layout = QHBoxLayout()
        color_label = QLabel("Color Map:")
        self.color_combo = QComboBox()
        self.color_combo.addItems(self.color_maps.keys())
        self.color_combo.currentTextChanged.connect(self.update_color_map)

        color_layout.addWidget(color_label)
        color_layout.addWidget(self.color_combo)

        # Speed control
        speed_layout = QHBoxLayout()
        speed_label = QLabel("Speed:")
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(20)
        self.speed_slider.setValue(2)
        self.speed_slider.valueChanged.connect(self.update_speed)

        speed_layout.addWidget(speed_label)
        speed_layout.addWidget(self.speed_slider)

        # Delta DRS control
        delta_layout = QHBoxLayout()
        delta_label = QLabel("ΔDRS N:")
        self.delta_spin = QSpinBox()
        self.delta_spin.setMinimum(1)
        self.delta_spin.setValue(1)
        self.delta_spin.valueChanged.connect(self.update_delta_n)

        delta_layout.addWidget(delta_label)
        delta_layout.addWidget(self.delta_spin)

        # Bold curve control
        bold_layout = QHBoxLayout()
        bold_label = QLabel("Bold Curve:")
        self.bold_spin = QSpinBox()
        self.bold_spin.setMinimum(0)
        self.bold_spin.valueChanged.connect(self.highlight_curve)

        bold_layout.addWidget(bold_label)
        bold_layout.addWidget(self.bold_spin)

        # Buttons
        btn_layout = QHBoxLayout()
        self.drs_btn = QPushButton("Plot DRS")
        self.drs_btn.clicked.connect(self.plot_drs)

        self.delta_btn = QPushButton("Plot ΔDRS")
        self.delta_btn.clicked.connect(self.plot_delta_drs)

        self.gif_btn = QPushButton("Create GIF")
        self.gif_btn.clicked.connect(self.export_gif)

        btn_layout.addWidget(self.drs_btn)
        btn_layout.addWidget(self.delta_btn)
        btn_layout.addWidget(self.gif_btn)

        # Add all controls to group
        controls_layout.addLayout(color_layout)
        controls_layout.addLayout(speed_layout)
        controls_layout.addLayout(delta_layout)
        controls_layout.addLayout(bold_layout)
        controls_layout.addLayout(btn_layout)
        controls_group.setLayout(controls_layout)

        # Add to left layout
        left_layout.addWidget(self.drs_canvas, 1)
        left_layout.addWidget(controls_group)

        # Right side: delta DRS and selected curve plots
        right_layout = QVBoxLayout()

        # Delta DRS plot
        self.delta_figure = Figure(figsize=(6, 4), dpi=100)
        self.delta_canvas = FigureCanvas(self.delta_figure)
        self.delta_ax = self.delta_figure.add_subplot(111)
        self.delta_ax.set_title("ΔDRS Spectra")
        self.delta_ax.set_xlabel("Wavelength (nm)")
        self.delta_ax.set_ylabel("ΔDRS")

        # Selected curve plot
        self.selected_figure = Figure(figsize=(6, 2), dpi=100)
        self.selected_canvas = FigureCanvas(self.selected_figure)
        self.selected_ax = self.selected_figure.add_subplot(111)
        self.selected_ax.set_title("Selected Spectrum")
        self.selected_ax.set_xlabel("Wavelength (nm)")
        self.selected_ax.set_ylabel("DRS")

        # Add to right layout
        right_layout.addWidget(self.delta_canvas, 1)
        right_layout.addWidget(self.selected_canvas)

        # Combine layouts
        main_layout.addLayout(left_layout, 2)
        main_layout.addLayout(right_layout, 1)

        self.plot_tab.setLayout(main_layout)
        self.tab_widget.addTab(self.plot_tab, "DRS Plot")

    def create_peak_tab(self):
        """Create the peak analysis tab"""
        self.peak_tab = QWidget()
        layout = QVBoxLayout()

        # Controls
        controls_layout = QHBoxLayout()
        self.peak_sensitivity_label = QLabel("Peak Sensitivity:")
        self.peak_sensitivity_spin = QDoubleSpinBox()
        self.peak_sensitivity_spin.setDecimals(2)
        self.peak_sensitivity_spin.setRange(0.01, 1.0)
        self.peak_sensitivity_spin.setSingleStep(0.01)
        self.peak_sensitivity_spin.setValue(0.1)
        self.peak_sensitivity_spin.valueChanged.connect(self.update_peak_analysis)
        controls_layout.addWidget(self.peak_sensitivity_label)
        controls_layout.addWidget(self.peak_sensitivity_spin)
        layout.addLayout(controls_layout)

        # Peak plot
        self.peak_figure = Figure(figsize=(6, 4), dpi=100)
        self.peak_canvas = FigureCanvas(self.peak_figure)
        self.peak_ax = self.peak_figure.add_subplot(111)
        self.peak_ax.set_title("Peak Evolution")
        self.peak_ax.set_xlabel("Averaged Spectrum Index")
        self.peak_ax.set_ylabel("Peak Intensity / Area")
        layout.addWidget(self.peak_canvas)

        self.peak_tab.setLayout(layout)
        self.tab_widget.addTab(self.peak_tab, "Peak Analysis")

    def load_default_spectral_data(self):
        """Load wavelength and energy data from static files in the app folder"""
        try:
            app_dir = os.path.dirname(os.path.abspath(__file__))
            self.wavelengths = np.loadtxt(os.path.join(app_dir, 'wavelength.txt'))
            self.energy_ev = np.loadtxt(os.path.join(app_dir, 'energy_eV.txt'))
        except Exception as e:
            print(f"Error loading spectral data: {e}")
            self.wavelengths = None
            self.energy_ev = None

    def load_data(self):
        """Load data from CSV file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open DRS Data File", "", "CSV Files (*.csv);;All Files (*)"
        )

        if file_path:
            try:
                self.file_edit.setText(file_path)
                self.folder_path = os.path.dirname(file_path)
                self.file_name = os.path.basename(file_path)

                # Read CSV data with semicolon separator
                self.raw_data = pd.read_csv(file_path, sep=';')

                # Update data table
                self.update_data_table()

                # Enable processing button
                self.process_btn.setEnabled(True)

            except Exception as e:
                print(f"Error loading file: {e}")

    def update_data_table(self):
        """Update the data table with loaded data"""
        if self.raw_data is not None:
            self.data_table.setRowCount(self.raw_data.shape[0])
            self.data_table.setColumnCount(self.raw_data.shape[1])

            # Set headers
            headers = []
            if len(self.raw_data.columns) > 2 and self.wavelengths is not None:
                # First two columns are time and status
                headers.extend(self.raw_data.columns[:2])
                # Remaining columns are wavelengths
                headers.extend([f"{w:.1f} nm" for w in self.wavelengths[:self.raw_data.shape[1] - 2]])
            else:
                headers = list(self.raw_data.columns)

            self.data_table.setHorizontalHeaderLabels(headers)

            # Populate table
            for i in range(self.raw_data.shape[0]):
                for j in range(self.raw_data.shape[1]):
                    item = QTableWidgetItem(str(self.raw_data.iloc[i, j]))
                    self.data_table.setItem(i, j, item)

    def process_data(self):
        """Process the raw data with averaging"""
        if self.raw_data is None:
            return

        try:
            avg_window = self.avg_spin.value()
            num_spectra = len(self.raw_data)
            self.num_avg = num_spectra // avg_window

            # Extract spectral data (skip first two columns: time and status)
            spectral_data = self.raw_data.iloc[:, 2:].values

            # Apply averaging
            self.processed_data = np.zeros((self.num_avg, spectral_data.shape[1]))
            for i in range(self.num_avg):
                start_idx = i * avg_window
                end_idx = (i + 1) * avg_window
                self.processed_data[i, :] = np.mean(spectral_data[start_idx:end_idx, :], axis=0)

            # Calculate delta DRS
            self.calculate_delta_drs()

            # Update processed data table
            self.update_processed_table()

            # Enable plotting controls
            self.enable_plotting_controls()

            # Update bold curve spinner range
            self.bold_spin.setMaximum(self.num_avg)

            # Update peak analysis
            self.update_peak_analysis()

        except Exception as e:
            print(f"Error processing data: {e}")

    def export_averaged_drs(self):
        """Export the averaged DRS data to CSV"""
        if self.processed_data is None:
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Averaged DRS Data",
            os.path.join(self.folder_path, "averaged_drs.csv"),
            "CSV Files (*.csv)"
        )
        if not save_path:
            return
        # Prepare DataFrame
        df = pd.DataFrame(self.processed_data, columns=[f"{w:.1f} nm" for w in self.wavelengths[:self.processed_data.shape[1]]])
        df.to_csv(save_path, index=False)

    def update_peak_analysis(self):
        """Detect peaks and plot their evolution (intensity and area)"""
        if self.processed_data is None or self.wavelengths is None:
            self.peak_ax.clear()
            self.peak_ax.set_title("Peak Evolution")
            self.peak_ax.set_xlabel("Averaged Spectrum Index")
            self.peak_ax.set_ylabel("Peak Intensity / Area")
            self.peak_canvas.draw()
            return

        sensitivity = self.peak_sensitivity_spin.value()
        peak_heights = []
        peak_areas = []
        for spectrum in self.processed_data:
            peaks = self.detect_peaks(spectrum, sensitivity)
            if peaks:
                # Use the highest peak
                main_peak = max(peaks, key=lambda p: p['intensity'])
                peak_heights.append(main_peak['intensity'])
                peak_areas.append(main_peak['area'])
            else:
                peak_heights.append(np.nan)
                peak_areas.append(np.nan)
        self.peak_ax.clear()
        self.peak_ax.plot(range(1, len(peak_heights)+1), peak_heights, marker='o', label='Peak Intensity')
        self.peak_ax.plot(range(1, len(peak_areas)+1), peak_areas, marker='s', label='Peak Area')
        self.peak_ax.set_title("Peak Evolution")
        self.peak_ax.set_xlabel("Averaged Spectrum Index")
        self.peak_ax.set_ylabel("Peak Intensity / Area")
        self.peak_ax.grid(True)
        self.peak_ax.legend()
        self.peak_canvas.draw()

    def calculate_delta_drs(self):
        """Calculate delta DRS spectra"""
        if self.processed_data is None:
            return

        delta_n = self.delta_spin.value()
        num_spectra = len(self.processed_data)

        self.delta_data = np.zeros((num_spectra - delta_n, self.processed_data.shape[1]))
        for i in range(num_spectra - delta_n):
            self.delta_data[i, :] = self.processed_data[i + delta_n, :] - self.processed_data[i, :]

    def update_processed_table(self):
        """Update the processed data table"""
        if self.processed_data is not None and self.wavelengths is not None:
            self.processed_table.setRowCount(self.processed_data.shape[0])
            self.processed_table.setColumnCount(self.processed_data.shape[1])

            # Set headers with wavelengths
            headers = [f"{w:.1f} nm" for w in self.wavelengths[:self.processed_data.shape[1]]]
            self.processed_table.setHorizontalHeaderLabels(headers)

            # Populate table
            for i in range(self.processed_data.shape[0]):
                for j in range(self.processed_data.shape[1]):
                    item = QTableWidgetItem(f"{self.processed_data[i, j]:.4f}")
                    self.processed_table.setItem(i, j, item)

    def enable_plotting_controls(self):
        """Enable plotting controls after processing"""
        self.drs_btn.setEnabled(True)
        self.delta_btn.setEnabled(True)
        self.gif_btn.setEnabled(True)
        self.bold_spin.setEnabled(True)

    def update_color_map(self, color_name):
        """Update the color map based on selection"""
        self.current_color_map = self.color_maps.get(color_name, 'Blues')

    def update_speed(self, value):
        """Update plotting speed"""
        self.plot_speed = 1.0 / value  # Inverse relationship (higher value = faster)

    def update_delta_n(self, value):
        """Update delta N value and recalculate delta DRS"""
        self.delta_n = value
        if self.processed_data is not None:
            self.calculate_delta_drs()

    def highlight_curve(self, index):
        """Highlight a specific curve in the DRS plot"""
        if index == 0 or self.processed_data is None or self.wavelengths is None:
            # Clear the selected curve plot
            self.selected_ax.clear()
            self.selected_canvas.draw()
            return

        # Plot the selected curve
        self.selected_ax.clear()
        self.selected_ax.plot(self.wavelengths, self.processed_data[index - 1, :],
                              'r-', linewidth=2)
        self.selected_ax.set_title(f"Spectrum {index}")
        self.selected_ax.set_xlabel("Wavelength (nm)")
        self.selected_ax.set_ylabel("DRS")
        self.selected_ax.grid(True)

        # Auto-scale axes
        self.selected_ax.relim()
        self.selected_ax.autoscale_view()

        self.selected_canvas.draw()

    def plot_drs(self):
        """Plot DRS spectra with color gradient"""
        if self.processed_data is None or self.wavelengths is None:
            return

        self.drs_ax.clear()

        # Get colormap
        cmap = plt.get_cmap(self.current_color_map)
        colors = cmap(np.linspace(0, 1, self.num_avg))

        # Plot each spectrum
        for i in range(self.num_avg):
            self.drs_ax.plot(self.wavelengths, self.processed_data[i, :],
                             color=colors[i], linewidth=1)

            # Pause for animation effect
            QApplication.processEvents()
            plt.pause(self.plot_speed)

        self.drs_ax.set_title("DRS Spectra")
        self.drs_ax.set_xlabel("Wavelength (nm)")
        self.drs_ax.set_ylabel("DRS")
        self.drs_ax.grid(True)

        # Auto-scale axes
        self.drs_ax.relim()
        self.drs_ax.autoscale_view()

        self.drs_canvas.draw()

    def plot_delta_drs(self):
        """Plot delta DRS spectra"""
        if self.delta_data is None or self.wavelengths is None:
            return

        self.delta_ax.clear()

        # Get colormap
        cmap = plt.get_cmap(self.current_color_map)
        colors = cmap(np.linspace(0, 1, len(self.delta_data)))

        # Plot each delta spectrum
        for i in range(len(self.delta_data)):
            self.delta_ax.plot(self.wavelengths, self.delta_data[i, :],
                               color=colors[i], linewidth=1)

            # Pause for animation effect
            QApplication.processEvents()
            plt.pause(self.plot_speed)

        self.delta_ax.set_title("ΔDRS Spectra")
        self.delta_ax.set_xlabel("Wavelength (nm)")
        self.delta_ax.set_ylabel("ΔDRS")
        self.delta_ax.grid(True)

        # Auto-scale axes
        self.delta_ax.relim()
        self.delta_ax.autoscale_view()

        self.delta_canvas.draw()

    def export_gif(self):
        """Export DRS animation as GIF"""
        if self.processed_data is None or self.wavelengths is None:
            return

        # Get save path
        base_name = os.path.splitext(self.file_name)[0]
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save GIF Animation",
            os.path.join(self.folder_path, f"{base_name}_animation.gif"),
            "GIF Files (*.gif)"
        )

        if not save_path:
            return

        # Create frames for GIF
        frames = []
        fig, ax = plt.subplots(figsize=(8, 6))

        # Get colormap
        cmap = plt.get_cmap(self.current_color_map)
        colors = cmap(np.linspace(0, 1, self.num_avg))

        for i in range(self.num_avg):
            ax.clear()
            ax.plot(self.wavelengths, self.processed_data[i, :],
                    color=colors[i], linewidth=2)
            ax.set_title(f"DRS Spectra - Frame {i + 1}/{self.num_avg}")
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("DRS")
            ax.grid(True)

            # Draw the frame
            fig.canvas.draw()

            # Convert to image array
            frame = np.array(fig.canvas.renderer.buffer_rgba())
            frames.append(frame)

            # Pause for animation effect
            plt.pause(self.plot_speed)

        plt.close(fig)

        # Save as GIF
        try:
            imageio.mimsave(save_path, frames, duration=self.plot_speed * 1000)
            print(f"GIF saved to {save_path}")
        except Exception as e:
            print(f"Error saving GIF: {e}")

    def track_peak_evolution(self, peak_wavelength, tolerance=5):
        """Track the area of a peak near a given wavelength over all averaged spectra"""
        if self.processed_data is None or self.wavelengths is None:
            return []
        idx = (np.abs(self.wavelengths - peak_wavelength)).argmin()
        evolution = []
        for spectrum in self.processed_data:
            # Search for the local max within the tolerance window
            left = max(0, idx - tolerance)
            right = min(len(self.wavelengths) - 1, idx + tolerance)
            local_idx = left + np.argmax(spectrum[left:right+1])
            area = self.calculate_peak_area(spectrum, local_idx)
            evolution.append(area)
        return evolution

    def detect_peaks(self, spectrum, sensitivity=0.1):
        """Detect peaks in a spectrum and calculate their area"""
        from scipy.signal import find_peaks

        peaks, props = find_peaks(spectrum, height=sensitivity)
        peak_data = []
        for peak_idx in peaks:
            area = self.calculate_peak_area(spectrum, peak_idx)
            peak_data.append({
                'index': peak_idx,
                'wavelength': self.wavelengths[peak_idx],
                'intensity': spectrum[peak_idx],
                'area': area
            })
        return peak_data

    def calculate_peak_area(self, spectrum, peak_idx):
        """Calculate area under a peak using trapezoidal rule"""
        left = peak_idx
        right = peak_idx
        # Find left boundary
        while left > 0 and spectrum[left] > spectrum[left - 1]:
            left -= 1
        # Find right boundary
        while right < len(spectrum) - 1 and spectrum[right] > spectrum[right + 1]:
            right += 1
        # Area under the peak
        area = np.trapz(spectrum[left:right+1], self.wavelengths[left:right+1])
        return area


if __name__ == "__main__":
    app = QApplication(sys.argv)
    import qt_material
    from PyQt5.QtGui import QFontDatabase  # <-- Add this line
    qt_material.apply_stylesheet(app, theme='dark_teal.xml')
    window = DRSAnalyzer()
    window.show()
    sys.exit(app.exec_())