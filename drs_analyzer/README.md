# DRS Analyzer

Advanced Diffuse Reflectance Spectroscopy (DRS) analysis toolkit with comprehensive data processing, analysis, and visualization capabilities.

## Features

### 🔬 Data Processing
- **Multi-format support**: MATLAB (.mat), CSV, Excel, HDF5, text files
- **Advanced preprocessing**: Baseline correction, smoothing, normalization
- **Batch processing**: Handle multiple files efficiently
- **Quality control**: Automatic data validation and outlier detection

### 📊 Analysis Capabilities
- **Principal Component Analysis (PCA)**: Dimensionality reduction and data exploration
- **Peak Detection**: Advanced algorithms for spectral peak identification
- **Statistical Analysis**: Comprehensive statistical tools
- **Clustering**: Group similar spectra automatically
- **Correlation Analysis**: Identify relationships in spectral data

### 📈 Visualization
- **Interactive Plots**: Plotly-powered interactive visualizations
- **Publication-Quality Figures**: Matplotlib integration for publication
- **Real-time Preview**: Live parameter adjustment and preview
- **3D Visualizations**: Advanced 3D plotting capabilities

### 💾 Export & Reporting
- **Multiple Formats**: PNG, PDF, SVG, HTML, Excel, MATLAB, HDF5
- **Comprehensive Reports**: Automated analysis report generation
- **Batch Export**: Export multiple analyses simultaneously
- **Animation Creation**: Time-series spectral animations

## Installation

### From PyPI
```bash
pip install drs-analyzer
```

### From Source
```bash
git clone https://github.com/spectroscopylab/drs-analyzer.git
cd drs-analyzer
pip install -e .
```

### With GUI Support
```bash
pip install drs-analyzer[gui]
```

### Development Installation
```bash
pip install drs-analyzer[dev]
```

## Quick Start

### GUI Application
```bash
drs-analyzer
```

### Command Line Interface
```bash
# Process single file
drs-cli process input.mat -o output.xlsx --baseline polynomial --smooth savgol

# Batch analysis
drs-cli batch *.mat -o results/ --analysis pca peaks stats
```

### Python API
```python
from drs_analyzer import DataLoader, SpectralProcessor, PCAAnalyzer

# Load data
loader = DataLoader()
success, data = loader.load_file('spectra.mat')

# Process spectra
processor = SpectralProcessor()
processed = processor.process_batch(
    data['spectra'],
    baseline_method='polynomial',
    smoothing_method='savgol',
    normalization_method='minmax'
)

# Perform PCA
pca = PCAAnalyzer()
result = pca.fit_transform(processed)
print(f"Explained variance: {result['explained_variance_ratio']}")
```

## Documentation

### Processing Options

#### Baseline Correction
- **Polynomial**: Fit polynomial to baseline
- **ALS (Asymmetric Least Squares)**: Advanced baseline correction
- **Rolling Ball**: Rolling ball baseline estimation
- **SNIP**: Statistics-sensitive Non-linear Iterative Peak-clipping

#### Smoothing
- **Savitzky-Golay**: Polynomial smoothing filter
- **Gaussian**: Gaussian filter smoothing
- **Moving Average**: Simple moving average
- **Median**: Median filter for noise reduction

#### Normalization
- **MinMax**: Scale to range [0,1]
- **Standard**: Z-score normalization
- **SNV**: Standard Normal Variate
- **MSC**: Multiplicative Scatter Correction

### File Format Support

| Format | Extension | Import | Export |
|--------|-----------|--------|--------|
| MATLAB | .mat | ✅ | ✅ |
| CSV | .csv | ✅ | ✅ |
| Excel | .xlsx | ✅ | ✅ |
| HDF5 | .h5 | ✅ | ✅ |
| Text | .txt | ✅ | ✅ |
| JSON | .json | ✅ | ✅ |

## Examples

### Example 1: Basic Processing
```python
from drs_analyzer import DataLoader, SpectralProcessor

# Load spectral data
loader = DataLoader()
success, data = loader.load_file('samples.mat')

if success:
    spectra = data['spectra']
    wavelengths = data['wavelengths']
    
    # Create processor
    processor = SpectralProcessor()
    
    # Apply processing pipeline
    processed = processor.process_spectrum(
        spectra[0],  # First spectrum
        baseline_correction={'method': 'polynomial', 'degree': 2},
        smoothing={'method': 'savgol', 'window_length': 5},
        normalization={'method': 'minmax'}
    )
```

### Example 2: PCA Analysis
```python
from drs_analyzer import PCAAnalyzer, DRSPlotter

# Perform PCA
pca = PCAAnalyzer()
result = pca.fit_transform(processed_spectra)

# Create plots
plotter = DRSPlotter()
fig = plotter.plot_pca_results(result, wavelengths, interactive=True)
plotter.save_figure('pca_analysis.html', fig)
```

### Example 3: Peak Detection
```python
from drs_analyzer import PeakDetector

# Detect peaks
detector = PeakDetector()
peaks = detector.detect_peaks_batch(
    processed_spectra,
    wavelengths,
    height=0.1,
    prominence=0.05,
    distance=10
)

# Plot results
fig = plotter.plot_peak_analysis(
    processed_spectra, 
    wavelengths, 
    peaks,
    interactive=True
)
```

## Configuration

### Settings File
Create `~/.drs_analyzer/config.yaml`:

```yaml
processing:
  default_baseline: 'polynomial'
  default_smoothing: 'savgol'
  default_normalization: 'minmax'

plotting:
  theme: 'default'
  dpi: 300
  figure_size: [10, 8]

export:
  default_format: 'png'
  include_metadata: true
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
```bash
git clone https://github.com/spectroscopylab/drs-analyzer.git
cd drs-analyzer
pip install -e .[dev]
```

### Running Tests
```bash
pytest tests/ -v --cov=drs_analyzer
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use DRS Analyzer in your research, please cite:

```bibtex
@software{drs_analyzer,
  title={DRS Analyzer: Advanced Diffuse Reflectance Spectroscopy Analysis Toolkit},
  author={Spectroscopy Lab},
  year={2024},
  url={https://github.com/spectroscopylab/drs-analyzer}
}
```

## Support

- 📧 Email: lab@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/spectroscopylab/drs-analyzer/issues)
- 📖 Documentation: [Read the Docs](https://drs-analyzer.readthedocs.io)
- 💬 Discussions: [GitHub Discussions](https://github.com/spectroscopylab/drs-analyzer/discussions)