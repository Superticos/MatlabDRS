"""
Example: Batch processing of multiple spectral files
"""
import numpy as np
from pathlib import Path
from drs_analyzer.core.data_loader import DataLoader
from drs_analyzer.core.data_processor import DataProcessor
from drs_analyzer.analysis.pca_analyser import PCAAnalyzer
from drs_analyzer.plotting.export_manager import ExportManager

def batch_process_example():
    """
    Example of batch processing workflow
    """
    print("🔬 DRS Analyzer - Batch Processing Example")
    print("=" * 50)
    
    # Initialize components
    loader = DataLoader()
    processor = DataProcessor()
    pca = PCAAnalyzer()
    exporter = ExportManager()
    
    # Define input directory (create sample if needed)
    input_dir = Path("sample_data")
    if not input_dir.exists():
        print("Creating sample data directory...")
        create_sample_data(input_dir)
    
    # Find all supported files
    file_patterns = ["*.mat", "*.csv", "*.xlsx"]
    files = []
    for pattern in file_patterns:
        files.extend(input_dir.glob(pattern))
    
    if not files:
        print("❌ No supported files found in sample_data directory")
        return
    
    print(f"📁 Found {len(files)} files to process")
    
    # Process each file
    all_processed = []
    all_wavelengths = None
    
    for i, file_path in enumerate(files):
        print(f"Processing {i+1}/{len(files)}: {file_path.name}")
        
        try:
            # Load data
            success, data = loader.load_file(file_path)
            if not success:
                print(f"  ❌ Failed to load: {data.get('error', 'Unknown error')}")
                continue
            
            # Process spectra
            processed = processor.process_batch(
                data['spectra'],
                baseline_method='polynomial',
                smoothing_method='savgol',
                normalization_method='minmax'
            )
            
            all_processed.append(processed)
            if all_wavelengths is None:
                all_wavelengths = data['wavelengths']
            
            print(f"  ✅ Processed {processed.shape[0]} spectra")
            
        except Exception as e:
            print(f"  ❌ Error processing {file_path.name}: {e}")
            continue
    
    if not all_processed:
        print("❌ No files were successfully processed")
        return
    
    # Combine all processed spectra
    combined_spectra = np.vstack(all_processed)
    print(f"\n📊 Combined Dataset: {combined_spectra.shape[0]} spectra")
    
    # Perform PCA analysis
    print("🔍 Performing PCA analysis...")
    pca_results = pca.fit_transform(combined_spectra)
    
    print(f"✅ PCA completed:")
    print(f"  - Components: {pca_results['n_components']}")
    print(f"  - Variance explained: {pca_results['total_variance_explained']:.3f}")
    
    # Export results
    output_dir = Path("batch_results")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n💾 Exporting results to {output_dir}/")
    
    # Export processed spectra
    export_data = {
        'spectra': combined_spectra,
        'wavelengths': all_wavelengths,
        'pca_results': pca_results
    }
    
    success = exporter.export_data(export_data, output_dir / "batch_results.xlsx")
    if success:
        print("  ✅ Excel export completed")
    
    # Export PCA plots
    from drs_analyzer.plotting.drs_plotter import DRSPlotter
    plotter = DRSPlotter()
    
    fig = plotter.plot_pca_results(pca_results, all_wavelengths)
    plotter.save_figure(output_dir / "pca_analysis.png", fig)
    print("  ✅ PCA plots saved")
    
    print("\n🎉 Batch processing completed successfully!")

def create_sample_data(output_dir: Path):
    """Create sample spectral data for demonstration"""
    output_dir.mkdir(exist_ok=True)
    
    # Generate sample data
    wavelengths = np.linspace(400, 800, 100)
    
    for i in range(3):
        # Create slightly different spectra
        n_spectra = np.random.randint(5, 10)
        base_spectrum = np.exp(-(wavelengths - 600)**2 / 5000) + 0.1
        
        spectra = []
        for j in range(n_spectra):
            noise = np.random.normal(0, 0.02, len(wavelengths))
            shift = np.random.normal(0, 10)  # wavelength shift
            amplitude = np.random.normal(1, 0.1)  # amplitude variation
            
            shifted_wavelengths = wavelengths + shift
            spectrum = amplitude * np.interp(wavelengths, shifted_wavelengths, base_spectrum) + noise
            spectra.append(spectrum)
        
        spectra = np.array(spectra)
        
        # Save as MATLAB file
        import scipy.io as sio
        sio.savemat(output_dir / f"sample_{i+1}.mat", {
            'spectra': spectra,
            'wavelengths': wavelengths
        })
    
    print(f"✅ Created 3 sample files in {output_dir}")

if __name__ == "__main__":
    batch_process_example()