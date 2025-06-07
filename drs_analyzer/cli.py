"""
Command line interface for DRS Analyzer
"""
import argparse
import sys
import logging
from pathlib import Path
from typing import List, Optional
from typing import Dict, Any

from drs_analyzer.core.data_loader import DataLoader
from drs_analyzer.core.data_processor import DataProcessor
from drs_analyzer.analysis.pca_analyser import PCAAnalyzer  # Fixed import path
from drs_analyzer.analysis.statistical_analyser import StatisticalAnalyzer  # Fixed import path
from drs_analyzer.utils.export_manager import ExportManager  # Fixed import path
from drs_analyzer.utils.logger import setup_logging
from drs_analyzer.config.settings import AppSettings
import numpy as np

def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=log_level, console_output=True)
    
    try:
        if args.command == 'gui':
            launch_gui()
        elif args.command == 'process':
            process_files(args)
        elif args.command == 'analyze':
            analyze_data(args)
        elif args.command == 'batch':
            batch_process(args)
        elif args.command == 'info':
            show_file_info(args)
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)

def create_parser() -> argparse.ArgumentParser:
    """Create command line parser"""
    parser = argparse.ArgumentParser(
        description='DRS Analyzer - Advanced spectroscopy data analysis toolkit',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  drs-cli gui                           Launch GUI interface
  drs-cli process data.mat -o results/  Process single file
  drs-cli batch data/ -o results/       Batch process directory
  drs-cli analyze data.mat --pca        Perform PCA analysis
  drs-cli info data.mat                 Show file information
        """
    )
    
    parser.add_argument('--version', action='version', version='DRS Analyzer 1.0.0')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # GUI command
    gui_parser = subparsers.add_parser('gui', help='Launch GUI interface')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process single file')
    process_parser.add_argument('input', help='Input file path')
    process_parser.add_argument('-o', '--output', help='Output directory')
    process_parser.add_argument('--baseline', choices=['polynomial', 'als', 'rolling_ball'], 
                               default='polynomial', help='Baseline correction method')
    process_parser.add_argument('--smooth', choices=['savgol', 'gaussian', 'moving_average'],
                               default='savgol', help='Smoothing method')
    process_parser.add_argument('--normalize', choices=['minmax', 'standard', 'robust'],
                               help='Normalization method')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Perform analysis')
    analyze_parser.add_argument('input', help='Input file path')
    analyze_parser.add_argument('-o', '--output', help='Output directory')
    analyze_parser.add_argument('--pca', action='store_true', help='Perform PCA analysis')
    analyze_parser.add_argument('--stats', action='store_true', help='Generate statistics')
    analyze_parser.add_argument('--clustering', action='store_true', help='Perform clustering')
    analyze_parser.add_argument('--components', type=int, default=5, help='Number of PCA components')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch process directory')
    batch_parser.add_argument('input', help='Input directory path')
    batch_parser.add_argument('-o', '--output', help='Output directory')
    batch_parser.add_argument('--pattern', default='*.mat', help='File pattern to match')
    batch_parser.add_argument('--baseline', choices=['polynomial', 'als', 'rolling_ball'],
                             default='polynomial', help='Baseline correction method')
    batch_parser.add_argument('--smooth', choices=['savgol', 'gaussian', 'moving_average'],
                             default='savgol', help='Smoothing method')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show file information')
    info_parser.add_argument('input', help='Input file path')
    
    return parser

def launch_gui():
    """Launch GUI interface"""
    from drs_analyzer.main import main as gui_main
    gui_main()

def process_files(args):
    """Process single file"""
    logger = logging.getLogger(__name__)
    
    # Initialize components
    settings = AppSettings()
    loader = DataLoader(settings)
    processor = DataProcessor(settings)
    
    logger.info(f"Processing file: {args.input}")
    
    # Load data
    success, data = loader.load_file(args.input)
    if not success:
        raise ValueError(f"Failed to load {args.input}: {data.get('error', 'Unknown error')}")
    
    logger.info(f"Loaded {data['spectra'].shape[0]} spectra with {data['spectra'].shape[1]} wavelength points")
    
    # Process data
    processed = processor.process_spectra(
        data['spectra'],
        baseline_method=args.baseline,
        smoothing_method=args.smooth,
        normalization_method=args.normalize
    )
    
    logger.info("Processing completed")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        export_manager = ExportManager()
        export_data = {
            'spectra': processed,
            'wavelengths': data['wavelengths'],
            'metadata': data['metadata']
        }
        
        success = export_manager.export_data(export_data, output_path / "processed_data.xlsx")
        if success:
            logger.info(f"Results saved to {output_path}")
        else:
            logger.error("Failed to save results")

def analyze_data(args):
    """Perform data analysis"""
    logger = logging.getLogger(__name__)
    
    # Initialize components
    settings = AppSettings()
    loader = DataLoader(settings)
    
    logger.info(f"Analyzing file: {args.input}")
    
    # Load data
    success, data = loader.load_file(args.input)
    if not success:
        raise ValueError(f"Failed to load {args.input}: {data.get('error', 'Unknown error')}")
    
    results = {}
    
    # PCA Analysis
    if args.pca:
        logger.info("Performing PCA analysis...")
        pca = PCAAnalyzer(n_components=args.components)
        pca_results = pca.fit_transform(data['spectra'])
        results['pca'] = pca_results
        logger.info(f"PCA completed - explained variance: {pca_results['total_variance_explained']:.3f}")
    
    # Statistical Analysis
    if args.stats:
        logger.info("Generating statistical analysis...")
        stats_analyzer = StatisticalAnalyzer()
        stats_results = stats_analyzer.generate_report(data['spectra'], data['wavelengths'])
        results['statistics'] = stats_results
        logger.info("Statistical analysis completed")
    
    # Clustering
    if args.clustering:
        logger.info("Performing clustering analysis...")
        stats_analyzer = StatisticalAnalyzer()
        clustering_results = stats_analyzer.perform_clustering(data['spectra'])
        results['clustering'] = clustering_results
        logger.info(f"Clustering completed - {clustering_results['n_clusters']} clusters found")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        export_manager = ExportManager()
        export_data = {
            'spectra': data['spectra'],
            'wavelengths': data['wavelengths'],
            'metadata': data['metadata'],
            'analysis_results': results
        }
        
        success = export_manager.export_data(export_data, output_path / "analysis_results.xlsx")
        if success:
            logger.info(f"Analysis results saved to {output_path}")

def batch_process(args):
    """Batch process directory"""
    logger = logging.getLogger(__name__)
    
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        raise ValueError(f"Input must be a directory: {args.input}")
    
    # Find files
    files = list(input_dir.glob(args.pattern))
    if not files:
        raise ValueError(f"No files found matching pattern '{args.pattern}' in {input_dir}")
    
    logger.info(f"Found {len(files)} files to process")
    
    # Initialize components
    settings = AppSettings()
    loader = DataLoader(settings)
    processor = DataProcessor(settings)
    
    all_processed = []
    all_wavelengths = None
    successful_files = []
    
    for i, file_path in enumerate(files):
        logger.info(f"Processing {i+1}/{len(files)}: {file_path.name}")
        
        try:
            # Load data
            success, data = loader.load_file(file_path)
            if not success:
                logger.warning(f"Failed to load {file_path.name}: {data.get('error')}")
                continue
            
            # Process
            processed = processor.process_spectra(
                data['spectra'],
                baseline_method=args.baseline,
                smoothing_method=args.smooth
            )
            
            all_processed.append(processed)
            if all_wavelengths is None:
                all_wavelengths = data['wavelengths']
            
            successful_files.append(file_path.name)
            
        except Exception as e:
            logger.warning(f"Error processing {file_path.name}: {e}")
            continue
    
    if not all_processed:
        raise ValueError("No files were successfully processed")
    
    logger.info(f"Successfully processed {len(all_processed)} files")
    
    # Combine results
    combined_spectra = np.vstack(all_processed)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        export_manager = ExportManager()
        export_data = {
            'spectra': combined_spectra,
            'wavelengths': all_wavelengths,
            'metadata': {
                'processed_files': successful_files,
                'processing_settings': {
                    'baseline_method': args.baseline,
                    'smoothing_method': args.smooth
                }
            }
        }
        
        success = export_manager.export_data(export_data, output_path / "batch_results.xlsx")
        if success:
            logger.info(f"Batch results saved to {output_path}")

def show_file_info(args):
    """Show file information"""
    logger = logging.getLogger(__name__)
    
    settings = AppSettings()
    loader = DataLoader(settings)
    
    file_info = loader.get_file_info(args.input)
    
    print(f"\nFile Information: {args.input}")
    print("=" * 50)
    
    if file_info['exists']:
        print(f"Size: {file_info['size_mb']:.2f} MB")
        print(f"Modified: {file_info['modified']}")
        print(f"Format: {file_info['format']}")
        print(f"Supported: {'Yes' if file_info['supported'] else 'No'}")
        
        if file_info['supported']:
            # Try to load and show data info
            success, data = loader.load_file(args.input)
            if success:
                print(f"\nData Information:")
                print(f"Spectra count: {data['spectra'].shape[0]}")
                print(f"Wavelength points: {data['spectra'].shape[1]}")
                print(f"Wavelength range: {data['wavelengths'].min():.1f} - {data['wavelengths'].max():.1f}")
                print(f"Data range: {data['spectra'].min():.4f} - {data['spectra'].max():.4f}")
            else:
                print(f"\nError loading data: {data.get('error')}")
    else:
        print("File does not exist or cannot be accessed")
        if 'error' in file_info:
            print(f"Error: {file_info['error']}")

if __name__ == "__main__":
    main()