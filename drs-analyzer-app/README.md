# DRS Analyzer Application

## Overview
The DRS Analyzer is a Python application designed for analyzing Diffuse Reflectance Spectroscopy (DRS) data. It provides functionalities for loading, processing, plotting, and analyzing spectral data.

## Features
- Load DRS data from CSV files.
- Process raw data with averaging.
- Plot DRS and ΔDRS spectra.
- Perform peak analysis and track peak evolution.
- Export averaged DRS data and create GIF animations of the spectra.

## Installation

### Prerequisites
Ensure you have Python installed on your system. It is recommended to use Python 3.6 or higher.

### Dependencies
To install the required dependencies, run the following command:

```
pip install -r requirements.txt
```

## Running the Application
To run the DRS Analyzer application, execute the following command:

```
python src/drs_plotter.py
```

## File Structure
- `src/drs_plotter.py`: Main application code.
- `src/wavelength.txt`: Wavelength data for spectral analysis.
- `src/energy_eV.txt`: Energy data in electron volts.
- `src/__init__.py`: Marks the directory as a Python package.
- `requirements.txt`: Lists the project dependencies.
- `README.md`: Documentation for the project.
- `setup.py`: Packaging configuration.
- `build_exe.spec`: Configuration for building the executable.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.