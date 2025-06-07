"""
Setup script for DRS Analyzer package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = requirements_file.read_text().strip().split('\n')

setup(
    name="drs-analyzer",
    version="1.0.0",
    author="Spectroscopy Lab",
    author_email="lab@example.com",
    description="Advanced DRS (Diffuse Reflectance Spectroscopy) analysis toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/spectroscopylab/drs-analyzer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "gui": ["PyQt5>=5.15.0"],
        "plotting": ["plotly>=5.0.0", "kaleido>=0.2.1"],
        "matlab": ["scipy>=1.7.0"],
        "excel": ["openpyxl>=3.0.0"],
        "hdf5": ["h5py>=3.0.0"],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "mypy>=0.910",
        ]
    },
    entry_points={
        "console_scripts": [
            "drs-analyzer=drs_analyzer.main:main",
            "drs-cli=drs_analyzer.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "drs_analyzer": [
            "config/*.yaml",
            "gui/icons/*.png",
            "gui/styles/*.qss",
        ]
    },
    zip_safe=False,
)