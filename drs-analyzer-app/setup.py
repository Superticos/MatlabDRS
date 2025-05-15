from setuptools import setup, find_packages

setup(
    name="drs-analyzer-app",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A DRS Analyzer application for spectral analysis.",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        "PyQt5",
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
        "imageio"
    ],
    entry_points={
        'gui_scripts': [
            'drs-analyzer=drs_plotter:main',  # Assuming you have a main function in drs_plotter.py
        ],
    },
    include_package_data=True,
    zip_safe=False,
)