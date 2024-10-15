from setuptools import setup, find_packages

setup(
    name="projet07",  # You can customize this name
    version="0.1",
    description="Custom transformers and utilities for Projet07",
    packages=find_packages(),  # Automatically find and include Python packages
    include_package_data=True,  # Include additional files from MANIFEST.in if needed
    install_requires=[],  # Leave this empty if using conda.yaml or poetry
    zip_safe=False
)