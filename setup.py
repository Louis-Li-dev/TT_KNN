from setuptools import setup, find_packages

setup(
    name='TT_KNN',  # Package name
    version='1.0.0',  # Version number
    packages=find_packages(include=['ttknn', 'ttknn.*']),  # Use find_packages to include submodules
    author="AnSyu Li",
    author_email="yessir0621@gmail.com",
    license="MIT",
    description="A package for temporal trajectory analysis using K-Nearest Neighbors",
    long_description=open("README.md").read(),  # Use README.md for detailed description
    long_description_content_type="text/markdown",  # Specify Markdown format
    url="https://github.com/Louis-Li-dev/TT_KNN",
    include_package_data=True,  # Includes non-Python files specified in MANIFEST.in
    install_requires=[
        "pandas>=2.0.0",
        "tqdm>=4.0.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.0.0",
        "numpy>=1.18.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",  # Specify supported Python versions
)
