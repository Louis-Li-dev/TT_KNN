from setuptools import setup, find_packages

setup(
    name='TT_KNN',
    version='1.1.3', 
    packages=find_packages(include=['ttknn', 'ttknn.*']),  
    author="AnSyu Li",
    author_email="yessir0621@gmail.com",
    license="MIT",
    description="A package for temporal trajectory analysis using K-Nearest Neighbors",
    url="https://github.com/Louis-Li-dev/TT_KNN",
    include_package_data=True, 
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
