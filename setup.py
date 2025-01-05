from setuptools import setup, find_packages

setup(
    name='TT_KNN',
    version='1.0.0',
    packages=find_packages(),
    author="AnSyu Li",
    author_email="yessir0621@gmail.com",  
    description="A package for temporal trajectory analysis using K-Nearest Neighbors",
    url="https://github.com/Louis-Li-dev/TT_KNN",  
    install_requires=[
        "pandas",
        "tqdm",
        "scikit-learn",
        "matplotlib",
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Replace with your license
        "Operating System :: OS Independent",
    ],

)
