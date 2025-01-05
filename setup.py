from setuptools import setup, find_packages

setup(
    name='TT_KNN',
    version='1.0.0',
    packages=['ttknn'],
    author="AnSyu Li",
    author_email="yessir0621@gmail.com",  
    license="MIT",
    description="A package for temporal trajectory analysis using K-Nearest Neighbors",
    url="https://github.com/Louis-Li-dev/TT_KNN",  
    include_package_data=True,
    install_requires=[
        "pandas",
        "tqdm",
        "scikit-learn",
        "matplotlib",
        "numpy",
    ],

)
