from setuptools import setup, find_packages

setup(
    name='TT_KNN',
    version='1.0.0',
    packages=find_packages(),
    author="AnSyu Li",
    install_requires=[
        "pandas",
        "tqdm",
        "scikit-learn",
        "matplotlib",
        "numpy",
    ],
)

