from setuptools import setup, find_packages

setup(
    name="fe_aco",
    version="1.1.0",
    description="Federated Explainable Augmented Colony Optimization - GPU Graph Engine",
    author="Idris Rayan Chakib (Co-founder SMART ADN)",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy",
        "scikit-learn",
        "scipy",
        "matplotlib",
        "torch_geometric"
    ],
    python_requires='>=3.8',
)