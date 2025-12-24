from setuptools import setup, find_packages

setup(
    name="fe_aco",
    version="2.0.0",  # Passage en v2.0 suite à l'optimisation H100 & Sparse
    description="SMART ADN Core Engine: Federated Explainable Augmented Colony Optimization for Biomedical Knowledge Graphs",
    long_description="Moteur d'IA neuro-symbolique optimisé pour le criblage thérapeutique sur GPU H100. Intègre la gestion des tenseurs Sparse, la simulation fédérée et l'explicabilité pharmacologique.",
    author="Idris Rayan Chakib",
    maintainer="SMART ADN R&D Team",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",       # Requis pour les meilleures performances Sparse sur H100
        "numpy",
        "scikit-learn",
        "scipy",
        "matplotlib",
        "torch_geometric",    # Indispensable pour la simulation biomédicale
        "networkx>=3.0"       # <--- NOUVEAU : Requis pour visualiser les clusters de médicaments
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="drug-discovery, graph-neural-networks, swarm-intelligence, federated-learning, bioinformatics",
    python_requires='>=3.9', # Les environnements H100 utilisent généralement Python récent
)