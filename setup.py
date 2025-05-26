from setuptools import setup, find_packages

setup(
    name="rag_medical",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "bertopic>=0.15.0",
        "pandas>=2.0.0",
        "hdbscan>=0.8.33",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "sentence-transformers>=2.2.2",
        "umap-learn>=0.5.3"
    ],
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    description="Medical Question Topic Clustering System",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/rag_medical",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)