"""Setup script for environmental monitoring system"""

from setuptools import setup, find_packages
from pathlib import Path

# Read requirements
requirements = Path("requirements.txt").read_text().splitlines()

# Read README
readme = Path("README.md").read_text() if Path("README.md").exists() else ""

setup(
    name="environmental-monitoring",
    version="1.0.0",
    description="AI-Powered Environmental Monitoring System",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Environmental Monitoring Team",
    author_email="team@environmental-monitoring.com",
    url="https://github.com/yourusername/environmental-monitoring",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Environmental Monitoring",
    ],
    entry_points={
        "console_scripts": [
            "train-model=scripts.train_model:main",
            "run-monitoring=scripts.run_monitoring:main",
            "deploy-api=scripts.deploy_api:main",
            "download-data=scripts.download_data:main",
        ],
    },
)