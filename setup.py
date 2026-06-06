# ==============================================================================
# Name: Vivek Kumar
# Enroll: 23125038
# Email: vivek_k@mfs.iitr.ac.in
# ==============================================================================
"""Setup configuration for EV Charging Tariff Optimization package."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh.readlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="ev-charging-tariff-optimization",
    version="1.0.0",
    author="OP26 Analytics Team",
    description="Agentic AI-Based Dynamic Tariff Optimization for EV Charging Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ev-charging-tariff-optimization",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "ev-tariff-train=scripts.run_pipeline:main",
        ],
    },
)
