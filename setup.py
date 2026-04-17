"""
Ol-y 1.1B ACTT - Setup configuration.
"""

from setuptools import setup, find_packages

setup(
    name="oly-1.1b-actt",
    version="1.0.0",
    description="Ol-y 1.1B Transformer with Affective Communication Tokens (ACT)",
    author="Ol-y Project",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "tokenizers>=0.15.0",
        "datasets>=2.16.0",
        "safetensors>=0.4.0",
        "tqdm>=4.66.0",
        "numpy>=1.24.0",
        "psutil>=5.9.0",
    ],
    extras_require={
        "training": [
            "accelerate>=0.25.0",
            "bitsandbytes>=0.41.0",
            "deepspeed>=0.12.0",
            "peft>=0.7.0",
        ],
        "claude": [
            "anthropic>=0.40.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
    },
)
