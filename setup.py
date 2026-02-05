"""
VLA Project Setup
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="vla",
    version="0.1.0",
    author="VLA Team",
    description="Vision-Language-Action models for robot manipulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/vla",
    packages=find_packages(exclude=["tests", "tutorials", "scripts"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Robotics",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.37.0",
        "accelerate>=0.25.0",
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "h5py>=3.9.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
        ],
        "wandb": ["wandb>=0.15.0"],
        "lora": ["peft>=0.7.0"],
    },
    entry_points={
        "console_scripts": [
            "vla-train=scripts.train:main",
            "vla-eval=scripts.evaluate:main",
            "vla-collect=scripts.collect_demos:main",
        ],
    },
)
