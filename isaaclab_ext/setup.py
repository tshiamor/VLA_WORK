"""Setup for VLA Isaac Lab Extension."""

from setuptools import setup, find_packages

setup(
    name="ext_vla_tasks",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "isaaclab",
    ],
    author="VLA Team",
    description="Vision-Language-Action tasks for Isaac Lab",
)
