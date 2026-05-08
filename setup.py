from setuptools import setup, find_packages

setup(
    name="torch_arch_inspector",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
    ],
    author="Your Name",
    description="An interactive HTML inspector for PyTorch model architectures",
)
