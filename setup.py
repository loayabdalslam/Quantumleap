from setuptools import setup, find_packages

setup(
    name='quantum-leap',
    version='0.1.0',
    description='The Ultimate Hybrid AI Accelerator Simulator',
    author='Jules',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'tensornetwork',
    ],
)
