from setuptools import setup, find_packages

setup(
    name='quantum-leap',
    version='0.7.0', # Version bump for NumExpr integration
    description='The Ultimate Hybrid AI Accelerator Simulator',
    author='Jules',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'numexpr',
    ],
)
