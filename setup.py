from setuptools import find_packages, setup

setup(
    name='src',
    package_dir={"": "src"},
    packages=find_packages("src"),
    version='0.1.0',
    description='Nutrisage end-to-end ML pipeline',
    author='Zohreh Asaee',
    license='MIT',
)
