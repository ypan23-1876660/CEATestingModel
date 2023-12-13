from setuptools import find_packages, setup

with open("requirements.txt") as file:
    requirements = file.read().splitlines()

setup(
    name="ml4cea",
    packages=find_packages(),
    install_requires=requirements
)
