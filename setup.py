from setuptools import setup, find_packages
import os
import re
import subprocess

#base directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

with open("README.md", "r") as fh:
    long_description = fh.read()

def read_requirements(filename):
    """
    Extract requirements from a pip formatted requirements file
    """
    filepath = os.path.join(BASE_DIR, filename)  
    if os.path.isfile(filepath):
        with open(filepath) as f:
            return [
                line.strip() 
                for line in f 
                if line.strip() and not line.startswith("#") and not line.startswith("git+")
            ]
    return []

def get_git_dependencies(filename):
    """Extract Git-based dependencies from requirements.txt."""
    filepath = os.path.join(BASE_DIR, filename)
    git_deps = []
    if os.path.isfile(filepath):
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line.startswith("git+"):
                    git_deps.append(line)
    return git_deps

setup(
    name="benchpush",
    version="0.1.1",
    license="MIT",
    description= "Benchmarking Non-prehensile Interactive Navigation",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url= "https://github.com/IvanIZ/BenchNPIN",
    install_requires= read_requirements("requirements.txt"),
    dependency_links=get_git_dependencies("requirements.txt"),
    python_requires=">=3.10",
    packages=find_packages() , 
    include_package_data=True,
    package_data={"benchpush": ["**/*.yaml"] , 
                  "benchpush": ["**/*.pk"]},
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
