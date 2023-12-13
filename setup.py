from distutils.core import setup

from setuptools import find_packages

setup(
    name="mindcraft",
    version="0.1rc1",
    packages=find_packages(),
    package_data={p: ["*"] for p in find_packages()},
    url="",
    license="MIT",
    install_requires=[
        "transformers==4.35.2",
        "sentence_transformers==2.2.2",
        "autoawq==0.1.7",
        "requests~=2.31.0",
        "torch==2.1.1"
    ],
    python_requires=">=3.10.0",
    author="Juan.Martinez",
    author_email="jjmcarrascosa@gmail.com",
    description="Mindcraft: an LLM-based engine for creating real NPCs",
)