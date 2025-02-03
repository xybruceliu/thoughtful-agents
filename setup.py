"""Setup file for the thoughtful_agents package."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="thoughtful_agents",
    version="0.1.0",
    author="Bruce Liu",
    author_email="xingyuliu@ucla.edu",
    description="A Python framework for building proactive LLM agents that simulate human-like cognitive processes. Enables agents to continuously generate and evaluate thoughts in parallel with conversations, autonomously determining when and how to engage.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/liubruce/thoughtful-agents",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "faiss-cpu>=1.7.0",
        "openai>=1.0.0",
        "pytest>=6.0.0",
        "pytest-asyncio>=0.14.0",
    ],
    extras_require={
        "dev": [
            "black",
            "flake8",
            "isort",
            "mypy",
            "pytest-cov",
        ],
    },
) 