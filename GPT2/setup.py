from setuptools import setup, find_packages

setup(
    name="gpt2",
    version="0.1.0",
    description="GPT-2 implementation with power law scaling",
    author="AugmentedCat",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "regex>=2023.0.0",
        "matplotlib>=3.7.0",
        "transformers>=4.30.0",
    ],
    entry_points={
        "console_scripts": [
            "gpt2=gpt2.__main__:main",
        ],
    },
)
