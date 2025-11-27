from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ExplainableML",
    version="0.1.0",
    author="Huy Huynh",
    author_email="huy.huynh@example.com",
    description="Explainble ML",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/huytjuh/ExplainableML",
    packages=find_packages(where="src", include=["src*", "src.*"]),
    package_dir={"": "src"},

    install_requires=[
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.3.0",
    ],

    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_data={"src": ["data/*.pkl"]}
)

