from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="src",
    version="0.0.1",
    author="Pratik",
    description="Package of ANN Implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PratikBorkar04/mnist_ann",
    author_email="pratikab01gmail.com",
    packages=["src"],
    python_requires=">=3.8",
    install_requires=[
        "tensorflow",
        "matplotlib",
        "seaborn",
        "numpy",
        "pandas"
    ]
)