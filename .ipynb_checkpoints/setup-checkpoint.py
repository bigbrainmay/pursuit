from setuptools import setup, find_packages

setup(
    name = "pursuit",
    version= "0.0.1",
    author_email= "bigbrainmay@gmail.com",
    description= "tools for reading pursuit expriment data",
    url = "https://github.com/bigbrainmay/pursuit",
    packages = find_packages(),
    install_requires = [
        line.strip() for line in open("requirements.txt", "r") if line.strip() and not line.startswith("#")
    ],
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires = ">=3.6",

)