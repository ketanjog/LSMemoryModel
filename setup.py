import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="LSMemoryModel",
    version="0.1-dev",
    author="Ketan Jog",
    author_email="kj2473@columbia.edu",
    description="Codebase to explore online learning algorithms for memory consolidation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ketanjog/LSMemoryModel",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)
