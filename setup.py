import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="citychef",
    version="0.0.1",
    author="Arup City Modelling Lab",
    author_email="citymodelling@arup.com",
    description="Randomly generate a city, and output statistical data about it.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fredshone/citychef",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)