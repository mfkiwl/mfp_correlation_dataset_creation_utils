import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mfp_correlation_data_utils",
    version="0.0.1",
    author="Miguel Pereira",
    author_email="",
    description="A package to facilitate the creation of the datasets required\
                for training and testing classifier the models used in the paper:\
                'Classification of Line of Sight and Non-Line of Sight GNSS\
                Signals for Smartphone Positioning in Urban Environments' by\
                Miguel Pereira and Dr Anahid Basiri",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    project_urls={},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL-3.0-or-later",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
