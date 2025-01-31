from setuptools import setup, find_packages

setup(
    name="commune",
    version="0.0.1",
    description="Global toolbox that allows you to connect and verify your tools",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    author="Commune AI Organization",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'commune': ['*.py'],
    },
    entry_points={
        'console_scripts': [
            'commune=commune.cli:main',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">3.9",
)
