from setuptools import setup, find_packages

setup(
    name="vibe2",
    version="0.1.0",
    description="A module for generating, managing, and visualizing dope vibes",
    author="Commune AI",
    author_email="info@commune.ai",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'vibe2=vibe2.cli:main',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)