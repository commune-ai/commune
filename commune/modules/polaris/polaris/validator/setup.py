from setuptools import setup, find_packages

setup(
    name="validator_node",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "typer",
        "paramiko",
        "requests",
        "loguru",
        "python-dotenv",
    ],
)
