from setuptools import find_packages, setup

setup(
    name="mod-net-client",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "substrate-interface>=1.7.0",
        "ipfshttpclient==0.8.0a2",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.4.0",
    ],
)
