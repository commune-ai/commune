
from setuptools import setup, find_packages
from os import path
from io import open
from pkg_resources import parse_requirements

libname = 'eth'

here = path.abspath(path.dirname(__file__))

with open(f'{here}/README.md', encoding='utf-8') as f:
    long_description = f.read()

with open(f'{here}/requirements.txt') as requirements_file:
    install_requires = [str(requirement) for requirement in parse_requirements(requirements_file)]

setup(
    name=libname,
    version='0.0.1',
    description='A simple CLI tool to help you manage your projects and tasks and connecting all of them together in a simple way',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/commune-ai/eth',
    author='physics',
    packages=find_packages(),
    include_package_data=True,
    author_email='',
    license='AGIDOESNTCAREABOUTYOURLISCENCES',
    install_requires=install_requires,
    entry_points={
        'console_scripts': [
            'eth=eth.cli:main'
        ],
    },
    classifiers=[
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        "AGIDOESNTCAREABOUTYOURLISCENCES"
        # Pick your license as you wish
        'License :: IDGAF License, Do What You Want, I wont sue you',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.8',
    ], 
    python_requires='>=3.8')