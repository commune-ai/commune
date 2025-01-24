
from setuptools import setup, find_packages
from os import path
from io import open
from pkg_resources import parse_requirements
here = path.abspath(path.dirname(__file__))
repo = 'commune'  # name of the package is assumed to be the name of the directory
with open(f'{here}/README.md', encoding='utf-8') as f:
    long_description = f.read()
with open(f'{here}/requirements.txt') as requirements_file:
    install_requires = [str(requirement) for requirement in parse_requirements(requirements_file)]
setup(
    name=repo,
    version='1.0.0',
    description='a way for connecting and verifying tools for the global toolbox',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/commune-ai/commune',
    author='physics',
    packages=find_packages(),
    include_package_data=True,
    author_email='',
    license='IDGAF License, Do What You Want, I wont sue you',
    install_requires=install_requires,
    entry_points={'console_scripts': [f'c={repo}.cli:main']},
    classifiers=['FUCK SHIT UP'], 
    python_requires='>=3.8')