
from setuptools import setup, find_packages
from pkg_resources import parse_requirements
from os import path
from io import open
import codecs
import re
import os

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
    
with open('requirements.txt') as requirements_file:
    install_requires = [str(requirement) for requirement in parse_requirements(requirements_file)]

setup(
    name='commune',
    version='0.0.0',
    description='commune',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/commune-ai/commune',
    author='commune',
    packages=find_packages(),
    include_package_data=True,
    author_email='',
    license='MIT',
    install_requires=install_requires,
    scripts=[],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    python_requires='>=3.7',
    extras_requires={
        'cubit': ['cubit>=1.1.0 @ git+https://github.com/opentensor/cubit.git']
    }
)
