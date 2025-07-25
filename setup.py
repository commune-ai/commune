#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup script for commune package
"""

from setuptools import setup, find_packages
import os
import sys

# Get the long description from README
with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

# Get version from commune package
sys.path.insert(0, os.path.abspath('.'))
try:
    from commune import __version__
    version = __version__
except ImportError:
    version = '0.1.0'  # Default version if import fails

# Core dependencies
install_requires = [
    'rich>=13.6.0',
    'fastapi>=0.115.13',
    'sse-starlette>=2.1,<2.3.7',
    'paramiko>=3.5.1',
    'nest_asyncio>=1.6.0',
    'uvicorn>=0.34.3',
    'scalecodec>=1.2.10,<1.3',
    'aiofiles>=24.1.0',
    'aiohttp>=3.12.13',
    'openai>=1.91.0',
    'torch>=2.7.1',
    'safetensors>=0.5.3',
    'msgpack_numpy>=0.4.8',
    'munch>=4.0.0',
    'netaddr>=1.3.0',
    'loguru>=0.7.3',
    'pyyaml>=6.0.2',
    'pandas>=2.3.0',
    'python-dotenv',
    'websocket-client>=0.57.0',
    'base58>=1.0.3',
    'certifi>=2019.3.9',
    'idna>=2.1.0',
    'requests>=2.21.0',
    'xxhash>=1.3.0',
    'ecdsa>=0.17.0',
    'eth-keys>=0.2.1',
    'eth_utils>=1.3.0',
    'pycryptodome>=3.11.0',
    'PyNaCl>=1.0.1',
    'py-sr25519-bindings>=0.2.0',
    'py-ed25519-zebra-bindings>=1.0',
    'py-bip39-bindings>=0.1.9',
    'psutil>=7.0.0',
]

# Optional dependencies
extra_requires = {
    'quality': [
        'black==22.3',
        'click==8.0.4',
        'isort>=5.5.4',
        'flake8>=3.8.3',
    ],
    'testing': [
        'pytest>=7.2.0',
    ],
}

# Add 'all' extra that includes everything
extra_requires['all'] = extra_requires['quality'] + extra_requires['testing']

setup(
    name='commune',
    version=version,
    description='Global toolbox that allows you to connect and share any tool (module)',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='developers',
    author_email='dev@communeai.org',
    url='https://communeai.org/',
    project_urls={
        'Homepage': 'https://communeai.org/',
        'Repository': 'https://github.com/commune-ai/commune',
        'Issues': 'https://github.com/commune-ai/commune/issues',
    },
    packages=find_packages(exclude=['tests*', 'docs*']),
    include_package_data=True,
    python_requires='>=3.8, <3.13',
    install_requires=install_requires,
    extras_require=extra_requires,
    entry_points={
        'console_scripts': [
            'c=commune:main',
        ],
    },
    keywords=['modular', 'sdk', 'machine learning', 'deep-learning', 'crypto'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
    license='MIT',  # Assuming MIT based on LICENSE file reference
    zip_safe=False,
)
