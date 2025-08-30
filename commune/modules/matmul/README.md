# Matrix Multiplication Proof of Work (Matmul)

This repository contains a Proof of Work (PoW) system based on matrix multiplication, which can be utilized for computational tasks such as image processing.

## Overview

The Matmul system provides a way to generate computational challenges, compute proofs, and verify the results. It uses matrix multiplication as the core computational task and includes cryptographic signing and verification to ensure data integrity.

## Features

- Generate random matrix pairs for computational challenges
- Compute proofs by performing matrix multiplication
- Cryptographic signing of results for verification
- Verification of computed proofs
- Configurable difficulty level

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/matmul-pow.git
cd matmul-pow

# Install dependencies
pip install numpy msgpack msgpack-numpy
```

## Usage

### Basic Usage

```python
from matmul import Matmul

# Initialize with default difficulty (5)
pow_system = Matmul()

# Generate challenge parameters
matrix_a, matrix_b = pow_system.generate_params(size=64)

# Compute proof
proof = pow_system.compute_proof(matrix_a, matrix_b)

# Verify proof
is_valid = pow_system.verify_proof(proof)
print(f"Proof is valid: {is_valid}")
```

### Running a Test

```python
# Run a simple test
result = pow_system.test()
print(f"Test passed: {result['verified']}")
```

## API Reference

### `Matmul` Class

#### `__init__(difficulty=5)`
Initialize the PoW system with a specified difficulty.
- `difficulty`: The number of leading zeros required in the hash.

#### `generate_params(size=64)`
Generate two random matrices for the challenge.
- `size`: The size of the matrices (size x size).
- Returns: Tuple of two random matrices A and B.

#### `compute_proof(a, b, key=None)`
Perform the matrix multiplication and compute the hash.
- `a`: First matrix.
- `b`: Second matrix.
- `key`: Optional cryptographic key for signing.
- Returns: Dictionary containing parameters, result, timestamp, key address, and signature.

#### `verify_proof(proof, key=None)`
Verify that a given proof is valid.
- `proof`: The proof data to verify.
- `key`: Optional cryptographic key for verification.
- Returns: Boolean indicating if the proof is valid.

#### `hash_matrix(data)`
Hash a matrix using msgpack and the system's hash function.
- `data`: The numpy array to hash.
- Returns: Hash of the matrix.

#### `test()`
Run a simple test of the PoW system.
- Returns: Dictionary with verification status and proof.

## Dependencies

- NumPy: For matrix operations
- msgpack and msgpack-numpy: For serialization of numpy arrays
- Custom cryptographic module (referred to as 'c' in the code)

## License