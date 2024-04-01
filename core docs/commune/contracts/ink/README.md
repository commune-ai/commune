# ink! Smart Contract Code Examples 

Welcome to the ink! code examples directory, your home for learning to build smart contracts with ink!. These examples will provide you with a robust understanding of how to use ink! to develop Substrate smart contracts. 

## Getting Started 

Before diving in, you will need to install [`cargo-contract`](https://github.com/paritytech/cargo-contract) on your system. `cargo-contact` is essential for compiling the Rust-based ink! smart contracts into WebAssembly (WASM).

You can install `cargo-contract` by using the following command:

```
cargo install cargo-contract --force
```

Please note that the `--force` flag ensures that the latest version of `cargo-contract` is installed.

## Building a Sample Contract and Extracting Metadata

To build any of our sample contracts and create the contract's Wasm file, navigate to the contract's root directory and run the following command:

`cargo contract build`

If everything runs successfully, you should your contract's optimized `<contract-name>.wasm` file and `metadata.json` file in the `target` directory of your contract's root. As a comprehensive package, a `<contract-name>.contract` file will also be created which combines the Wasm and metadata into one file that is used for deploying the contract.

## Explore and Learn!

Each example is a stepping stone to becoming proficient with ink!. Enjoy exploring the potential of blockchain development with these ink! code examples.
