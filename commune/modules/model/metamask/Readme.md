# MetaMask Module

This module provides an interface for connecting to Ethereum blockchain using MetaMask. It includes a Streamlit application for displaying Ethereum transactions.


## Table of Contents

- [Description](#description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)


## Description

This module leverages the power of [Web3 Streamlit Components] to provide a seamless experience for interacting with the Ethereum blockchain.


## Features

- **Ethereum Connection**: Facilitates a secure connection to the Ethereum blockchain using the MetaMask wallet.
- **Transaction Display**: Showcases Ethereum transactions within a Streamlit application, enabling real-time monitoring and interaction.
- **Uniswap Integration**: Allows users to perform token swaps via Uniswap directly within the application (optional feature based on your implementation). (This is yet to be implemented)

## Installation

1. Set up your environment variables in a `.env` file. An Ethereum endpoint is required.
2. Infura key is provided for testing purposes.
3. Run the following command: `pip install streamlit-wallet-connect`


## Usage

To run the module, use the following command:

```
c model.metamask run
```

One could be able to see  the following message in console:
  
  "You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
  ...
  " 
  
The browser should open automatically.

## Contributing

We welcome contributions! Please feel free to submit a pull request or open an issue if you have suggestions or find a bug.

## License

This project is licensed under the MIT License - see the LICENSE file for details.