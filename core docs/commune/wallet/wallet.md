# Project - Commune Wallet

**Overview:**
The aim of this project is to provide interfaces for the Commune Wallet to execute various operations such as transfers, payment, stakes and more.

This project comprises of the Wallet class with several methods that provide an interactive User Interface to perform transactions using the Commune wallet.

## Requirements
- Python 3.6 and above
- Streamlit
- Commune
- Plotly
- Pandas

## Usage

Import the Wallet from the package and make use of the various methods available to execute transactions through the Commune Wallet.

**Dashboard:**
The 'dashboard' function initializes Streamlit and provides an interactive interface to the user for all available operations on the wallet.

**Transfer operation:**
'transfer_dashboard' function of this code provides an interface to make transfer of a specified amount to one or multiple addresses.

**Module Dashboard:**
'module_dashboard' function allows user to search the namespaces and collects details for modules that match the search criteria

**Stake Operation:**
The 'stake_dashboard' allows users to stake currencies through the UI.

**Staking Plot:**
'staking_plot' function provides the user with a visual representation of their current stakes.

**Unstake Operation:**
The 'unstake_dashboard' provides an interface to remove stake from one or multiple modules.

**Registration:**
The 'register_dashboard' function provides the user with an interface for registering a module.

**User Information:**
The 'my_info' function displays the user's public address and stake information.

Run the code by cloning this code repository to your local machine, navigate to your preferred terminal, install the necessary dependencies with:
`pip install -r requirements.txt`
Then run the code with
`python main.py`

## Disclaimer:
This program handles monetary transactions, usage of this program should be done with caution. Developers do not take responsibility for any financial loss.