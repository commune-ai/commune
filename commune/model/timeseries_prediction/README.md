# SushiSwap Data Analysis and Prediction

This project fetches and analyzes swap data from the SushiSwap decentralized exchange using The Graph's GraphQL API. It includes a machine learning model that processes the fetched data and attempts to identify patterns or insights.


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Functionality](#functionality)
- [Model](#model)
- [Data](#data)
- [Contributing](#contributing)

## Installation

To run this project, you will need Python and the following libraries:

- `requests`
- `pandas`
- `scikit-learn`
- `TensorFlow`
- `pywavelets`


You can install the required libraries using the following command:
```
pip install requests pandas scikit-learn tensorflow pywavelets
```

## WETH to USDT Conversion Rate Prediction

This project is focused on predicting the conversion rate between Ethereum (ETH) and Tether (USDT). We use historical data of ETH and USDT prices, fetched from The Graph's API, to train a time series prediction model.

The model takes into account various factors such as past price trends, volume changes, and market sentiment to predict future conversion rates. This can be useful for traders and investors who want to understand potential future price movements and make informed decisions.

The project uses Python for data fetching, preprocessing, and model training. We use the `python-dotenv` library to securely handle API keys and other sensitive information.

Follow the instructions in this README to set up and run the project. Please note that you will need your own API key from The Graph to fetch the data.


## Environment Variables

This project uses environment variables to keep sensitive information like API keys secure. Follow the steps below to use the `.env` file:

1. Create a new file in the project root directory and name it `.env`.
2. Inside this file, add your API key like this: `THEGRAPH_API_KEY=your_api_key_here`.
3. Save and close the file. The application will now use this key to make requests to The Graph's API.

## Usage

Run the script using the following command:
```
c model.timeseries_prediction main
```
Parameters can be provide among LSTM, fourier, wavelet being transformer as default.
Below is an example for fourier, 
```
c model.timeseries_prediction main model_type="fourier"
```
Make sure to update the `start_date` and `end_date` in the script to the desired time range for analysis.

## Functionality

- `fetch_sushiswap_data`: Fetches swap data within a specified time range.
- `preprocess_data`: Prepares the data for analysis by scaling the values.
- `build_transformer_model`: Defines a Transformer-based neural network for pattern recognition in time-series data.
- `build_lstm_model`: Defines a LSTM model.
- `build_fourier_model`: Defines a Fourier model
- `build_wavelet_model`: Defines a Wavelet model
- `main`: The main execution function that orchestrates data fetching, preprocessing, and machine learning model training.

## Model

This project uses several models to predict the conversion rate between Ethereum (ETH) and Tether (USDT):

- **Transformer Model**: This model uses Transformer module, with their self-attention mechanism, to predict future conversion rates based on past data.

- **LSTM Model**: This model uses Long Short-Term Memory units, a type of recurrent neural network, to predict future conversion rates based on past data.

- **Wavelet Model**: This model uses wavelet transformations to analyze the frequency components of the time series data and predict future conversion rates.

- **Fourier Model**: This model uses Fourier transformations to analyze the frequency components of the time series data and predict future conversion rates.

## Data

The data used in this project is fetched from SushiSwap's subgraph on The Graph and includes transaction information such as token amounts and USD values.

## Contributing

Contributions to the project are welcome. Please ensure to follow the code of conduct and submit pull requests for review.