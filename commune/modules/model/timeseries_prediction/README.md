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

You can install the required libraries using the following command:
```
pip install requests pandas scikit-learn tensorflow
```

## ETH to USDT Conversion Rate Prediction

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
python3 app.py
```

Make sure to update the `start_date` and `end_date` in the script to the desired time range for analysis.

## Functionality

- `fetch_sushiswap_data`: Fetches swap data within a specified time range.
- `preprocess_data`: Prepares the data for analysis by scaling the values.
- `build_transformer_model`: Defines a Transformer-based neural network for pattern recognition in time-series data.
- `main`: The main execution function that orchestrates data fetching, preprocessing, and machine learning model training.

## Model

The model used in this project is a simple Transformer, which is primarily designed to demonstrate the data flow and could be replaced with more sophisticated models for production use.

## Data

The data used in this project is fetched from SushiSwap's subgraph on The Graph and includes transaction information such as token amounts and USD values.

## Contributing

Contributions to the project are welcome. Please ensure to follow the code of conduct and submit pull requests for review.