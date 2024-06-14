import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling1D, Dropout, MultiHeadAttention, LayerNormalization, LSTM
from tensorflow.keras.optimizers import Adam
import commune as c
from dotenv import load_dotenv
import os
from numpy.fft import rfft, irfft, rfftfreq
import matplotlib.pyplot as plt
import pywt
from sklearn.linear_model import LinearRegression

class SushiSwapDataAnalysis(c.Module):
    def __init__(self, start_date="2021-01-01", end_date="2023-01-01", input_shape=(1,1), model_type="transformer"):
        self.start_date = start_date
        self.end_date = end_date
        self.input_shape = input_shape
        self.model_type = model_type
        
    # Fetch data from SushiSwap using The Graph API
    def fetch_sushiswap_data(self, start_timestamp, end_timestamp):
        load_dotenv()
        api_key = os.getenv('THEGRAPH_API_KEY')
        url = f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/7h1x51fyT5KigAhXd8sdE3kzzxQDJxxz1y66LTFiC3mS"
        query = """
        query {
        swaps(
            where: {
            tokenIn: "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
            tokenOut: "0xdac17f958d2ee523a2206206994597c13d831ec7",
            timestamp_gte: %d,
            timestamp_lte: %d
            }
        ) {
            amountIn
            amountInUSD
            amountOut
            amountOutUSD
            timestamp
        }
        }
        """ % (start_timestamp, end_timestamp)
        
        response = requests.post(url, json={'query': query})
        data = response.json()['data']['swaps']
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('timestamp', inplace=True)
        return df

    # Preprocess the data
    def preprocess_data(self, df):
        # Ensure there are no NaN values
        df = df.dropna(subset=['amountInUSD', 'amountOutUSD'])
        df['amountOutUSD'] = pd.to_numeric(df['amountOutUSD'], errors='coerce')
        df['amountInUSD'] = pd.to_numeric(df['amountInUSD'], errors='coerce')
        df = df.dropna(subset=['amountInUSD', 'amountOutUSD'])
        
        # Calculate the conversion rate
        df['conversion_rate'] = df['amountOutUSD'] / df['amountInUSD']
        
        # Scale the 'amountInUSD' feature
        scaler = MinMaxScaler()
        df['scaled_amountInUSD'] = scaler.fit_transform(df[['amountInUSD']])
        
        return df, scaler

    # Define the Transformer model
    def build_transformer_model(self, input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
        inputs = Input(shape=input_shape)
        x = inputs
        for _ in range(num_transformer_blocks):
            x = LayerNormalization(epsilon=1e-6)(x)
            x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
            x = Dropout(dropout)(x)
        x = GlobalAveragePooling1D(data_format='channels_first')(x)
        for dim in mlp_units:
            x = Dense(dim, activation='relu')(x)
            x = Dropout(mlp_dropout)(x)
        outputs = Dense(1)(x)
        return Model(inputs, outputs)

    def build_lstm_model(self, input_shape):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        return model
    
    def build_fourier_model(self, df, n_predict, n_harmonics=5):
        """
        Perform Fourier-based time series prediction.
        """
        y = df['conversion_rate'].values
        n = len(y)
        f = rfft(y)
        frequencies = rfftfreq(n)

        indices = np.argsort(np.abs(f))[:-n_harmonics]
        f[indices] = 0

        t = np.arange(0, n + n_predict)
        reconstructed = irfft(f, n=t.size)

        return reconstructed[-n_predict:]
    
    def build_wavelet_model(self, df, n_predict):
        """
        Perform Wavelet-based time series prediction.
        """
        # Wavelet transform for feature extraction
        coeffs = pywt.wavedec(df['conversion_rate'].values, 'db1', level=2)
        cA2, cD2, cD1 = coeffs

        min_len = min(len(cA2), len(df['conversion_rate']))
        X = cA2[:min_len]
        y = df['conversion_rate'].values[-min_len:]

        # Fit a simple linear model
        model = LinearRegression()
        model.fit(X.reshape(-1, 1), y)

        # Predict future values
        predicted = model.predict(X[-n_predict:].reshape(-1, 1))

        return predicted
    
    # Main execution function
    def main(self, model_type = "transformer"):
        # Fetch the data
        start_timestamp = int(pd.Timestamp(self.start_date).timestamp())
        end_timestamp = int(pd.Timestamp(self.end_date).timestamp())
        self.model_type = model_type
        df = self.fetch_sushiswap_data(start_timestamp, end_timestamp)
        df, scaler = self.preprocess_data(df)

        if model_type == "transformer" or model_type == "LSTM":
            valid_rows = ~df['conversion_rate'].isnull()
            X = df.loc[valid_rows, 'scaled_amountInUSD'].values.reshape(-1, 1)
            y = df.loc[valid_rows, 'conversion_rate'].values

            assert X.shape[0] == y.shape[0], "The number of samples in X and y must be the same"
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if self.model_type == "transformer":
                model = self.build_transformer_model(
                    input_shape=self.input_shape,
                    head_size=256,
                    num_heads=4,
                    ff_dim=4,

                    num_transformer_blocks=4,
                    mlp_units=[128],
                    dropout=0.1,
                    mlp_dropout=0.1
                )
            elif self.model_type == "LSTM":
                model = self.build_lstm_model((X_train.shape[1], 1))


            model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=1e-4))
            X_train = X_train.reshape((-1, self.input_shape[0], self.input_shape[1]))
            X_test = X_test.reshape((-1, self.input_shape[0], self.input_shape[1]))
            model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)
            predictions_scaled = model.predict(X_test)
            predictions = scaler.inverse_transform(predictions_scaled)

        elif model_type == "fourier":
            n_predict = 20  
            predictions = self.build_fourier_model(df, n_predict)
            
        elif model_type == "wavelet":
            n_predict = 20
            predictions = self.build_wavelet_model(df, n_predict)
            
        print(predictions)