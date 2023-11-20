import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling1D, Dropout, MultiHeadAttention, LayerNormalization
from tensorflow.keras.optimizers import Adam
import commune as c
from dotenv import load_dotenv
import os

class SushiSwapDataAnalysis(c.Module):
    def __init__(self, start_date="2021-01-01", end_date="2023-01-01", input_shape=(1,2)):
        self.start_date = start_date
        self.end_date = end_date
        self.input_shape = input_shape
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
        df['amountOutUSD'] = pd.to_numeric(df['amountOutUSD'])
        df['amountInUSD'] = pd.to_numeric(df['amountInUSD'])
        df['conversion_rate'] = df['amountOutUSD'] / df['amountInUSD']
        scaler = MinMaxScaler()
        df[['amountInUSD']] = scaler.fit_transform(df[['amountInUSD']])
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

    # Main execution function
    def main(self):
        # Fetch the data
        start_timestamp = int(pd.Timestamp(self.start_date).timestamp())
        end_timestamp = int(pd.Timestamp(self.end_date).timestamp())
        df = self.fetch_sushiswap_data(start_timestamp, end_timestamp)
        df, scaler = self.preprocess_data(df)

        # Split the data into features and target
        X = df[['amountInUSD']].values
        y = df['conversion_rate'].values # This is a placeholder target variable

        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define the model
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
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=1e-4))
        
        # Reshape data for the transformer
        X_train = X_train.reshape((-1, self.input_shape[0], self.input_shape[1]))
        X_test = X_test.reshape((-1, self.input_shape[0], self.input_shape[1]))
        
        # Train the model
        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=10,
            batch_size=32
        )

        # Make predictions (as an example, here we use the test set)
        predictions_scaled = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions_scaled)
        print(predictions)