
# **Stock**

## Setting a `polygon-io` Key

To set a `polygon-io` key, use the following command:

```bash
c model.stock add_api_key YOUR_API_KEY
```

Once this is done, you won't need to specify the key every time you start your module. The key will be saved in the module's folder at `.commune/model.stock/api_keys.json`.

To remove an API key, use:

```bash
c model.stock rm_api_key YOUR_API_KEY
```

To list all API keys:

```bash
c model.stock api_keys
```

## Serving the Model

To serve the model with a specific tag and API key, use the following command:

```bash
c model.stock serve tag=10 api_key=....
```

## Registering the Model

To register the model with a specific tag and name, use:

```bash
c model.stock register api_key=.... tag=10
```

This will set the name to `model.stock::10`.

## Testing the Model

To test the model, use:

```bash
c model.stock call ticker="AAPL" start="2023-01-01" end="2023-01-10"
```

**Response:**
```
{"ticker":"AAPL","queryCount":6,"resultsCount":6,"adjusted":true,"results":[{"v":1.12117471e+08,"vw":125.725,"o":130.28,"c":125.07,"h":130.9,"l":124.17,"t":1672722000000,"n":1021065},{"v":8.9100633e+07,"vw":126.6464,"o":126.89,"c":126.36,"h":128.6557,"l":125.08,"t":1672808400000,"n":770042},{"v":8.0716808e+07,"vw":126.0883,"o":127.13,"c":125.02,"h":127.77,"l":124.76,"t":1672894800000,"n":665458},{"v":8.7754715e+07,"vw":128.1982,"o":126.01,"c":129.62,"h":130.29,"l":124.89,"t":1672981200000,"n":711520},{"v":7.0790813e+07,"vw":131.6292,"o":130.465,"c":130.15,"h":133.41,"l":129.89,"t":1673240400000,"n":645365},{"v":6.3896155e+07,"vw":129.822,"o":130.26,"c":130.73,"h":131.2636,"l":128.12,"t":1673326800000,"n":554940}],"status":"OK","request_id":"c3824d40c540ba17d122a36a205c99a9","count":6}
```
