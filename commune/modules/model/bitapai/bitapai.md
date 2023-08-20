
# **BITAPAI**

## Setting a Bitapai Key

To set a Bitapai key, use the following command:

```bash
c model.bitapai add_api_key YOUR_API_KEY
```

Once this is done, you won't need to specify the key every time you start your module. The key will be saved in the module's folder at `.commune/model.bitapai/api_keys.json`.

To remove an API key, use:

```bash
c model.bitapai rm_api_key YOUR_API_KEY
```

To list all API keys:

```bash
c model.bitapai list_api_keys
```

## Serving the Model

To serve the model with a specific tag and API key, use the following command:

```bash
c model.bitapai serve tag=10 api_key=....
```

## Registering the Model

To register the model with a specific tag and name, use:

```bash
c model.bitapai register api_key=.... tag=10
```

This will set the name to `model.bitapai::10`.

## Testing the Model

To test the model, use:

```bash
c model.bitapai forward "whatup"
```

**Response:**
```
Hello! How can I assist you today? Is there anything in particular that you would like me to know about yourself or the task at hand? Let me help make your day more productive and stress-free. Just tell me what it is you need!
```
