
# **DALL-E**

## Setting a DALL-E(OpenAI) Key

To set a [Dall-E](https://platform.openai.com/docs/guides/images) key, use the following command:

```bash
c model.dalle add_api_key YOUR_API_KEY
```

Once this is done, you won't need to specify the key every time you start your module. The key will be saved in the module's folder at `.commune/model.dalle/api_keys.json`.

To remove an API key, use:

```bash
c model.dalle rm_api_key YOUR_API_KEY
```

To list all API keys:

```bash
c model.dalle api_keys
```

## Serving the Model

To serve the model with a specific tag and API key, use the following command:

```bash
c model.dalle serve tag=10 api_key=....
```

## Registering the Model

To register the model with a specific tag and name, use:

```bash
c model.dalle register api_key=.... tag=10
```

This will set the name to `model.dalle::10`.

## Testing the Model

To test the model, use:

### Create image

```bash
c model.dalle generate prompt="a white siamese cat"
```

### Create image edit

```bash
c model.dalle edit prompt="A sunlit indoor lounge area with a pool containing a flamingo" image="image.png" mask="mask.png"
```

### Create image variation

```bash
c model.dalle variation image="image.png"
```

### Test with gradio

```bash
c model.dalle gradio
```