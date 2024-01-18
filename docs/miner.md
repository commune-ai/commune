

## Deploying a Miner Tutorial

In this tutorial, you will learn how to deploy a validator on the network and perform various tasks related to staking, registration, and validation.

### Step 1: Registering a Miner

To register a validator with a specific tag, use the following CLI command:

```bash
c model.openai register tag=whadup subnet=commune
or 
c register model.openai::whadup subnet=commune
```

```python 
c.module('model.openai').register(tag='whadup', subnet=commune)
```
Result
```bash
{'success': True, 'name': 'model.openai::whadup', 'address': '38.140.133.234:50227'}
```


Check the logs 
Now see if the miner is running

```bash
c logs model.openai::whadup

```


```bash
369|model. | INFO:     Started server process [51470]
369|model. | INFO:     Waiting for application startup.
369|model. | INFO:     Application startup complete.
369|model. | INFO:     Uvicorn running on http://0.0.0.0:50227 (Press CTRL+C to quit)
```




